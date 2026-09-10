import sys
import os
sys.path.insert(0, os.getcwd())

import glob
import statistics
import torch
import torch.nn.functional as F

from option import args_parser
from federated_main import set_seed
from models import adcol_model
from util import prepare_data_PACS_multi_clients


sys.argv = ["probe_lambda_G_pacs"]
args = args_parser()
args.exp = 1
args.mode = "ours"
args.lr = 0.01
args.number_workers = 0
args.dataset = "PACS"
args.num_classes = 7
args.size = 64
args.batch = 32
args.wk_iters = 10
args.adcol_mu = 0.1
args.adcol_beta = 0.1
args.adcol_epoch = 3
args.domain_keyed_proto = True
args.device = args.device if torch.cuda.is_available() else "cpu"
args.seed = 0
set_seed(args)

train_loader_list, client_domains, test_loader_list = prepare_data_PACS_multi_clients(args)
args.num_users = len(train_loader_list)
datasets_name = []
for d in client_domains:
    if d not in datasets_name:
        datasets_name.append(d)
domain_to_id = {d: i for i, d in enumerate(datasets_name)}
client_domain_ids = [domain_to_id[d] for d in client_domains]
print(f"domain_to_id = {domain_to_id}")

CKPT_DIR = "checkpoints_pacs_calib"
ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "seed0_checkpoint_round*.pt")))
assert len(ckpt_files) > 0, (
    f"no checkpoints found under {CKPT_DIR}/ -- download the 4 round4/9/14/19 snapshots from "
    f"calib_run_for_lambda_G_pacs.py's Kaggle output and place them in this folder before running."
)
print(f"probing {len(ckpt_files)} checkpoint snapshot(s): {ckpt_files}")

NUM_PROBE_BATCHES = 3

records = {d: {"g_D": [], "g_G": [], "cos": []} for d in datasets_name}


def flat_grad(params):
    return torch.cat([p.grad.detach().reshape(-1) for p in params if p.grad is not None])


for ckpt_path in ckpt_files:
    print(f"\n=== {ckpt_path} ===")
    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    global_proto = checkpoint["global_proto"]
    global_G = checkpoint.get("prev_G_state", None)
    local_states = checkpoint["local_models_state"]

    for client_idx in range(args.num_users):
        domain = client_domains[client_idx]
        my_domain_id = client_domain_ids[client_idx]
        client_global_proto = {
            cls: global_proto[(cls, my_domain_id)]
            for cls in range(args.num_classes)
            if (cls, my_domain_id) in global_proto
        }
        if len(client_global_proto) == 0 or global_G is None or len(global_G) == 0:
            print(f"  {domain}: skipped (empty proto or G_k not yet available at this checkpoint)")
            continue

        model = adcol_model(num_classes=args.num_classes).to(args.device)
        model.load_state_dict(local_states[client_idx])
        model.train()

        loader = train_loader_list[client_idx]
        it = iter(loader)
        n_done = 0
        while n_done < NUM_PROBE_BATCHES:
            try:
                images, labels = next(it)
            except StopIteration:
                break
            images, labels = images.to(args.device).float(), labels.to(args.device).long()

            rep, logits = model(images)

            proto_dim = next(iter(client_global_proto.values())).shape[0]
            proto_matrix = torch.zeros((args.num_classes, proto_dim), device=args.device)
            present_mask = torch.zeros(args.num_classes, dtype=torch.bool, device=args.device)
            for label, proto in client_global_proto.items():
                proto_matrix[label] = proto
                present_mask[label] = True
            sample_valid = present_mask[labels]
            if not sample_valid.any():
                continue
            rep_normalized = F.normalize(rep, dim=1)
            proto_matrix_normalized = F.normalize(proto_matrix, dim=1)
            C_y = proto_matrix_normalized[labels]
            numerator = torch.exp((rep_normalized * C_y).sum(dim=1) / args.T)
            sim = (rep_normalized @ proto_matrix.T) / args.T
            sim = sim.masked_fill(~present_mask.unsqueeze(0), float("-inf"))
            denominator = torch.exp(sim).sum(dim=1)
            loss1 = (-torch.log(numerator / denominator))[sample_valid].mean()

            G_dim = next(iter(global_G.values())).shape[0]
            G_matrix = torch.zeros((args.num_classes, G_dim), device=args.device)
            G_present_mask = torch.zeros(args.num_classes, dtype=torch.bool, device=args.device)
            for label, proto in global_G.items():
                G_matrix[label] = proto
                G_present_mask[label] = True
            G_sample_valid = G_present_mask[labels]
            if not G_sample_valid.any():
                continue
            rep_normalized_G = F.normalize(rep, dim=1)
            G_matrix_normalized = F.normalize(G_matrix, dim=1)
            C_y_G = G_matrix_normalized[labels]
            numerator_G = torch.exp((rep_normalized_G * C_y_G).sum(dim=1) / args.T)
            sim_G = (rep_normalized_G @ G_matrix.T) / args.T
            sim_G = sim_G.masked_fill(~G_present_mask.unsqueeze(0), float("-inf"))
            denominator_G = torch.exp(sim_G).sum(dim=1)
            loss_cross = (-torch.log(numerator_G / denominator_G))[G_sample_valid].mean()

            feat_params = list(model.features.parameters())

            model.zero_grad()
            loss1.backward(retain_graph=True)
            g_D = flat_grad(feat_params)

            model.zero_grad()
            loss_cross.backward()
            g_G = flat_grad(feat_params)

            cos = F.cosine_similarity(g_D.unsqueeze(0), g_G.unsqueeze(0)).item()
            records[domain]["g_D"].append(g_D.norm().item())
            records[domain]["g_G"].append(g_G.norm().item())
            records[domain]["cos"].append(cos)

            model.zero_grad()
            n_done += 1

        print(f"  {domain}: collected {n_done} probe batch(es)")

print("\n=== per-domain median gradient magnitudes / cosine (all checkpoints pooled) ===")
rho_list = [0.25, 0.5, 1.0]
header = f"{'domain':14s} {'med_gD':>10s} {'med_gG':>10s} {'med_cos':>10s} {'n_probes':>9s}"
for rho in rho_list:
    header += f"  lambda@{rho:<4}"
print(header)

ratios = {}
for d in datasets_name:
    gd, gg, cs = records[d]["g_D"], records[d]["g_G"], records[d]["cos"]
    if len(gd) == 0:
        print(f"{d:14s} {'--':>10s} {'--':>10s} {'--':>10s} {0:>9d}  (no valid probes)")
        continue
    med_gd, med_gg, med_cs = statistics.median(gd), statistics.median(gg), statistics.median(cs)
    r_d = med_gd / med_gg
    ratios[d] = r_d
    row = f"{d:14s} {med_gd:10.4f} {med_gg:10.4f} {med_cs:10.4f} {len(gd):>9d}"
    for rho in rho_list:
        lam = rho * args.adcol_mu * r_d
        row += f"  {lam:10.4f}"
    print(row)

if len(ratios) == 0:
    print("\nno domain produced valid probes -- cannot recommend lambda_G. check checkpoint contents.")
else:
    anchor_domain = min(ratios, key=ratios.get)
    r_anchor = ratios[anchor_domain]
    print(f"\nanchor domain (min g_D/g_G ratio, most cross-sensitive) = {anchor_domain}  (r = {r_anchor:.4f})")
    print("recommended lambda_G candidates (rho * adcol_mu * r_anchor), conservative envelope for ALL domains:")
    for rho in rho_list:
        lam = rho * args.adcol_mu * r_anchor
        print(f"  rho={rho}: lambda_G = {lam:.4f}")

    for d in datasets_name:
        cs = records[d]["cos"]
        if len(cs) == 0:
            continue
        med_cs = statistics.median(cs)
        if med_cs < -0.1:
            print(f"\n[WARNING] {d}: median cos(g_D, g_G) = {med_cs:.4f} -- notably negative, "
                  f"loss1 and loss_cross may be pulling in conflicting directions on this domain. "
                  f"Diagnostic flag only, not used to adjust lambda_G -- reconsider before starting D "
                  f"if this looks severe.")

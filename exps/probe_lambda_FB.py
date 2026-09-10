import sys
import os
sys.path.insert(0, os.getcwd())

import glob
import re
import statistics
import torch
import torch.nn.functional as F

from option import args_parser
from federated_main import set_seed
from models import adcol_model
from util import prepare_data_office_multi_clients


sys.argv = ["probe_lambda_FB"]
args = args_parser()
args.exp = 1
args.mode = "ours"
args.lr = 0.01
args.number_workers = 0
args.dataset = "office"
args.num_classes = 10
args.size = 64
args.batch = 32
args.wk_iters = 10
args.adcol_mu = 0.1
args.adcol_beta = 0.1
args.adcol_epoch = 3
args.domain_keyed_proto = True
args.no_discriminator_fix = False
args.device = args.device if torch.cuda.is_available() else "cpu"
args.seed = 0
set_seed(args)

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=4)
args.num_users = len(train_loader_list)
datasets_name = []
for d in client_domains:
    if d not in datasets_name:
        datasets_name.append(d)
domain_to_id = {d: i for i, d in enumerate(datasets_name)}
client_domain_ids = [domain_to_id[d] for d in client_domains]
dslr_client_indices = [i for i, d in enumerate(client_domains) if d == "dslr"]
print(f"domain_to_id = {domain_to_id}")
print(f"dslr client indices = {dslr_client_indices}")

dslr_client_sizes = {i: len(train_loader_list[i].sampler.indices) for i in dslr_client_indices}
total_dslr = sum(dslr_client_sizes.values())
expected_bank_size = {i: total_dslr - n for i, n in dslr_client_sizes.items()}
print(f"dslr client sizes = {dslr_client_sizes}, expected peer-bank sizes = {expected_bank_size}")

CKPT_DIR = "calib_lambda_FB_run"
ckpt_files = sorted(
    glob.glob(os.path.join(CKPT_DIR, "seed0_checkpoint_round*.pt")),
    key=lambda p: int(re.search(r"round(\d+)\.pt$", p).group(1)),
)
EXPECTED_ROUNDS = [4, 9, 14, 19]
found_rounds = [int(re.search(r"round(\d+)\.pt$", p).group(1)) for p in ckpt_files]
assert found_rounds == EXPECTED_ROUNDS, (
    f"expected exactly checkpoints at rounds {EXPECTED_ROUNDS}, found {found_rounds} under "
    f"{CKPT_DIR}/ -- run calib_run_for_lambda_FB.py first (or check for a partial/stale run)."
)
print(f"probing {len(ckpt_files)} checkpoint snapshot(s): {ckpt_files}")

NUM_PROBE_BATCHES = 3

records = {c: {"g_loss1": [], "g_FB": [], "cos": []} for c in dslr_client_indices}


def flat_grad(params):
    return torch.cat([p.grad.detach().reshape(-1) for p in params if p.grad is not None])


for ckpt_path in ckpt_files:
    print(f"\n=== {ckpt_path} ===")
    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    global_proto = checkpoint["global_proto"]
    feature_bank_prev = checkpoint.get("feature_bank_prev", {})
    local_states = checkpoint["local_models_state"]

    dslr_bank = feature_bank_prev.get("dslr", {})
    for client_idx in dslr_client_indices:
        my_domain_id = client_domain_ids[client_idx]
        client_global_proto = {
            cls: global_proto[(cls, my_domain_id)]
            for cls in range(args.num_classes)
            if (cls, my_domain_id) in global_proto
        }
        peer_items = [(feat, lab) for j, (feat, lab) in dslr_bank.items() if j != client_idx]
        if len(client_global_proto) == 0 or not peer_items:
            print(f"  client {client_idx}: skipped (empty proto or no peer bank at this checkpoint)")
            continue

        bank_features = torch.cat([feat for feat, _ in peer_items], dim=0)
        bank_labels = torch.cat([lab for _, lab in peer_items], dim=0)
        assert bank_features.shape[0] == expected_bank_size[client_idx], (
            f"client {client_idx}: peer bank has {bank_features.shape[0]} rows, "
            f"expected {expected_bank_size[client_idx]} -- peer exclusion or bank population is wrong."
        )
        assert not bank_features.requires_grad, (
            f"client {client_idx}: bank_features.requires_grad=True -- detach discipline violated "
            f"somewhere upstream, this would let gradient leak into peer clients' encoders."
        )
        bank_features = bank_features.detach().to(args.device)
        bank_labels = bank_labels.detach().to(args.device)
        bank_features_n = F.normalize(bank_features, dim=1)

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
                it = iter(loader)
                continue
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

            rep_n_fb = F.normalize(rep, dim=1)
            sim_fb = (rep_n_fb @ bank_features_n.T) / args.T
            pos_mask_fb = labels.unsqueeze(1).eq(bank_labels.unsqueeze(0))
            num_pos_fb = pos_mask_fb.sum(dim=1)
            has_pos_fb = num_pos_fb > 0
            if not has_pos_fb.any():
                continue
            log_prob_fb = sim_fb - torch.logsumexp(sim_fb, dim=1, keepdim=True)
            pos_log_prob_fb = (
                (log_prob_fb * pos_mask_fb.float()).sum(dim=1) / num_pos_fb.clamp_min(1).float()
            )
            loss_fb = -pos_log_prob_fb[has_pos_fb].mean()

            feat_params = list(model.features.parameters())

            model.zero_grad()
            loss1.backward(retain_graph=True)
            g_loss1 = flat_grad(feat_params)

            model.zero_grad()
            loss_fb.backward()
            g_FB = flat_grad(feat_params)

            cos = F.cosine_similarity(g_loss1.unsqueeze(0), g_FB.unsqueeze(0)).item()
            records[client_idx]["g_loss1"].append(g_loss1.norm().item())
            records[client_idx]["g_FB"].append(g_FB.norm().item())
            records[client_idx]["cos"].append(cos)

            model.zero_grad()
            n_done += 1

        print(f"  client {client_idx}: collected {n_done} probe batch(es)")

print("\n=== per-client median gradient magnitudes / cosine (all checkpoints pooled) ===")
rho_list = [0.25, 0.5, 1.0]
header = f"{'client':8s} {'med_gloss1':>12s} {'med_gFB':>10s} {'med_cos':>10s} {'n_probes':>9s}"
for rho in rho_list:
    header += f"  lambda@{rho:<4}"
print(header)

ratios = {}
for c in dslr_client_indices:
    g1, gf, cs = records[c]["g_loss1"], records[c]["g_FB"], records[c]["cos"]
    if len(g1) == 0:
        print(f"{c:<8d} {'--':>12s} {'--':>10s} {'--':>10s} {0:>9d}  (no valid probes)")
        continue
    med_g1, med_gf, med_cs = statistics.median(g1), statistics.median(gf), statistics.median(cs)
    r_c = med_g1 / med_gf
    ratios[c] = r_c
    row = f"{c:<8d} {med_g1:12.4f} {med_gf:10.4f} {med_cs:10.4f} {len(g1):>9d}"
    for rho in rho_list:
        lam = rho * args.adcol_mu * r_c
        row += f"  {lam:10.4f}"
    print(row)

for c, r in records.items():
    if len(r["g_FB"]) > 0:
        assert min(r["g_FB"]) > 0, f"client {c}: a g_FB probe was exactly 0 -- loss_fb produced no gradient"

if len(ratios) == 0:
    print("\nno client produced valid probes -- cannot recommend lambda_FB. check checkpoint contents.")
else:
    anchor_client = min(ratios, key=ratios.get)
    r_anchor = ratios[anchor_client]
    print(f"\nanchor client (min g_loss1/g_FB ratio, most FB-sensitive) = client {anchor_client}  (r = {r_anchor:.4f})")
    print("recommended lambda_FB candidates (rho * adcol_mu * r_anchor), conservative envelope for ALL dslr clients:")
    for rho in rho_list:
        lam = rho * args.adcol_mu * r_anchor
        print(f"  rho={rho}: lambda_FB = {lam:.4f}")

    print("\nrealized relative strength R_c = lambda_FB*g_FB,c / (mu*g_loss1,c) at rho=0.25 (should be <= 0.25 for all):")
    lam_025 = 0.25 * args.adcol_mu * r_anchor
    for c in dslr_client_indices:
        if c not in ratios:
            continue
        R_c = lam_025 * statistics.median(records[c]["g_FB"]) / (args.adcol_mu * statistics.median(records[c]["g_loss1"]))
        print(f"  client {c}: R = {R_c:.4f}")

    for c in dslr_client_indices:
        cs = records[c]["cos"]
        if len(cs) == 0:
            continue
        med_cs = statistics.median(cs)
        if med_cs < -0.1:
            print(f"\n[WARNING] client {c}: median cos(g_loss1, g_FB) = {med_cs:.4f} -- notably negative, "
                  f"loss1 and loss_fb may be pulling in conflicting directions on this client. "
                  f"Diagnostic flag only, not used to adjust lambda_FB -- reconsider before starting the "
                  f"100-round run if this looks severe.")

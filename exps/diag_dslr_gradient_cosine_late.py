import sys, os
sys.path.insert(0, os.getcwd())
import copy
import numpy as np
import torch
import torch.nn.functional as F

from option import args_parser
from federated_main import set_seed
from models import adcol_model
from util import prepare_data_office_multi_clients


sys.argv = ["diag_dslr_gradient_cosine_late"]
args = args_parser()
args.exp = 1
args.mode = "ours"
args.lr = 0.01
args.number_workers = 0
args.dataset = "office"
args.num_classes = 10
args.size = 64
args.batch = 32
args.adcol_mu = 0.1
args.adcol_beta = 0.1
args.domain_keyed_proto = True
args.device = "cpu"
args.seed = 0
set_seed(args)

CKPT_PATH = os.path.expanduser("~/Downloads/seed0_checkpoint.pt")
N_PROBE_BATCHES = 3

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args)
args.num_users = len(train_loader_list)

print(f"Loading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
print(f"  checkpoint is from round {ckpt['round']} (0-indexed)")
domain_to_id = ckpt["domain_to_id"]
global_proto = ckpt["global_proto"]
prev_G_state = ckpt["prev_G_state"]
local_models_state = ckpt["local_models_state"]

client_domain_ids = [domain_to_id[d] for d in client_domains]
datasets_name = sorted(domain_to_id.keys(), key=lambda d: domain_to_id[d])
num_classes = args.num_classes


def build_proto_matrix(proto_dict_slice, feat_dim):
    matrix = torch.zeros((num_classes, feat_dim))
    mask = torch.zeros(num_classes, dtype=torch.bool)
    for label, proto in proto_dict_slice.items():
        matrix[label] = proto
        mask[label] = True
    return matrix, mask


def compute_loss1(rep, labels, proto_matrix, present_mask, T=1.0):
    sample_valid = present_mask[labels]
    if not sample_valid.any():
        return None
    rep_n = F.normalize(rep, dim=1)
    proto_n = F.normalize(proto_matrix, dim=1)
    C_y = proto_n[labels]
    numerator = torch.exp((rep_n * C_y).sum(dim=1) / T)
    sim = (rep_n @ proto_matrix.T) / T
    sim = sim.masked_fill(~present_mask.unsqueeze(0), float('-inf'))
    denominator = torch.exp(sim).sum(dim=1)
    return (-torch.log(numerator / denominator))[sample_valid].mean()


def flat_grad(params):
    grads = [p.grad.detach().flatten() for p in params if p.grad is not None]
    return torch.cat(grads) if grads else None


results = {d: {"g_D": [], "g_G": [], "cos": []} for d in datasets_name}

for client_idx in range(args.num_users):
    domain = client_domains[client_idx]
    my_domain_id = client_domain_ids[client_idx]

    model = adcol_model(num_classes=args.num_classes)
    model.load_state_dict(local_models_state[client_idx])
    model.train()

    client_proto_slice = {
        cls: global_proto[(cls, my_domain_id)]
        for cls in range(num_classes) if (cls, my_domain_id) in global_proto
    }
    if len(client_proto_slice) == 0 or prev_G_state is None or len(prev_G_state) == 0:
        print(f"  client {client_idx} ({domain}): skipped (no P_d,k or no G_k available at this checkpoint)")
        continue

    feat_dim_D = next(iter(client_proto_slice.values())).shape[0]
    proto_matrix_D, mask_D = build_proto_matrix(client_proto_slice, feat_dim_D)
    feat_dim_G = next(iter(prev_G_state.values())).shape[0]
    proto_matrix_G, mask_G = build_proto_matrix(prev_G_state, feat_dim_G)

    train_iter = iter(train_loader_list[client_idx])
    n_batches = min(N_PROBE_BATCHES, len(train_loader_list[client_idx]))
    for _ in range(n_batches):
        try:
            images, labels = next(train_iter)
        except StopIteration:
            break
        images, labels = images.float(), labels.long()

        model.zero_grad()
        rep, _ = model(images)
        loss_D = compute_loss1(rep, labels, proto_matrix_D, mask_D, T=args.T)
        if loss_D is None:
            continue
        loss_D.backward(retain_graph=False)
        g_D = flat_grad(model.features.parameters())
        model.zero_grad()

        rep, _ = model(images)
        loss_G = compute_loss1(rep, labels, proto_matrix_G, mask_G, T=args.T)
        if loss_G is None:
            continue
        loss_G.backward(retain_graph=False)
        g_G = flat_grad(model.features.parameters())
        model.zero_grad()

        if g_D is None or g_G is None:
            continue
        cos = F.cosine_similarity(g_D.unsqueeze(0), g_G.unsqueeze(0)).item()
        results[domain]["g_D"].append(g_D.norm().item())
        results[domain]["g_G"].append(g_G.norm().item())
        results[domain]["cos"].append(cos)
    print(f"  client {client_idx} ({domain}): probed {len(results[domain]['cos'])} batches so far")

print(f"\n{'domain':10s}{'median(g_D)':14s}{'median(g_G)':14s}{'median(cos)':14s}{'n_probes':10s}")
for d in datasets_name:
    r = results[d]
    if len(r["cos"]) == 0:
        print(f"{d:10s}  (no probes -- domain not present at this checkpoint or all batches skipped)")
        continue
    print(f"{d:10s}{np.median(r['g_D']):<14.4f}{np.median(r['g_G']):<14.4f}{np.median(r['cos']):<14.4f}{len(r['cos']):<10d}")

print(f"\n{'domain':10s}{'CV(g_D)':12s}{'CV(g_G)':12s}{'std(cos)':12s}")
for d in datasets_name:
    r = results[d]
    if len(r["cos"]) < 2:
        print(f"{d:10s}  (n<2, CV/std not meaningful)")
        continue
    cv_D = np.std(r["g_D"]) / np.mean(r["g_D"])
    cv_G = np.std(r["g_G"]) / np.mean(r["g_G"])
    print(f"{d:10s}{cv_D:<12.4f}{cv_G:<12.4f}{np.std(r['cos']):<12.4f}")

MU = args.adcol_mu
LAMBDA_G = 0.05
print(f"\n{'domain':10s}{'mu*g_D':12s}{'lambda_G*g_G':16s}{'ratio(lG*gG/mu*gD)':20s}")
for d in datasets_name:
    r = results[d]
    if len(r["cos"]) == 0:
        continue
    mu_gD = MU * np.median(r["g_D"])
    lg_gG = LAMBDA_G * np.median(r["g_G"])
    print(f"{d:10s}{mu_gD:<12.5f}{lg_gG:<16.5f}{lg_gG/mu_gD:<20.4f}")

print("\nreference (Phase 1 calibration probe, EARLY training rounds 4/9/14/19):")
print("  caltech  median(cos)=-0.0095")
print("  amazon   median(cos)= 0.4442")
print("  webcam   median(cos)= 0.5975")
print("  dslr     median(cos)= 0.6965  <-- compare THIS run's dslr number against this")

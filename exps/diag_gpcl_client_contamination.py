import sys, os
sys.path.insert(0, os.getcwd())
import torch
import torch.nn.functional as F

from option import args_parser
from federated_main import set_seed
from util import prepare_data_office_multi_clients
from models import adcol_model


sys.argv = ["diag_gpcl_client_contamination"]
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
args.domain_keyed_proto_gpcl = True
args.device = "cpu"
args.seed = 0
set_seed(args)

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args)
args.num_users = len(train_loader_list)
print("client_domains:", client_domains)

WEIGHTS_DIR = os.path.expanduser("~/Downloads/l2_gpcl/weights_seed0")


@torch.no_grad()
def extract_local_proto_and_counts(client_idx, domain):
    model = adcol_model(num_classes=args.num_classes)
    model.load_state_dict(torch.load(
        f"{WEIGHTS_DIR}/best_local_model_client{client_idx}_{domain}.pth",
        map_location="cpu", weights_only=False))
    model.eval()
    feats_by_class = {c: [] for c in range(args.num_classes)}
    for images, labels in train_loader_list[client_idx]:
        images = images.float()
        rep, _ = model(images)
        for i, lbl in enumerate(labels.tolist()):
            feats_by_class[lbl].append(rep[i])
    protos, counts = {}, {}
    for c, feats in feats_by_class.items():
        if len(feats) > 0:
            protos[c] = F.normalize(torch.stack(feats).mean(dim=0), dim=0)
            counts[c] = len(feats)
    return protos, counts


print("\nExtracting each client's own local per-class prototype + sample counts...")
all_protos, all_counts = {}, {}
for idx, domain in enumerate(client_domains):
    protos, counts = extract_local_proto_and_counts(idx, domain)
    all_protos[idx] = protos
    all_counts[idx] = counts
    print(f"  client{idx}({domain}): classes present={sorted(protos.keys())}, "
          f"sample counts={[(c, counts[c]) for c in sorted(counts.keys())]}")

domain_groups = {}
for idx, domain in enumerate(client_domains):
    domain_groups.setdefault(domain, []).append(idx)

print("\n=== GPCL weight vs sample count, per (class, domain) group with >1 client ===")
print(f"{'domain':10s}{'class':6s}{'client':8s}{'n_samples':10s}{'dist_from_mean':16s}{'gpcl_weight':12s}")
flagged = []
for domain, client_idxs in domain_groups.items():
    if len(client_idxs) < 2:
        continue
    for c in range(args.num_classes):
        present = [idx for idx in client_idxs if c in all_protos[idx]]
        if len(present) < 2:
            continue
        stacked = torch.stack([all_protos[idx][c] for idx in present], dim=0)
        mu = stacked.mean(dim=0)
        dists = ((stacked - mu) ** 2).sum(dim=1)
        d_total = dists.sum()
        weights = dists / d_total if d_total.item() > 1e-12 else torch.ones(len(present)) / len(present)
        for i, idx in enumerate(present):
            n = all_counts[idx][c]
            w = weights[i].item()
            print(f"{domain:10s}{c:<6d}client{idx:<3d}{n:<10d}{dists[i].item():<16.4f}{w:<12.4f}")
            expected_share = 1.0 / len(present)
            if n <= 3 and w > expected_share * 1.5:
                flagged.append((domain, c, idx, n, w, expected_share))

print("\n=== flagged: low-sample client (n<=3) getting >1.5x its 'fair share' of GPCL weight ===")
if not flagged:
    print("  none found -- no evidence of low-sample-client contamination in this checkpoint.")
else:
    for domain, c, idx, n, w, expected in flagged:
        print(f"  domain={domain} class={c} client{idx} n={n} weight={w:.4f} (fair share would be {expected:.4f})")

print("\n=== done ===")

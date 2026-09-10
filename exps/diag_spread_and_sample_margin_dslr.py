import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F
import numpy as np
import csv
from collections import Counter

from option import args_parser
from federated_main import set_seed
from models import adcol_model
from util import prepare_data_office_multi_clients


if len(sys.argv) < 4:
    raise SystemExit("Usage: diag_spread_and_sample_margin_dslr.py <condition_label> <checkpoint_path> <prototype_tensor_path>")
_condition = sys.argv[1]
_checkpoint_path_arg = sys.argv[2]
_proto_tensor_path = sys.argv[3]
_M = 2
sys.argv = ["diag_spread_and_sample_margin_dslr"]
args = args_parser()
args.dataset = "office"
args.num_classes = 10
args.size = 64
args.batch = 32
args.number_workers = 0
args.device = args.device if __import__("torch").cuda.is_available() else "cpu"
args.seed = 0
set_seed(args)

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=_M)
unique_domains = list(dict.fromkeys(client_domains))
dslr_test_loader = test_loader_list[unique_domains.index('dslr')]
dslr_client_indices = [i for i, d in enumerate(client_domains) if d == 'dslr']

checkpoint = torch.load(_checkpoint_path_arg, weights_only=False, map_location=args.device)
proto_data = torch.load(_proto_tensor_path, weights_only=False, map_location=args.device)
p_mk = proto_data["p_mk"]

print(f"\n{'='*90}\ncondition={_condition}  (checkpoint round={checkpoint['round']})\n{'='*90}")

all_train_margins = []
all_W_rows = []
pooled_test_dispersion = []

for ci in dslr_client_indices:
    model = adcol_model(args.num_classes).to(args.device)
    model.load_state_dict(checkpoint["local_models_state"][ci])
    model.eval()

    proto_bank = {c: F.normalize(p_mk[ci][c].to(args.device), dim=0) for c in p_mk[ci]}

    feats_by_class = {}
    margins_this_client = []
    with torch.no_grad():
        for images, labels in train_loader_list[ci]:
            images, labels = images.to(args.device).float(), labels.to(args.device).long()
            rep, logits = model(images)
            for f, z, y in zip(rep, logits, labels.tolist()):
                feats_by_class.setdefault(y, []).append(f.detach())
                m = (z[y] - torch.cat([z[:y], z[y+1:]]).max()).item()
                margins_this_client.append(m)
    all_train_margins.extend(margins_this_client)

    for c, fs in feats_by_class.items():
        if c not in proto_bank:
            continue
        n = len(fs)
        cos_dists = [1 - F.cosine_similarity(f.unsqueeze(0), proto_bank[c].unsqueeze(0)).item() for f in fs]
        W = sum(cos_dists) / n
        all_W_rows.append((ci, c, n, W))

    mm = margins_this_client
    print(f"\ndslr client {ci}: n_train_images={len(mm)}  "
          f"train-sample margin: mean={np.mean(mm):.3f} median={np.median(mm):.3f} "
          f"Q10={np.percentile(mm,10):.3f} Q25={np.percentile(mm,25):.3f}")

    with torch.no_grad():
        for images, labels in dslr_test_loader:
            images, labels = images.to(args.device).float(), labels.to(args.device).long()
            rep, _ = model(images)
            for f, y in zip(rep, labels.tolist()):
                if y not in proto_bank:
                    continue
                d = 1 - F.cosine_similarity(f.unsqueeze(0), proto_bank[y].unsqueeze(0)).item()
                pooled_test_dispersion.append(d)

def bin_n(n):
    return "n=1-2" if n <= 2 else ("n=3-5" if n <= 5 else "n>5")

bins = {"n=1-2": [], "n=3-5": [], "n>5": []}
for ci, c, n, W in all_W_rows:
    bins[bin_n(n)].append(W)

print(f"\n--- 6A: within-class spread W_(m,k) = mean[1-cos(f_i,p_mk)], by n-bin ---")
for b in ["n=1-2", "n=3-5", "n>5"]:
    vals = bins[b]
    if vals:
        print(f"  {b}: mean={np.mean(vals):.4f}  (n_classes_in_bin={len(vals)})")
    else:
        print(f"  {b}: -- (empty)")

print(f"\n--- 6B: individual TRAIN-sample margin (pooled across both clients, n={len(all_train_margins)}) ---")
am = np.array(all_train_margins)
print(f"  mean={am.mean():.3f}  median={np.median(am):.3f}  Q10={np.percentile(am,10):.3f}  Q25={np.percentile(am,25):.3f}  "
      f"frac_negative={(am<0).mean()*100:.2f}%")

print(f"\n--- bonus (exploratory): pooled TEST-set dispersion 1-cos(f(x),p_(m,y)), n={len(pooled_test_dispersion)} ---")
td = np.array(pooled_test_dispersion)
print(f"  mean={td.mean():.4f}  median={np.median(td):.4f}")

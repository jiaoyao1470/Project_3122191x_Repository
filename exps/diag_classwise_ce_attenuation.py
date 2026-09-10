import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

from option import args_parser
from federated_main import set_seed
from models import adcol_model
from util import prepare_data_office_multi_clients


if len(sys.argv) < 5:
    raise SystemExit("Usage: diag_classwise_ce_attenuation.py <condition_label> <checkpoint_path> <client_index> <output_prefix>")
_condition = sys.argv[1]
_checkpoint_path_arg = sys.argv[2]
_client_index = int(sys.argv[3])
_output_prefix = sys.argv[4]
_M = 2
sys.argv = ["diag_classwise_ce_attenuation"]
args = args_parser()
args.dataset = "office"
args.num_classes = 10
args.size = 64
args.batch = 32
args.number_workers = 0
args.device = args.device if __import__("torch").cuda.is_available() else "cpu"
args.seed = 0
set_seed(args)

LABEL_NAMES = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard',
               'laptop_computer', 'monitor', 'mouse', 'mug', 'projector']

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=_M)

checkpoint = torch.load(_checkpoint_path_arg, weights_only=False, map_location=args.device)
print(f"\n{'='*90}\ncondition={_condition}  round={checkpoint['round']}  client={_client_index}\n{'='*90}")

model = adcol_model(args.num_classes).to(args.device)
model.load_state_dict(checkpoint["local_models_state"][_client_index])
model.eval()

criterion_CE = nn.CrossEntropyLoss()
backbone_params = list(model.features.parameters())
head_params = list(model.classifier.parameters())

imgs_by_class = {}
for images, labels in train_loader_list[_client_index]:
    for img, lab in zip(images, labels.tolist()):
        imgs_by_class.setdefault(lab, []).append(img)


def grad_norm_for(loss, params):
    for p in params:
        if p.grad is not None:
            p.grad = None
    loss.backward(retain_graph=False)
    grads = [p.grad.flatten() for p in params if p.grad is not None]
    if not grads:
        return float('nan')
    return torch.cat(grads).norm().item()


rows = []
for c in sorted(imgs_by_class.keys()):
    images = torch.stack(imgs_by_class[c]).to(args.device).float()
    labels = torch.full((len(imgs_by_class[c]),), c, device=args.device, dtype=torch.long)

    rep, logits = model(images)
    per_sample_ce = F.cross_entropy(logits, labels, reduction='none')
    ce_loss = per_sample_ce.mean()
    pred = logits.argmax(dim=1)
    train_acc_c = pred.eq(labels).float().mean().item()
    margins = []
    for z in logits:
        m = (z[c] - torch.cat([z[:c], z[c+1:]]).max()).item()
        margins.append(m)

    gB = grad_norm_for(ce_loss, backbone_params)
    rep2, logits2 = model(images)
    ce_loss2 = F.cross_entropy(logits2, labels)
    gH = grad_norm_for(ce_loss2, head_params)

    row = {
        "class_id": c, "class_name": LABEL_NAMES[c], "n_train": len(imgs_by_class[c]),
        "CE_loss": ce_loss.item(), "train_acc": train_acc_c,
        "train_margin_mean": float(np.mean(margins)), "gB_CE": gB, "gH_CE": gH,
    }
    rows.append(row)
    print(f"  class {c:2d} {LABEL_NAMES[c]:16s} n={len(imgs_by_class[c]):3d}  CE_loss={ce_loss.item():7.4f}  "
          f"train_acc={train_acc_c*100:6.2f}%  margin={np.mean(margins):7.3f}  gB_CE={gB:8.4f}  gH_CE={gH:7.4f}")

import csv
csv_path = f"{_output_prefix}_{_condition}_client{_client_index}_round{checkpoint['round']}_classwise_ce.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nWrote {len(rows)} rows to {csv_path}")

overall = [r["gB_CE"] for r in rows]
print(f"\noverall gB_CE across classes: median={np.median(overall):.4f}  "
      f"min={min(overall):.4f}(class {rows[np.argmin(overall)]['class_id']})  "
      f"max={max(overall):.4f}(class {rows[np.argmax(overall)]['class_id']})")

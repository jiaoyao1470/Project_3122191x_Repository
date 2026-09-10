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
from util import prepare_data_office_multi_clients, get_mean_domain_keyed


if len(sys.argv) < 5:
    raise SystemExit(
        "Usage: diag_prototype_pipeline_ablation.py <condition_label> <checkpoint_path> "
        "<M1_checkpoint_path_or_none> <output_prefix>"
    )
_condition = sys.argv[1]
_checkpoint_path_arg = sys.argv[2]
_m1_checkpoint_path_arg = sys.argv[3]
_output_prefix = sys.argv[4]
_M = 2
sys.argv = ["diag_prototype_pipeline_ablation"]
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

DSLR_DOMAIN_ID = 0


def load_dslr_setup(M):
    train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=M)
    dslr_client_indices = [i for i, d in enumerate(client_domains) if d == 'dslr']
    return train_loader_list, dslr_client_indices


def extract_features_by_class(model, loader):
    model.eval()
    feats_by_class = {}
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(args.device).float(), labels.to(args.device).long()
            rep, _ = model(images)
            for f, c in zip(rep, labels.tolist()):
                feats_by_class.setdefault(c, []).append(f.detach().clone())
    return feats_by_class


def compute_proto(feats_by_class):
    return {c: torch.mean(torch.stack(fs), dim=0) for c, fs in feats_by_class.items()}


def load_client_models(checkpoint_path, dslr_client_indices):
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=args.device)
    models = {}
    for ci in dslr_client_indices:
        m = adcol_model(args.num_classes).to(args.device)
        m.load_state_dict(checkpoint["local_models_state"][ci])
        models[ci] = m
    return models, checkpoint


print(f"=== Loading M={_M}, condition={_condition} ===")
train_loader_list, dslr_client_indices = load_dslr_setup(_M)
models, checkpoint = load_client_models(_checkpoint_path_arg, dslr_client_indices)
print(f"Loaded checkpoint from round {checkpoint['round']}")

client_feats_by_class = {}
train_counts = {}
for ci in dslr_client_indices:
    idx = train_loader_list[ci].sampler.indices
    labels_arr = np.array(train_loader_list[ci].dataset.labels)[idx]
    train_counts[ci] = Counter(labels_arr.tolist())
    client_feats_by_class[ci] = extract_features_by_class(models[ci], train_loader_list[ci])

p_mk = {ci: compute_proto(client_feats_by_class[ci]) for ci in dslr_client_indices}

print("Computing leave-one-client-out fidelity references (re-encoding other clients' images)...")
other_client_images_by_class = {}
for ci in dslr_client_indices:
    for images, labels in train_loader_list[ci]:
        for img, lab in zip(images, labels.tolist()):
            other_client_images_by_class.setdefault(lab, []).append((ci, img))

R_mk_leave_one_out = {}
for ci in dslr_client_indices:
    R_mk_leave_one_out[ci] = {}
    model_m = models[ci]
    model_m.eval()
    for c in range(args.num_classes):
        others_imgs = [img for (owner, img) in other_client_images_by_class.get(c, []) if owner != ci]
        if len(others_imgs) == 0:
            continue
        batch = torch.stack(others_imgs).to(args.device).float()
        with torch.no_grad():
            rep, _ = model_m(batch)
        R_mk_leave_one_out[ci][c] = rep.mean(dim=0)

proto_list = [p_mk[ci] for ci in dslr_client_indices]
num_list = [train_counts[ci] for ci in dslr_client_indices]
client_domain_ids = [DSLR_DOMAIN_ID for _ in dslr_client_indices]
P_dk_keyed = get_mean_domain_keyed(args, proto_list, num_list, client_domain_ids)
P_dk = {cls: vec for (cls, dom), vec in P_dk_keyed.items() if dom == DSLR_DOMAIN_ID}

G_k_real = None
if checkpoint.get("prev_G_state") is not None:
    G_k_real = checkpoint["prev_G_state"]
    print(f"Found real trained G_k in checkpoint (classes present: {sorted(G_k_real.keys())})")

R_dk = None
if _m1_checkpoint_path_arg.lower() != "none":
    print(f"=== Loading M=1 reference (condition={_condition}) for R_{{d,k}} ===")
    _, dslr_client_indices_m1 = load_dslr_setup(1)
    models_m1, _ = load_client_models(_m1_checkpoint_path_arg, dslr_client_indices_m1)
    assert len(dslr_client_indices_m1) == 1, "M=1 should have exactly one dslr client"
    m1_ci = dslr_client_indices_m1[0]
    train_loader_list_m1, _ = load_dslr_setup(1)
    R_dk = compute_proto(extract_features_by_class(models_m1[m1_ci], train_loader_list_m1[m1_ci]))
else:
    print("No M=1 reference given -- skipping Diagnostic 4b (domain_fidelity_* will be NaN)")

rows = []
for ci in dslr_client_indices:
    for c in range(args.num_classes):
        n_train_c = train_counts[ci].get(c, 0)
        row = {
            "condition": _condition, "M": _M, "client_id": ci, "class_id": c, "class_name": LABEL_NAMES[c],
            "n_train": n_train_c, "coverage_zero": int(n_train_c == 0),
        }
        if c in p_mk[ci]:
            p = p_mk[ci][c]
            row["local_proto_norm"] = p.norm().item()
            if c in R_mk_leave_one_out.get(ci, {}):
                r = R_mk_leave_one_out[ci][c]
                row["fidelity_cos"] = F.cosine_similarity(p.unsqueeze(0), r.unsqueeze(0)).item()
                row["fidelity_l2"] = (F.normalize(p, dim=0) - F.normalize(r, dim=0)).norm().item()
            else:
                row["fidelity_cos"] = float('nan')
                row["fidelity_l2"] = float('nan')
            if c in P_dk:
                P = P_dk[c]
                row["to_domain_cos"] = F.cosine_similarity(p.unsqueeze(0), P.unsqueeze(0)).item()
                row["to_domain_l2"] = (F.normalize(p, dim=0) - F.normalize(P, dim=0)).norm().item()
            else:
                row["to_domain_cos"] = float('nan')
                row["to_domain_l2"] = float('nan')
        else:
            row["local_proto_norm"] = float('nan')
            row["fidelity_cos"] = float('nan')
            row["fidelity_l2"] = float('nan')
            row["to_domain_cos"] = float('nan')
            row["to_domain_l2"] = float('nan')
        if R_dk is not None and c in P_dk and c in R_dk:
            row["domain_proto_norm"] = P_dk[c].norm().item()
            row["domain_fidelity_cos"] = F.cosine_similarity(P_dk[c].unsqueeze(0), R_dk[c].unsqueeze(0)).item()
            row["domain_fidelity_l2"] = (F.normalize(P_dk[c], dim=0) - F.normalize(R_dk[c], dim=0)).norm().item()
        else:
            row["domain_proto_norm"] = float('nan')
            row["domain_fidelity_cos"] = float('nan')
            row["domain_fidelity_l2"] = float('nan')
        rows.append(row)

csv_path = f"{_output_prefix}_{_condition}_prototype_diag.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {len(rows)} rows to {csv_path}")

tensor_path = f"{_output_prefix}_{_condition}_prototype_tensors.pt"
torch.save({
    "p_mk": {ci: {c: v.cpu() for c, v in d.items()} for ci, d in p_mk.items()},
    "R_mk_leave_one_out": {ci: {c: v.cpu() for c, v in d.items()} for ci, d in R_mk_leave_one_out.items()},
    "P_dk": {c: v.cpu() for c, v in P_dk.items()},
    "R_dk_M1_reference": {c: v.cpu() for c, v in R_dk.items()} if R_dk is not None else None,
    "G_k_real_trained": {c: v.cpu() for c, v in G_k_real.items()} if G_k_real is not None else None,
    "train_counts": {ci: dict(cnt) for ci, cnt in train_counts.items()},
    "round": checkpoint["round"], "M": _M, "condition": _condition,
}, tensor_path)
print(f"Wrote tensors to {tensor_path}")

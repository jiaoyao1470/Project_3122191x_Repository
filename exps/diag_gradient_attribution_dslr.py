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
from models import adcol_model, Discriminator
from util import prepare_data_office_multi_clients


if len(sys.argv) < 5 or sys.argv[1] not in ("1", "2", "4") or sys.argv[2] not in ("baseline", "D"):
    raise SystemExit("Usage: diag_gradient_attribution_dslr.py <1|2|4> <baseline|D> <checkpoint_path> <output_prefix>")
_M = int(sys.argv[1])
_condition = sys.argv[2]
_checkpoint_path_arg = sys.argv[3]
_output_prefix = sys.argv[4]
sys.argv = ["diag_gradient_attribution_dslr"]
args = args_parser()

args.dataset = "office"
args.num_classes = 10
args.size = 64
args.batch = 32
args.number_workers = 0
args.device = args.device if __import__("torch").cuda.is_available() else "cpu"
args.seed = 0
args.domain_keyed_proto = (_condition == "D")
set_seed(args)

LABEL_NAMES = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard',
               'laptop_computer', 'monitor', 'mouse', 'mug', 'projector']

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=_M)
dslr_client_indices = [i for i, d in enumerate(client_domains) if d == 'dslr']
args.num_users = len(train_loader_list)

checkpoint = torch.load(_checkpoint_path_arg, weights_only=False, map_location=args.device)
print(f"Loaded checkpoint from round {checkpoint['round']} (M={_M}, condition={_condition})")

domain_to_id = checkpoint.get("domain_to_id", None)
global_proto = checkpoint.get("global_proto", {})
global_G = checkpoint.get("prev_G_state", None) if _condition == "D" else None
num_domains = args.num_users if _condition == "baseline" else (len(domain_to_id) if domain_to_id else 4)

criterion_CE = nn.CrossEntropyLoss()
kl_loss_func = nn.KLDivLoss(reduction="batchmean")


def bin_for(n):
    if n <= 2: return "n=1-2"
    elif n <= 5: return "n=3-5"
    else: return "n>5"


def compute_losses(model, discriminator, images, labels, client_global_proto):
    rep, logits = model(images)
    losses = {}
    losses["CE"] = criterion_CE(logits, labels)

    if len(client_global_proto) == 0:
        losses["proto"] = None
    elif args.domain_keyed_proto:
        proto_dim = next(iter(client_global_proto.values())).shape[0]
        proto_matrix = torch.zeros((args.num_classes, proto_dim), device=args.device)
        present_mask = torch.zeros(args.num_classes, dtype=torch.bool, device=args.device)
        for label_c, proto in client_global_proto.items():
            proto_matrix[label_c] = proto
            present_mask[label_c] = True
        sample_valid = present_mask[labels]
        if sample_valid.any():
            rep_n = F.normalize(rep, dim=1)
            proto_matrix_n = F.normalize(proto_matrix, dim=1)
            C_y = proto_matrix_n[labels]
            numerator = torch.exp((rep_n * C_y).sum(dim=1) / args.T)
            sim = (rep_n @ proto_matrix.T) / args.T
            sim = sim.masked_fill(~present_mask.unsqueeze(0), float('-inf'))
            denominator = torch.exp(sim).sum(dim=1)
            losses["proto"] = (-torch.log(numerator / denominator))[sample_valid].mean()
        else:
            losses["proto"] = None
    else:
        num_classes_present = len(client_global_proto)
        proto_dim = next(iter(client_global_proto.values())).shape[0]
        proto_matrix = torch.zeros((num_classes_present, proto_dim), device=args.device)
        label_to_row = {}
        for row, (label_c, proto) in enumerate(client_global_proto.items()):
            proto_matrix[row] = proto
            label_to_row[label_c] = row
        valid = torch.tensor([l.item() in label_to_row for l in labels], device=args.device)
        if valid.any():
            rep_n = F.normalize(rep, dim=1)
            proto_matrix_n = F.normalize(proto_matrix, dim=1)
            rows = torch.tensor([label_to_row[l.item()] for l in labels if l.item() in label_to_row], device=args.device)
            C_y = proto_matrix_n[rows]
            numerator = torch.exp((rep_n[valid] * C_y).sum(dim=1) / args.T)
            denominator = torch.exp((rep_n[valid] @ proto_matrix_n.T) / args.T).sum(dim=1)
            losses["proto"] = (-torch.log(numerator / denominator)).mean()
        else:
            losses["proto"] = None

    if global_G is not None and len(global_G) > 0:
        proto_dim_G = next(iter(global_G.values())).shape[0]
        G_matrix = torch.zeros((args.num_classes, proto_dim_G), device=args.device)
        G_present_mask = torch.zeros(args.num_classes, dtype=torch.bool, device=args.device)
        for label_c, proto in global_G.items():
            G_matrix[label_c] = proto.to(args.device)
            G_present_mask[label_c] = True
        G_sample_valid = G_present_mask[labels]
        if G_sample_valid.any():
            rep_n = F.normalize(rep, dim=1)
            G_matrix_n = F.normalize(G_matrix, dim=1)
            C_y_G = G_matrix_n[labels]
            numerator_G = torch.exp((rep_n * C_y_G).sum(dim=1) / args.T)
            sim_G = (rep_n @ G_matrix.T) / args.T
            sim_G = sim_G.masked_fill(~G_present_mask.unsqueeze(0), float('-inf'))
            denominator_G = torch.exp(sim_G).sum(dim=1)
            losses["cross"] = (-torch.log(numerator_G / denominator_G))[G_sample_valid].mean()
        else:
            losses["cross"] = None
    else:
        losses["cross"] = None

    client_index = discriminator(rep)
    client_index_softmax = F.log_softmax(client_index, dim=-1)
    target_index = torch.full(client_index.shape, 1.0 / num_domains, device=args.device)
    target_index_softmax = F.softmax(target_index, dim=-1)
    losses["KL"] = kl_loss_func(client_index_softmax, target_index_softmax)
    return losses


def grad_vector_for(loss, backbone_params):
    for p in backbone_params:
        if p.grad is not None:
            p.grad = None
    loss.backward(retain_graph=True)
    flat = torch.cat([p.grad.flatten() for p in backbone_params if p.grad is not None])
    return flat.norm().item(), flat.detach().clone()


def cosine(v1, v2):
    if v1 is None or v2 is None:
        return float('nan')
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


results = []
for ci in dslr_client_indices:
    model = adcol_model(args.num_classes).to(args.device)
    model.load_state_dict(checkpoint["local_models_state"][ci])
    model.eval()

    discriminator = Discriminator(model, num_domains).to(args.device)
    discriminator.load_state_dict(checkpoint["discriminator_state"])
    discriminator.eval()
    for p in discriminator.parameters():
        p.requires_grad_(False)

    if args.domain_keyed_proto:
        my_domain_id = domain_to_id["dslr"] if domain_to_id else None
        client_global_proto = {cls: global_proto[(cls, my_domain_id)] for cls in range(args.num_classes)
                                if (cls, my_domain_id) in global_proto}
    else:
        client_global_proto = global_proto

    backbone_params = list(model.features.parameters())

    idx = train_loader_list[ci].sampler.indices
    labels_arr = np.array(train_loader_list[ci].dataset.labels)[idx]
    train_counts = Counter(labels_arr.tolist())

    imgs_by_bin = {"n=1-2": [], "n=3-5": [], "n>5": []}
    labs_by_bin = {"n=1-2": [], "n=3-5": [], "n>5": []}
    for images, labels in train_loader_list[ci]:
        for img, lab in zip(images, labels.tolist()):
            b = bin_for(train_counts[lab])
            imgs_by_bin[b].append(img)
            labs_by_bin[b].append(lab)

    for b in ["n=1-2", "n=3-5", "n>5"]:
        if len(imgs_by_bin[b]) == 0:
            continue
        images = torch.stack(imgs_by_bin[b]).to(args.device).float()
        labels = torch.tensor(labs_by_bin[b]).to(args.device).long()

        losses = compute_losses(model, discriminator, images, labels, client_global_proto)
        grad_norms = {}
        grad_vecs = {}
        for name in ["CE", "KL", "proto", "cross"]:
            l = losses[name]
            if l is None:
                grad_norms[name] = float('nan')
                grad_vecs[name] = None
                continue
            grad_norms[name], grad_vecs[name] = grad_vector_for(l, backbone_params)
        for p in backbone_params:
            p.grad = None

        lambda_G_val = 0.05 if _condition == "D" else 0.0
        row = {
            "client_id": ci, "n_bin": b, "n_images_in_bin": len(imgs_by_bin[b]),
            "grad_CE": grad_norms["CE"], "grad_KL": grad_norms["KL"],
            "grad_proto": grad_norms["proto"], "grad_cross": grad_norms["cross"],
            "weighted_KL": grad_norms["KL"] * args.adcol_beta if not np.isnan(grad_norms["KL"]) else float('nan'),
            "weighted_proto": grad_norms["proto"] * args.adcol_mu if not np.isnan(grad_norms["proto"]) else float('nan'),
            "weighted_cross": grad_norms["cross"] * lambda_G_val if not np.isnan(grad_norms["cross"]) else float('nan'),
            "cos_CE_KL": cosine(grad_vecs["CE"], grad_vecs["KL"]),
            "cos_CE_proto": cosine(grad_vecs["CE"], grad_vecs["proto"]),
            "cos_CE_cross": cosine(grad_vecs["CE"], grad_vecs["cross"]),
        }
        results.append(row)
        print(f"client{ci} {b} (n={len(imgs_by_bin[b])}): CE={grad_norms['CE']:.4f} KL={grad_norms['KL']:.4f} "
              f"proto={grad_norms['proto']:.4f} cross={grad_norms['cross']}  "
              f"cos(CE,proto)={row['cos_CE_proto']:.3f} cos(CE,cross)={row['cos_CE_cross']:.3f}")

import csv
csv_path = f"{_output_prefix}_M{_M}_{_condition}_gradient_attribution.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)
print(f"\nWrote {len(results)} rows to {csv_path}")

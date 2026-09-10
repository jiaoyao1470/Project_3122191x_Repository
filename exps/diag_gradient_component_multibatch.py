import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv

from option import args_parser
from federated_main import set_seed
from models import adcol_model, Discriminator
from util import prepare_data_office_multi_clients


if len(sys.argv) < 8:
    raise SystemExit(
        "Usage: diag_gradient_component_multibatch.py <condition_label> <checkpoint_path> "
        "<discriminator_fix_on:0|1> <domain_keyed_proto_on:0|1> <lambda_G_value> <n_batches> <output_prefix>"
    )
_condition = sys.argv[1]
_checkpoint_path_arg = sys.argv[2]
_disc_fix_on = bool(int(sys.argv[3]))
_domain_keyed_on = bool(int(sys.argv[4]))
_lambda_G = float(sys.argv[5])
_n_batches = int(sys.argv[6])
_output_prefix = sys.argv[7]
_M = 2
_batch_size = 20
sys.argv = ["diag_gradient_component_multibatch"]
args = args_parser()

args.dataset = "office"
args.num_classes = 10
args.size = 64
args.batch = 32
args.number_workers = 0
args.device = args.device if __import__("torch").cuda.is_available() else "cpu"
args.seed = 0
args.domain_keyed_proto = _domain_keyed_on
set_seed(args)

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=_M)
dslr_client_indices = [i for i, d in enumerate(client_domains) if d == 'dslr']
args.num_users = len(train_loader_list)

checkpoint = torch.load(_checkpoint_path_arg, weights_only=False, map_location=args.device)
print(f"Loaded checkpoint from round {checkpoint['round']} (condition={_condition}, "
      f"disc_fix_on={_disc_fix_on}, domain_keyed_on={_domain_keyed_on}, lambda_G={_lambda_G})")

domain_to_id = checkpoint.get("domain_to_id", None)
global_proto = checkpoint.get("global_proto", {})
global_G = checkpoint.get("prev_G_state", None) if _domain_keyed_on else None
num_domains = (len(domain_to_id) if (domain_to_id and _disc_fix_on) else args.num_users)

criterion_CE = nn.CrossEntropyLoss()
kl_loss_func = nn.KLDivLoss(reduction="batchmean")


def compute_losses(model, discriminator, images, labels, client_global_proto):
    rep, logits = model(images)
    losses = {}
    losses["CE"] = criterion_CE(logits, labels)

    if len(client_global_proto) == 0:
        losses["proto"] = None
    elif _domain_keyed_on:
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


def grad_vector_for(loss, params):
    for p in params:
        if p.grad is not None:
            p.grad = None
    loss.backward(retain_graph=True)
    grads = [p.grad.flatten() for p in params if p.grad is not None]
    if not grads:
        return float('nan'), None
    flat = torch.cat(grads)
    return flat.norm().item(), flat.detach().clone()


def cosine(v1, v2):
    if v1 is None or v2 is None:
        return float('nan')
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


rows = []
for ci in dslr_client_indices:
    model = adcol_model(args.num_classes).to(args.device)
    model.load_state_dict(checkpoint["local_models_state"][ci])
    model.eval()

    discriminator = Discriminator(model, num_domains).to(args.device)
    discriminator.load_state_dict(checkpoint["discriminator_state"])
    discriminator.eval()
    for p in discriminator.parameters():
        p.requires_grad_(False)

    if _domain_keyed_on:
        my_domain_id = domain_to_id["dslr"] if domain_to_id else None
        client_global_proto = {cls: global_proto[(cls, my_domain_id)] for cls in range(args.num_classes)
                                if (cls, my_domain_id) in global_proto}
    else:
        client_global_proto = global_proto

    backbone_params = list(model.features.parameters())
    head_params = list(model.classifier.parameters())

    all_imgs, all_labs = [], []
    for images, labels in train_loader_list[ci]:
        all_imgs.extend(images)
        all_labs.extend(labels.tolist())
    all_imgs = torch.stack(all_imgs)
    all_labs = torch.tensor(all_labs)
    n_pool = len(all_labs)

    g = torch.Generator().manual_seed(0)
    for b_idx in range(_n_batches):
        perm = torch.randperm(n_pool, generator=g)[:_batch_size]
        images = all_imgs[perm].to(args.device).float()
        labels = all_labs[perm].to(args.device).long()

        losses = compute_losses(model, discriminator, images, labels, client_global_proto)

        grad_norms_B, grad_vecs_B = {}, {}
        grad_norms_H, grad_vecs_H = {}, {}
        for name in ["CE", "KL", "proto", "cross"]:
            l = losses[name]
            if l is None:
                grad_norms_B[name], grad_vecs_B[name] = float('nan'), None
                grad_norms_H[name], grad_vecs_H[name] = float('nan'), None
                continue
            grad_norms_B[name], grad_vecs_B[name] = grad_vector_for(l, backbone_params)
            grad_norms_H[name], grad_vecs_H[name] = grad_vector_for(l, head_params)
        for p in backbone_params + head_params:
            p.grad = None

        def proj_ratio(g_other, g_ce):
            if g_other is None or g_ce is None:
                return float('nan')
            return (g_other @ g_ce).item() / (g_ce.norm().item() ** 2)

        row = {
            "client_id": ci, "batch_idx": b_idx,
            "gB_CE": grad_norms_B["CE"], "gB_KL": grad_norms_B["KL"],
            "gB_proto": grad_norms_B["proto"], "gB_cross": grad_norms_B["cross"],
            "gH_CE": grad_norms_H["CE"], "gH_KL": grad_norms_H["KL"],
            "gH_proto": grad_norms_H["proto"], "gH_cross": grad_norms_H["cross"],
            "cosB_CE_KL": cosine(grad_vecs_B["CE"], grad_vecs_B["KL"]),
            "cosB_CE_proto": cosine(grad_vecs_B["CE"], grad_vecs_B["proto"]),
            "cosB_CE_cross": cosine(grad_vecs_B["CE"], grad_vecs_B["cross"]),
            "cosB_proto_cross": cosine(grad_vecs_B["proto"], grad_vecs_B["cross"]),
            "R_proto_B": (args.adcol_mu * grad_norms_B["proto"] / grad_norms_B["CE"]) if grad_norms_B["CE"] else float('nan'),
            "R_cross_B": (_lambda_G * grad_norms_B["cross"] / grad_norms_B["CE"]) if (grad_norms_B["CE"] and not np.isnan(grad_norms_B["cross"])) else float('nan'),
            "proj_cross_onto_CE_B": (_lambda_G * proj_ratio(grad_vecs_B["cross"], grad_vecs_B["CE"])) if grad_vecs_B["cross"] is not None else float('nan'),
        }
        rows.append(row)

import statistics
print(f"\n{'='*90}\ncondition={_condition}  round={checkpoint['round']}  n_batches={_n_batches}\n{'='*90}")
for ci in dslr_client_indices:
    client_rows = [r for r in rows if r["client_id"] == ci]
    print(f"\n--- client {ci} (median over {len(client_rows)} batches, IQR in brackets) ---")
    for key in ["gB_CE", "gB_KL", "gB_proto", "gB_cross", "gH_CE", "gH_KL", "gH_proto", "gH_cross",
                "cosB_CE_KL", "cosB_CE_proto", "cosB_CE_cross", "cosB_proto_cross",
                "R_proto_B", "R_cross_B", "proj_cross_onto_CE_B"]:
        vals = [r[key] for r in client_rows if not (isinstance(r[key], float) and np.isnan(r[key]))]
        if vals:
            med = statistics.median(vals)
            q25 = np.percentile(vals, 25)
            q75 = np.percentile(vals, 75)
            print(f"  {key:24s}: median={med:8.4f}  [{q25:8.4f}, {q75:8.4f}]")
        else:
            print(f"  {key:24s}: -- (all NaN)")

csv_path = f"{_output_prefix}_{_condition}_round{checkpoint['round']}_gradient_multibatch.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nWrote {len(rows)} rows to {csv_path}")

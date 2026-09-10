import sys
import os
sys.path.insert(0, os.getcwd())

import shutil
import torch

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


def make_args(iters):
    sys.argv = ["smoke_test_peer_update_mixing"]
    args = args_parser()
    args.exp = 1
    args.mode = "ours"
    args.iters = iters
    args.lr = 0.01
    args.number_workers = 0
    args.device = "cpu"
    args.dataset = "office"
    args.seed = 0
    args.num_classes = 10
    args.size = 64
    args.batch = 32
    args.wk_iters = 1
    args.adcol_epoch = 1
    args.adcol_mu = 0.1
    args.adcol_beta = 0.1
    args.domain_keyed_proto = True
    args.no_discriminator_fix = False
    set_seed(args)
    return args


def build(args):
    tl, cd, testl = prepare_data_office_multi_clients(args, dslr_client_override=4)
    args.num_users = len(tl)
    return tl, cd, testl


def load_and_discard(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if os.path.exists(path):
        os.remove(path)
    return ckpt


TMP_DIR = os.path.join(os.getcwd(), "_smoke_peer_update_mixing_tmp")
os.makedirs(TMP_DIR, exist_ok=True)

print("=" * 78)
print("GROUP 1: accuracy-trajectory checks")
print("=" * 78)

print("(a) peer_update_mixing_domains=None, iters=2")
args = make_args(2)
tl, cd, testl = build(args)
_, acc_a, names = ours(args, tl, testl, cd)

print("(b) domains={'dslr'}, alpha=0.0, iters=2")
args = make_args(2)
tl, cd, testl = build(args)
loss_b, acc_b, _ = ours(args, tl, testl, cd, peer_update_mixing_domains={"dslr"}, peer_update_mixing_alpha=0.0)
worst_ab = max(abs(acc_a[d][r] - acc_b[d][r]) for d in names for r in range(len(acc_a[d])))
print(f"  max |acc(a)-acc(b)| over 2 rounds = {worst_ab:.10f}")
print("  " + ("PASS -- alpha=0.0 is a true no-op" if worst_ab == 0.0 else "FAIL"))

print("(c) domains={'dslr'}, alpha=0.1, iters=2")
args = make_args(2)
tl, cd, testl = build(args)
loss_c, acc_c, _ = ours(args, tl, testl, cd, peer_update_mixing_domains={"dslr"}, peer_update_mixing_alpha=0.1)
r0_diff = max(abs(acc_b[d][0] - acc_c[d][0]) for d in names)
print(f"  round0 max |acc(b)-acc(c)| = {r0_diff:.10f}  (must be 0.0 -- mixing hasn't fired yet)")
print("  " + ("PASS" if r0_diff == 0.0 else "FAIL"))
r1_loss_diff = max(abs(loss_b[d][1] - loss_c[d][1]) for d in names)
r1_acc_diff = max(abs(acc_b[d][1] - acc_c[d][1]) for d in names)
print(f"  round1 max |train_loss(b)-train_loss(c)| = {r1_loss_diff:.10f}  "
      f"(must be > 0.0 -- mixing changed round1's start)")
print("  " + ("PASS -- mixing actually reaches training" if r1_loss_diff > 0.0 else "FAIL -- feature is inert"))
print(f"  round1 max |acc(b)-acc(c)| = {r1_acc_diff:.10f}  (informational only -- may legitimately be "
      f"0.0 this early even when mixing is working; see comment above, not asserted as PASS/FAIL)")

print()
print("=" * 78)
print("GROUP 2: parameter-level checks (checkpoints)")
print("=" * 78)

DSLR_CLIENTS = [6, 7, 8, 9]

print("(b1) alpha=0.0 reference, iters=1, checkpointed")
args = make_args(1)
tl, cd, testl = build(args)
ckpt_path_b1 = os.path.join(TMP_DIR, "b1.pt")
ours(args, tl, testl, cd, checkpoint_path=ckpt_path_b1, checkpoint_every=1,
     peer_update_mixing_domains={"dslr"}, peer_update_mixing_alpha=0.0)
ckpt_b1 = load_and_discard(ckpt_path_b1)

print("(d) alpha=0.1, iters=1, checkpointed")
args = make_args(1)
tl, cd, testl = build(args)
ckpt_path_d = os.path.join(TMP_DIR, "d.pt")
ours(args, tl, testl, cd, checkpoint_path=ckpt_path_d, checkpoint_every=1,
     peer_update_mixing_domains={"dslr"}, peer_update_mixing_alpha=0.1)
ckpt_d = load_and_discard(ckpt_path_d)

print("(e) alpha=0.3, iters=1, checkpointed")
args = make_args(1)
tl, cd, testl = build(args)
ckpt_path_e = os.path.join(TMP_DIR, "e.pt")
ours(args, tl, testl, cd, checkpoint_path=ckpt_path_e, checkpoint_every=1,
     peer_update_mixing_domains={"dslr"}, peer_update_mixing_alpha=0.3)
ckpt_e = load_and_discard(ckpt_path_e)

shutil.rmtree(TMP_DIR, ignore_errors=True)

counts = {}
for i in DSLR_CLIENTS:
    idxs = getattr(tl[i].sampler, "indices", None)
    counts[i] = len(idxs) if idxs is not None else len(tl[i].dataset)
total = float(sum(counts.values()))
print(f"dslr client sample counts: {counts}  (total={total:.0f})")

import torch.nn as nn


def eligible_names(model):
    bn_names = set()
    for name, m in model.features.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.weight is not None:
            bn_names.add(f"{name}.weight")
            bn_names.add(f"{name}.bias")
    return [n for n, _ in model.features.named_parameters() if n not in bn_names], bn_names


from models import adcol_model
probe_model = adcol_model(num_classes=args.num_classes)
elig_names, bn_names = eligible_names(probe_model)
print(f"eligible (non-BN) feature params: {len(elig_names)}   BN weight/bias params: {len(bn_names)}")

max_scale_err = 0.0
for i in DSLR_CLIENTS:
    sd_b1 = ckpt_b1["local_models_state"][i]
    sd_d = ckpt_d["local_models_state"][i]
    sd_e = ckpt_e["local_models_state"][i]
    for name in elig_names:
        key = f"features.{name}"
        corr_d = (sd_d[key] - sd_b1[key]).double()
        corr_e = (sd_e[key] - sd_b1[key]).double()
        err = (corr_e - 3.0 * corr_d).abs().max().item()
        max_scale_err = max(max_scale_err, err)
print(f"max |correction(alpha=0.3) - 3.0*correction(alpha=0.1)| = {max_scale_err:.8f}")
print("  " + ("PASS -- mixing correction scales linearly in alpha" if max_scale_err < 1e-4 else "FAIL"))

for label, ckpt in [("d (alpha=0.1)", ckpt_d), ("e (alpha=0.3)", ckpt_e)]:
    max_mean_err = 0.0
    for name in elig_names:
        key = f"features.{name}"
        mean_b1 = sum(counts[i] * ckpt_b1["local_models_state"][i][key].double() for i in DSLR_CLIENTS) / total
        mean_x = sum(counts[i] * ckpt["local_models_state"][i][key].double() for i in DSLR_CLIENTS) / total
        max_mean_err = max(max_mean_err, (mean_x - mean_b1).abs().max().item())
    print(f"[{label}] max |weighted-mean(mixed) - weighted-mean(unmixed)| = {max_mean_err:.8f}")
    print("  " + ("PASS -- mean-preserving" if max_mean_err < 1e-4 else "FAIL"))

max_bn_diff = 0.0
for i in DSLR_CLIENTS:
    for name in bn_names:
        key = f"features.{name}"
        diff = (ckpt_d["local_models_state"][i][key] - ckpt_b1["local_models_state"][i][key]).abs().max().item()
        max_bn_diff = max(max_bn_diff, diff)
print(f"max BN weight/bias diff (dslr clients, b1 vs d) = {max_bn_diff:.10f}")
print("  " + ("PASS -- BN affine excluded from mixing" if max_bn_diff == 0.0 else "FAIL"))

sd0_b1 = ckpt_b1["local_models_state"][0]
sd0_d = ckpt_d["local_models_state"][0]
identical0 = all(torch.equal(sd0_b1[k], sd0_d[k]) for k in sd0_b1)
print(f"client 0 (non-dslr) state_dict identical b1 vs d: {identical0}")
print("  " + ("PASS -- domain scoping correct" if identical0 else "FAIL"))

gc_identical = all(torch.equal(ckpt_b1["global_classifier_state"][k], ckpt_d["global_classifier_state"][k])
                    for k in ckpt_b1["global_classifier_state"])
disc_identical = all(torch.equal(ckpt_b1["discriminator_state"][k], ckpt_d["discriminator_state"][k])
                      for k in ckpt_b1["discriminator_state"])
print(f"global_classifier_state identical b1 vs d: {gc_identical}")
print(f"discriminator_state identical b1 vs d: {disc_identical}")
print("  " + ("PASS -- mixing does not leak outside model.features" if (gc_identical and disc_identical) else "FAIL"))

print()
print("Smoke test finished.")

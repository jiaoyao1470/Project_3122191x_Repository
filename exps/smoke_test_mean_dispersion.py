import sys
import os
sys.path.insert(0, os.getcwd())

import torch

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


def make_args():
    sys.argv = ["smoke_test_mean_dispersion"]
    args = args_parser()
    args.exp = 1
    args.mode = "ours"
    args.iters = 3
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


print("=" * 78)
print("PHASE (a): omit use_mean_dispersion entirely")
args = make_args()
tl, cd, testl = build(args)
_, acc_a, names = ours(args, tl, testl, cd)

print("=" * 78)
print("PHASE (b): use_mean_dispersion=False explicitly -- must be byte-identical to (a)")
args = make_args()
tl, cd, testl = build(args)
_, acc_b, _ = ours(args, tl, testl, cd, use_mean_dispersion=False)
worst = max(abs(acc_a[d][r] - acc_b[d][r]) for d in names for r in range(len(acc_a[d])))
print(f"  max |acc(a) - acc(b)| = {worst:.10f}")
print("  " + ("PASS -- explicit False is a true no-op" if worst == 0.0 else "FAIL"))

print("=" * 78)
print("PHASE (c): use_mean_dispersion=True -- must not crash, no NaN/Inf, dispersion >= floor")
args = make_args()
tl, cd, testl = build(args)
train_loss_c, acc_c, _ = ours(args, tl, testl, cd, use_mean_dispersion=True,
                               checkpoint_path=os.environ.get("TEMP", ".") + "/md_smoke_ckpt.pt",
                               checkpoint_every=1)
for d in names:
    print(f"  {d:<8} acc trajectory: {acc_c[d]}")
    if any(not torch.isfinite(torch.tensor(v)) for v in acc_c[d]):
        print("  FAIL -- non-finite accuracy")

ckpt = torch.load(os.environ.get("TEMP", ".") + "/md_smoke_ckpt.pt", map_location="cpu", weights_only=False)
disp = ckpt["global_dispersion"]
mean_md = ckpt["global_mean_md"]
print(f"  global_dispersion has {len(disp)} (class,domain) keys, global_mean_md has {len(mean_md)}")
all_finite = True
all_above_floor = True
for key, v in disp.items():
    if not torch.isfinite(v).all():
        all_finite = False
        print(f"    FAIL: non-finite dispersion at {key}")
    if (v < 1e-4 - 1e-8).any():
        all_above_floor = False
        print(f"    FAIL: dispersion below floor at {key}: min={v.min().item()}")
for key, v in mean_md.items():
    if not torch.isfinite(v).all():
        all_finite = False
        print(f"    FAIL: non-finite mean_md at {key}")
print("  " + ("PASS -- all dispersion/mean_md finite" if all_finite else "FAIL"))
print("  " + ("PASS -- all dispersion values >= var_floor" if all_above_floor else "FAIL"))
print("Smoke test finished.")

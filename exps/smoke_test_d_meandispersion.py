import sys
import os
sys.path.insert(0, os.getcwd())

import torch

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


def make_args():
    sys.argv = ["smoke_test_d_meandispersion"]
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


print("PHASE (baseline): preserve_feature_probe_state=True, use_mean_dispersion=False")
args = make_args()
tl, cd, testl = build(args)
_, acc_base, names = ours(args, tl, testl, cd,
                           preserve_feature_probe_state=True, use_mean_dispersion=False)
for d in names:
    print(f"  {d:<8}: {acc_base[d]}")

print("PHASE (meandispersion): preserve_feature_probe_state=True, use_mean_dispersion=True -- "
      "COMBINATION never tested together before")
args = make_args()
tl, cd, testl = build(args)
_, acc_md, _ = ours(args, tl, testl, cd,
                     checkpoint_path=os.environ.get("TEMP", ".") + "/dmd_smoke_ckpt.pt",
                     checkpoint_every=1,
                     preserve_feature_probe_state=True, use_mean_dispersion=True)
all_finite = True
for d in names:
    print(f"  {d:<8}: {acc_md[d]}")
    if any(not torch.isfinite(torch.tensor(v)) for v in acc_md[d]):
        all_finite = False
        print(f"    FAIL: non-finite accuracy in {d}")
print("  " + ("PASS -- all accuracy values finite" if all_finite else "FAIL"))

ckpt = torch.load(os.environ.get("TEMP", ".") + "/dmd_smoke_ckpt.pt", map_location="cpu", weights_only=False)
disp = ckpt["global_dispersion"]
mean_md = ckpt["global_mean_md"]
n_keys = len(disp)
print(f"  global_dispersion keys: {n_keys} (expect up to 40 = 10 classes x 4 domains)")
disp_finite = all(torch.isfinite(v).all() for v in disp.values())
disp_floor_ok = all((v >= 1e-4 - 1e-8).all() for v in disp.values())
mean_finite = all(torch.isfinite(v).all() for v in mean_md.values())
print("  " + ("PASS -- dispersion/mean_md finite" if (disp_finite and mean_finite) else "FAIL"))
print("  " + ("PASS -- dispersion >= floor" if disp_floor_ok else "FAIL"))

print("Smoke test finished.")

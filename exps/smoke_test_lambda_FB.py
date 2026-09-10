import sys
import os
sys.path.insert(0, os.getcwd())

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


sys.argv = ["smoke_test_lambda_FB"]


def build_args():
    args = args_parser()
    args.exp = 1
    args.mode = "ours"
    args.iters = 3
    args.lr = 0.01
    args.number_workers = 0
    args.device = "cpu"
    args.dataset = "office"
    args.seed = 0
    set_seed(args)
    args.num_classes = 10
    args.size = 64
    args.batch = 32
    args.wk_iters = 1
    args.adcol_epoch = 1
    args.adcol_mu = 0.1
    args.adcol_beta = 0.1
    args.domain_keyed_proto = True
    args.no_discriminator_fix = False
    return args


scratch = os.environ.get("TEMP", ".") + "/l2_lambda_FB_smoke_test/"
if os.path.exists(scratch):
    import shutil as _shutil
    _shutil.rmtree(scratch)
os.makedirs(scratch, exist_ok=True)

print("=" * 20 + " Phase (a): feature_bank_domains not passed at all " + "=" * 20)
args_a = build_args()
tl, cd, tel = prepare_data_office_multi_clients(args_a, dslr_client_override=4)
args_a.num_users = len(tl)
_, acc_nopassed, names = ours(
    args_a, tl, tel, cd, checkpoint_path=None, acc_log_path=scratch + "a_acc.csv",
    weights_dir=scratch + "weights_a/", lambda_G=0.05,
)
print("Phase (a) complete.")

print("\n" + "=" * 20 + " Phase (b): feature_bank_domains={'dslr'}, lambda_FB=0.0 explicit " + "=" * 20)
args_b = build_args()
tl2, cd2, tel2 = prepare_data_office_multi_clients(args_b, dslr_client_override=4)
args_b.num_users = len(tl2)
_, acc_zero, names2 = ours(
    args_b, tl2, tel2, cd2, checkpoint_path=None, acc_log_path=scratch + "b_acc.csv",
    weights_dir=scratch + "weights_b/", lambda_G=0.05,
    feature_bank_domains={"dslr"}, lambda_FB=0.0,
)
print("Phase (b) complete.")

assert names == names2
all_match = True
for d in names:
    if acc_nopassed[d] != acc_zero[d]:
        all_match = False
        print(f"  MISMATCH on {d}: nopassed={acc_nopassed[d]}  zero={acc_zero[d]}")
print(f"Zero-side-effect check (feature_bank_domains omitted vs {{'dslr'}}+lambda_FB=0.0 explicit): "
      f"{'PASS' if all_match else 'FAIL'}")

print("\n" + "=" * 20 + " Phase (c): feature_bank_domains={'dslr'}, lambda_FB=0.05 -- crash + checkpoint " + "=" * 20)
args_c = build_args()
tl3, cd3, tel3 = prepare_data_office_multi_clients(args_c, dslr_client_override=4)
args_c.num_users = len(tl3)
ckpt_path = scratch + "c_checkpoint.pt"
_, acc_c, names3 = ours(
    args_c, tl3, tel3, cd3, checkpoint_path=ckpt_path, checkpoint_every=1,
    acc_log_path=scratch + "c_acc.csv", weights_dir=scratch + "weights_c/", lambda_G=0.05,
    feature_bank_domains={"dslr"}, lambda_FB=0.05,
)
print("Phase (c): 3 rounds with feature_bank_domains={'dslr'}, lambda_FB=0.05 completed -> PASS")
for d in names3:
    print(f"  {d}: {[round(a, 4) for a in acc_c[d]]}")

import torch
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
fb = ckpt.get("feature_bank_prev", {})
assert "dslr" in fb, "feature_bank_prev missing 'dslr' key after a round with feature_bank_domains={'dslr'}"
dslr_bank = fb["dslr"]
print(f"  feature_bank_prev['dslr'] populated for clients: {sorted(dslr_bank.keys())}")
sizes = {k: v[0].shape[0] for k, v in dslr_bank.items()}
print(f"  per-client bank-source sizes (this round's own features, pre-exclusion): {sizes}")
expected_total = sum(sizes.values())
for k, n in sizes.items():
    peer_expected = expected_total - n
    print(f"    client {k}: own={n}, peer-bank-if-probed={peer_expected} "
          f"(expect {{6:94,7:94,8:95,9:95}}[{k}] if unchanged)")
for feat, lab in dslr_bank.values():
    assert not feat.requires_grad, "bank feature tensor requires_grad=True -- detach discipline broken"

import shutil
probe_dir = "calib_lambda_FB_run_SMOKETEST"
os.makedirs(probe_dir, exist_ok=True)
shutil.copy(ckpt_path, os.path.join(probe_dir, "seed0_checkpoint_round2.pt"))
print(f"\n  copied smoke-test checkpoint to {probe_dir}/seed0_checkpoint_round2.pt "
      f"for a real probe_lambda_FB.py end-to-end run (point CKPT_DIR at this folder).")

print("\n" + "=" * 20 + " Phase (d): guard -- feature_bank_domains set but client_domains=None " + "=" * 20)
args_d = build_args()
tl4, cd4, tel4 = prepare_data_office_multi_clients(args_d, dslr_client_override=4)
args_d.num_users = len(tl4)
try:
    ours(
        args_d, tl4, tel4, None,
        checkpoint_path=None, acc_log_path=scratch + "d_acc.csv",
        weights_dir=scratch + "weights_d/", feature_bank_domains={"dslr"}, lambda_FB=0.0,
    )
    print("  FAIL: expected ValueError, but ours() returned normally")
except ValueError as e:
    print(f"  PASS: ValueError raised as expected: {e}")

print("\n=== ALL PHASES COMPLETE ===")

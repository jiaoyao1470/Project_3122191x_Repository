import sys
import os
sys.path.insert(0, os.getcwd())

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


sys.argv = ["smoke_test_lambda_G"]

def build_args():
    args = args_parser()
    args.exp = 1
    args.mode = "ours"
    args.iters = 2
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
    return args

scratch = os.environ.get("TEMP", ".") + "/l2_lambda_G_smoke_test/"
os.makedirs(scratch, exist_ok=True)

print("=" * 20 + " Phase (a): lambda_G not passed at all (old call signature) " + "=" * 20)
args_a = build_args()
tl, cd, tel = prepare_data_office_multi_clients(args_a)
args_a.num_users = len(tl)
_, acc_nopassed, names = ours(
    args_a, tl, tel, cd, checkpoint_path=None, acc_log_path=scratch + "a_acc.csv",
    weights_dir=scratch + "weights_a/",
)
print("Phase (a) complete.")

print("\n" + "=" * 20 + " Phase (b): lambda_G=0.0 explicitly passed " + "=" * 20)
args_b = build_args()
tl2, cd2, tel2 = prepare_data_office_multi_clients(args_b)
args_b.num_users = len(tl2)
_, acc_zero, names2 = ours(
    args_b, tl2, tel2, cd2, checkpoint_path=None, acc_log_path=scratch + "b_acc.csv",
    weights_dir=scratch + "weights_b/", lambda_G=0.0,
)
print("Phase (b) complete.")

assert names == names2
all_match = True
for d in names:
    if acc_nopassed[d] != acc_zero[d]:
        all_match = False
        print(f"  MISMATCH on {d}: nopassed={acc_nopassed[d]}  zero={acc_zero[d]}")
print(f"Zero-side-effect check (lambda_G omitted vs explicit 0.0): {'PASS' if all_match else 'FAIL'}")

print("\n" + "=" * 20 + " Phase (c): lambda_G=0.05 (calibrated value), crash check only " + "=" * 20)
args_c = build_args()
tl3, cd3, tel3 = prepare_data_office_multi_clients(args_c)
args_c.num_users = len(tl3)
_, acc_c, names3 = ours(
    args_c, tl3, tel3, cd3, checkpoint_path=None, acc_log_path=scratch + "c_acc.csv",
    weights_dir=scratch + "weights_c/", lambda_G=0.05,
)
print("Phase (c): 2 rounds with lambda_G=0.05 completed without crashing -> PASS")
for d in names3:
    print(f"  {d}: {[round(a,4) for a in acc_c[d]]}")

print("\n=== ALL PHASES COMPLETE ===")

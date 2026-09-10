import sys
import os
sys.path.insert(0, os.getcwd())

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


sys.argv = ["smoke_test_lambda_G_supportaware"]


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


scratch = os.environ.get("TEMP", ".") + "/l2_lambda_G_supportaware_smoke_test/"
os.makedirs(scratch, exist_ok=True)

print("=" * 20 + " Phase (a): scalar lambda_G=0.05 " + "=" * 20)
args_a = build_args()
tl, cd, tel = prepare_data_office_multi_clients(args_a)
args_a.num_users = len(tl)
_, acc_scalar, names = ours(
    args_a, tl, tel, cd, checkpoint_path=None, acc_log_path=scratch + "a_acc.csv",
    weights_dir=scratch + "weights_a/", lambda_G=0.05,
)
print("Phase (a) complete.")

print("\n" + "=" * 20 + " Phase (b): dict lambda_G, EVERY domain=0.05 (recorded accuracies must match phase (a)) " + "=" * 20)
args_b = build_args()
tl2, cd2, tel2 = prepare_data_office_multi_clients(args_b)
args_b.num_users = len(tl2)
uniform_dict = {d: 0.05 for d in set(cd2)}
print(f"  uniform dict = {uniform_dict}")
_, acc_uniform_dict, names2 = ours(
    args_b, tl2, tel2, cd2, checkpoint_path=None, acc_log_path=scratch + "b_acc.csv",
    weights_dir=scratch + "weights_b/", lambda_G=uniform_dict,
)
print("Phase (b) complete.")

assert names == names2
all_match = True
for d in names:
    if acc_scalar[d] != acc_uniform_dict[d]:
        all_match = False
        print(f"  MISMATCH on {d}: scalar={acc_scalar[d]}  uniform_dict={acc_uniform_dict[d]}")
print(f"\nRecorded-accuracy equivalence check (scalar 0.05 vs uniform dict all=0.05): "
      f"{'PASS' if all_match else 'FAIL'}")
assert all_match, "dict-mode resolution does not match scalar-mode for equal values -- do not trust the dict path yet"

print("\n" + "=" * 20 + " Phase (c): support-aware dict, genuinely DIFFERENT per-domain values " + "=" * 20)
args_c = build_args()
tl3, cd3, tel3 = prepare_data_office_multi_clients(args_c)
args_c.num_users = len(tl3)
from collections import defaultdict
N_d, M_d = defaultdict(int), defaultdict(int)
for client_idx, domain in enumerate(cd3):
    M_d[domain] += 1
    N_d[domain] += len(tl3[client_idx].sampler.indices)
S_d = {d: N_d[d] / M_d[d] for d in N_d}
S_max = max(S_d.values())
supportaware_dict = {d: 0.05 * S_d[d] / S_max for d in S_d}
print(f"  support-aware dict = {supportaware_dict}")

expected_lambda_G = {"amazon": 0.05000, "caltech": 0.03908, "webcam": 0.03094, "dslr": 0.00411}
for d, expected in expected_lambda_G.items():
    actual = supportaware_dict[d]
    assert abs(actual - expected) < 1e-4, (
        f"FORMULA CHECK FAIL: {d} lambda_G={actual:.5f}, expected~={expected:.5f}"
    )
print("Formula correctness check (computed lambda_G matches known reference values): PASS")

_, acc_supportaware, names3 = ours(
    args_c, tl3, tel3, cd3, checkpoint_path=None, acc_log_path=scratch + "c_acc.csv",
    weights_dir=scratch + "weights_c/", lambda_G=supportaware_dict,
)
print("Phase (c): 2 rounds with genuinely different per-domain lambda_G completed without crashing -> PASS")
for d in names3:
    print(f"  {d}: {[round(a, 4) for a in acc_supportaware[d]]}")

print("\n=== ALL PHASES COMPLETE ===")

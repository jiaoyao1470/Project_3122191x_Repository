import sys
import os
sys.path.insert(0, os.getcwd())

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


def make_args():
    sys.argv = ["smoke_test_preserve_flag_wiring"]
    args = args_parser()
    args.exp = 1
    args.mode = "ours"
    args.iters = 2
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


print("PHASE (a): omit preserve_feature_probe_state entirely")
args = make_args()
tl, cd, testl = build(args)
_, acc_a, names = ours(args, tl, testl, cd)

print("PHASE (b): preserve_feature_probe_state=False explicitly -- must equal (a)")
args = make_args()
tl, cd, testl = build(args)
_, acc_b, _ = ours(args, tl, testl, cd, preserve_feature_probe_state=False)
worst_ab = max(abs(acc_a[d][r] - acc_b[d][r]) for d in names for r in range(len(acc_a[d])))
print(f"  max |acc(a)-acc(b)| = {worst_ab:.10f}")
print("  " + ("PASS -- default False is a true no-op" if worst_ab == 0.0 else "FAIL"))

print("PHASE (c): preserve_feature_probe_state=True -- must not crash, and must DIFFER from (a)")
args = make_args()
tl, cd, testl = build(args)
_, acc_c, _ = ours(args, tl, testl, cd, preserve_feature_probe_state=True)
worst_ac = max(abs(acc_a[d][r] - acc_c[d][r]) for d in names for r in range(len(acc_a[d])))
print(f"  max |acc(a)-acc(c)| = {worst_ac:.10f}")
print("  " + ("PASS -- True produces a different trajectory (flag has real effect)" if worst_ac > 0.0
              else "FAIL -- True and False produced identical results; flag may not be wired"))
print("Smoke test finished.")

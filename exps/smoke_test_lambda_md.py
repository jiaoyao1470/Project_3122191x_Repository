import sys
import os
sys.path.insert(0, os.getcwd())

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


def make_args():
    sys.argv = ["smoke_test_lambda_md"]
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


print("PHASE (a): omit lambda_md entirely, use_mean_dispersion=True")
args = make_args()
tl, cd, testl = build(args)
_, acc_a, names = ours(args, tl, testl, cd, preserve_feature_probe_state=True, use_mean_dispersion=True)

print("PHASE (b): lambda_md=None explicitly, use_mean_dispersion=True -- must equal (a)")
args = make_args()
tl, cd, testl = build(args)
_, acc_b, _ = ours(args, tl, testl, cd, preserve_feature_probe_state=True, use_mean_dispersion=True, lambda_md=None)
worst_ab = max(abs(acc_a[d][r] - acc_b[d][r]) for d in names for r in range(len(acc_a[d])))
print(f"  max |acc(a)-acc(b)| = {worst_ab:.10f}")
print("  " + ("PASS -- default None is a true no-op" if worst_ab == 0.0 else "FAIL"))

print("PHASE (c): lambda_md=2.4649, use_mean_dispersion=True -- must DIFFER from (a)")
args = make_args()
tl, cd, testl = build(args)
_, acc_c, _ = ours(args, tl, testl, cd, preserve_feature_probe_state=True, use_mean_dispersion=True, lambda_md=2.4649)
worst_ac = max(abs(acc_a[d][r] - acc_c[d][r]) for d in names for r in range(len(acc_a[d])))
print(f"  max |acc(a)-acc(c)| = {worst_ac:.10f}")
print("  " + ("PASS -- calibrated lambda_md changes the trajectory" if worst_ac > 0.0 else "FAIL"))
print("Smoke test finished.")

import sys
import os
sys.path.insert(0, os.getcwd())

import copy
import torch

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


sys.argv = ["smoke_test_input_mixup"]
args = args_parser()
args.exp = 1
args.mode = "ours"
args.lr = 0.01
args.number_workers = 0
args.device = "cpu"
args.dataset = "office"
args.seed = 0
args.num_classes = 10
args.size = 64
args.batch = 32
args.domain_keyed_proto = True
args.no_discriminator_fix = False
args.iters = 2
args.wk_iters = 1
args.adcol_epoch = 1
args.adcol_mu = 0.1
args.adcol_beta = 0.1

set_seed(args)
train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=4)
args.num_users = len(train_loader_list)

set_seed(args)
_, acc_omitted, names = ours(args, copy.deepcopy(train_loader_list), test_loader_list, client_domains)
set_seed(args)
_, acc_explicit_default, _ = ours(
    args, copy.deepcopy(train_loader_list), test_loader_list, client_domains,
    input_mixup_domains=None, lambda_input_mixup=0.0, input_mixup_alpha=0.2,
)
noop_ok = True
for d in names:
    same = acc_omitted[d] == acc_explicit_default[d]
    noop_ok &= same
    print(f"  no-op check [{d}]: omitted={acc_omitted[d]}  explicit_default={acc_explicit_default[d]}  match={same}")
print(f"NO-OP CHECK: {'PASS' if noop_ok else 'FAIL'}")

set_seed(args)
_, acc_mixup_on, _ = ours(
    args, copy.deepcopy(train_loader_list), test_loader_list, client_domains,
    input_mixup_domains={"dslr"}, lambda_input_mixup=0.5, input_mixup_alpha=0.2,
)
print()
for d in names:
    control = acc_omitted[d]
    treated = acc_mixup_on[d]
    diff = [round(t - c, 4) for t, c in zip(treated, control)]
    print(f"  domain={d:<8} control={control}  mixup_on={treated}  diff={diff}")

isolation_ok = all(acc_omitted[d] == acc_mixup_on[d] for d in names if d != "dslr")
fires_ok = acc_omitted["dslr"] != acc_mixup_on["dslr"]
print(f"ISOLATION CHECK (non-dslr domains untouched): {'PASS' if isolation_ok else 'FAIL'}")
print(f"MECHANISM-FIRES CHECK (dslr differs from control): {'PASS' if fires_ok else 'FAIL'}")

print()
print(f"OVERALL: {'ALL PASS' if (noop_ok and isolation_ok and fires_ok) else 'FAIL -- see above'}")

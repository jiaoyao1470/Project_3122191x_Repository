import sys
import os
import copy
sys.path.insert(0, os.getcwd())
sys.argv = ["smoke_test_mixup_threshold"]

import torch

from option import args_parser
from federated_main import set_seed
from util import prepare_data_office_multi_clients
from models import adcol_model, Discriminator, classifier_model
from update import LocalUpdate

CLIENT = 6
THR_LOW = 0
THR_HIGH = 5


def build(thr):
    a = args_parser()
    a.exp = 1
    a.mode = "ours"
    a.lr = 0.01
    a.number_workers = 0
    a.device = "cpu"
    a.dataset = "office"
    a.seed = 0
    a.num_classes = 10
    a.size = 64
    a.batch = 32
    a.wk_iters = 1
    a.adcol_mu = 0.1
    a.adcol_beta = 0.1
    a.domain_keyed_proto = True
    a.mixup_support_threshold = thr
    return a


a0 = build(THR_LOW)
set_seed(a0)
train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(a0)
a0.num_users = len(train_loader_list)
assert client_domains[CLIENT] == "dslr", f"expected client{CLIENT} to be dslr, got {client_domains[CLIENT]}"
print(f"client_domains: {client_domains}")
print(f"testing client{CLIENT} (domain={client_domains[CLIENT]})")

set_seed(a0)
gproto = {c: torch.randn(2048) for c in range(10)}


def run(thr):
    a = build(thr)
    a.num_users = len(train_loader_list)
    set_seed(a)
    m = adcol_model(num_classes=10)
    d = Discriminator(m, 4)
    gc = classifier_model(a, 10)
    node = LocalUpdate(args=a)
    set_seed(a)
    out, _ = node.update_weights_ours_debug(
        1, model=copy.deepcopy(m), discriminator=copy.deepcopy(d),
        classifier_model=copy.deepcopy(gc), train_loader=train_loader_list[CLIENT],
        global_proto=gproto, num_domains=4, momentum=a.momentum,
        global_G=None, lambda_G=0.05, lambda_mixup=0.1,
    )
    return out


print(f"\n--- threshold={THR_LOW} ---")
m_low = run(THR_LOW)
print(f"--- threshold={THR_HIGH} ---")
m_high = run(THR_HIGH)

sd_low, sd_high = m_low.state_dict(), m_high.state_dict()
diffs = {
    k: (sd_low[k].float() - sd_high[k].float()).abs().max().item()
    for k in sd_low if sd_low[k].dtype.is_floating_point
}
max_diff = max(diffs.values())
differing = {k: v for k, v in diffs.items() if v > 1e-9}

print(f"\nmax |weight difference| across all params: {max_diff:.3e}")
print(f"params that differ (>1e-9): {len(differing)} / {len(diffs)}")
for k in list(differing)[:6]:
    print(f"   {k}: {differing[k]:.3e}")

assert max_diff > 1e-9, (
    f"threshold={THR_HIGH} produced IDENTICAL weights to threshold={THR_LOW} -- the broadened "
    f"trigger is not reaching the loss. Check mixup_support_threshold plumbing in update.py."
)
print(f"\nSMOKE TEST PASSED: mixup_support_threshold is genuinely wired into training "
      f"(thr={THR_LOW} vs thr={THR_HIGH} give different models)")

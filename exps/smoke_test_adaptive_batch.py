import sys
import os
import copy
import math
sys.path.insert(0, os.getcwd())
sys.argv = ["smoke_test_adaptive_batch"]

import torch

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


def build(adaptive):
    a = args_parser()
    a.exp = 1
    a.mode = "ours"
    a.iters = 2
    a.lr = 0.01
    a.number_workers = 0
    a.device = "cpu"
    a.dataset = "office"
    a.seed = 0
    a.num_classes = 10
    a.size = 64
    a.batch = 32
    a.wk_iters = 1
    a.adcol_epoch = 1
    a.adcol_mu = 0.1
    a.adcol_beta = 0.1
    a.domain_keyed_proto = True
    a.adaptive_local_batch = adaptive
    return a


print("=" * 20 + " PART A: loader construction " + "=" * 20)
a_off = build(False)
set_seed(a_off)
tl_off, cd_off, _ = prepare_data_off = prepare_data_office_multi_clients(a_off)

a_on = build(True)
set_seed(a_on)
tl_on, cd_on, _ = prepare_data_office_multi_clients(a_on)

assert cd_off == cd_on, "PART A FAIL: client_domains differ between the two settings"

print(f"  {'client':<7}{'domain':<9}{'n_i':>5}{'B(off)':>8}{'nb(off)':>9}{'B(on)':>7}{'nb(on)':>8}")
n_changed = 0
for i, d in enumerate(cd_off):
    n_i = len(tl_off[i].sampler.indices)
    assert n_i == len(tl_on[i].sampler.indices), f"PART A FAIL: client {i} got a different subset"
    b_off, b_on = tl_off[i].batch_size, tl_on[i].batch_size
    nb_off, nb_on = len(tl_off[i]), len(tl_on[i])
    mark = "  <-- CHANGED" if b_off != b_on else ""
    if b_off != b_on:
        n_changed += 1
    print(f"  {i:<7}{d:<9}{n_i:>5}{b_off:>8}{nb_off:>9}{b_on:>7}{nb_on:>8}{mark}")

    assert b_off == a_off.batch, f"PART A FAIL: OFF path client {i} has batch {b_off}, expected {a_off.batch}"
    expected_on = min(a_on.batch, math.ceil(n_i / 2))
    assert b_on == expected_on, f"PART A FAIL: ON path client {i} has batch {b_on}, rule says {expected_on}"
    assert nb_on >= 2, f"PART A FAIL: client {i} still has {nb_on} minibatch(es) with the rule on"

assert n_changed > 0, "PART A FAIL: the rule changed no client at all"
changed_domains = {cd_off[i] for i in range(len(cd_off)) if tl_off[i].batch_size != tl_on[i].batch_size}
assert changed_domains == {"dslr"}, (
    f"PART A FAIL: expected the rule to be self-targeting to dslr on this partition, but it "
    f"changed {changed_domains}"
)
print(f"\nPART A PASS: {n_changed} client(s) changed, all of them dslr; every client now has >= 2 "
      f"minibatches; the OFF path is untouched at the global batch size")

print("\n" + "=" * 20 + " PART B: flag reaches training " + "=" * 20)
scratch = os.environ.get("TEMP", ".") + "/alb_smoke/"
os.makedirs(scratch, exist_ok=True)


def run(adaptive):
    a = build(adaptive)
    set_seed(a)
    tl, cd, tel = prepare_data_office_multi_clients(a)
    a.num_users = len(tl)
    set_seed(a)
    print(f"--- adaptive_local_batch={adaptive} ---")
    return ours(
        a, tl, tel, cd, checkpoint_path=f"{scratch}ckpt_{adaptive}.pt", checkpoint_every=1,
        acc_log_path=f"{scratch}acc_{adaptive}.csv", weights_dir=f"{scratch}w_{adaptive}/",
        lambda_G=0.05,
    )


for flag in (False, True):
    p = f"{scratch}ckpt_{flag}.pt"
    if os.path.exists(p):
        os.remove(p)

run(False)
run(True)

ck_off = torch.load(f"{scratch}ckpt_False.pt", map_location="cpu", weights_only=False)
ck_on = torch.load(f"{scratch}ckpt_True.pt", map_location="cpu", weights_only=False)

DSLR_CLIENT = 6
m_off = ck_off["local_models_state"][DSLR_CLIENT]
m_on = ck_on["local_models_state"][DSLR_CLIENT]
diffs = {
    k: (m_off[k].float() - m_on[k].float()).abs().max().item()
    for k in m_off if m_off[k].dtype.is_floating_point
}
max_diff = max(diffs.values())
n_diff = sum(1 for v in diffs.values() if v > 1e-9)
print(f"\ndslr client {DSLR_CLIENT}: max |weight difference| = {max_diff:.3e}, "
      f"params differing = {n_diff}/{len(diffs)}")
assert max_diff > 1e-9, (
    "PART B FAIL: adaptive_local_batch=True produced an IDENTICAL dslr model -- the flag is not "
    "reaching training."
)

CALTECH_CLIENT = 0
c_off = ck_off["local_models_state"][CALTECH_CLIENT]
c_on = ck_on["local_models_state"][CALTECH_CLIENT]
c_max = max(
    (c_off[k].float() - c_on[k].float()).abs().max().item()
    for k in c_off if c_off[k].dtype.is_floating_point
)
print(f"caltech client {CALTECH_CLIENT}: max |weight difference| = {c_max:.3e} "
      f"(its own loader is unchanged; any difference here is indirect, via the shared "
      f"discriminator/global classifier)")

print("\nPART B PASS: adaptive_local_batch genuinely changes dslr's local training")
print("\nALL TESTS PASSED")

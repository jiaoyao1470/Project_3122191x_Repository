import sys
import os
import copy
sys.path.insert(0, os.getcwd())
sys.argv = ["smoke_test_domain_balanced_gc"]

import torch

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


def build():
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
    return a


a0 = build()
set_seed(a0)
train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(a0)
a0.num_users = len(train_loader_list)

print("=" * 20 + " PART A: weight vector (no training) " + "=" * 20)
client_sample_counts = [len(train_loader_list[i].sampler.indices) for i in range(a0.num_users)]
domain_sample_counts = {}
for client_idx, d in enumerate(client_domains):
    domain_sample_counts[d] = domain_sample_counts.get(d, 0) + client_sample_counts[client_idx]

n_total = sum(domain_sample_counts.values())
n_domains = len(domain_sample_counts)
weight_chunks = []
domain_of_sample = []
for client_idx, d in enumerate(client_domains):
    n_client = client_sample_counts[client_idx]
    scaled_w = n_total / (n_domains * domain_sample_counts[d])
    weight_chunks.append(torch.full((n_client,), scaled_w))
    domain_of_sample.extend([d] * n_client)
sample_weights = torch.cat(weight_chunks, dim=0)

mean_w = sample_weights.mean().item()
print(f"mean per-sample weight = {mean_w:.8f} (must be 1.0 -- this is what keeps the gradient "
      f"scale equal to the default path's plain .mean())")
assert abs(mean_w - 1.0) < 1e-5, f"PART A FAIL: mean weight is {mean_w}, expected 1.0"
per_domain_w = {d: n_total / (n_domains * n) for d, n in domain_sample_counts.items()}
print(f"per-sample weight by domain: { {d: round(v, 4) for d, v in per_domain_w.items()} }")

print(f"pooled samples per domain: {domain_sample_counts}")
total_by_domain = {}
for w, d in zip(sample_weights.tolist(), domain_of_sample):
    total_by_domain[d] = total_by_domain.get(d, 0.0) + w
print("total weight per domain (should all be equal):")
for d, t in total_by_domain.items():
    print(f"  {d:10s} {t:.10f}")

vals = list(total_by_domain.values())
expected_total = n_total / n_domains
assert (max(vals) - min(vals)) / expected_total < 1e-4, (
    f"PART A FAIL: domains do not carry equal total weight: {total_by_domain}"
)
assert all(abs(v - expected_total) / expected_total < 1e-4 for v in vals), (
    f"PART A FAIL: each domain's total weight should be N_total/D = {expected_total}: {total_by_domain}"
)

default_share = {d: n / sum(domain_sample_counts.values()) for d, n in domain_sample_counts.items()}
balanced_share = {d: t / sum(vals) for d, t in total_by_domain.items()}
print("\ninfluence share on the global classifier:")
print(f"  {'domain':10s}{'default (per-sample)':>22s}{'domain-balanced':>18s}")
for d in domain_sample_counts:
    print(f"  {d:10s}{default_share[d]:>21.1%}{balanced_share[d]:>18.1%}")
assert abs(default_share["dslr"] - balanced_share["dslr"]) > 0.05, (
    "PART A FAIL: the balanced allocation is not meaningfully different from the default"
)
print("\nPART A PASS: every domain carries equal total weight, and this differs materially from "
      "the default sample-count-proportional allocation")

print("\n" + "=" * 20 + " PART B: flag actually changes training " + "=" * 20)
scratch = os.environ.get("TEMP", ".") + "/dbgc_smoke/"
os.makedirs(scratch, exist_ok=True)


def run(flag):
    a = build()
    set_seed(a)
    tl, cd, tel = prepare_data_office_multi_clients(a)
    a.num_users = len(tl)
    set_seed(a)
    print(f"--- domain_balanced_gc={flag} ---")
    return ours(
        a, tl, tel, cd, checkpoint_path=f"{scratch}ckpt_{flag}.pt", checkpoint_every=1,
        acc_log_path=f"{scratch}acc_{flag}.csv", weights_dir=f"{scratch}w_{flag}/",
        lambda_G=0.05, domain_balanced_gc=flag,
    )


for flag in (False, True):
    p = f"{scratch}ckpt_{flag}.pt"
    if os.path.exists(p):
        os.remove(p)

run(False)
run(True)

ck_off = torch.load(f"{scratch}ckpt_False.pt", map_location="cpu", weights_only=False)
ck_on = torch.load(f"{scratch}ckpt_True.pt", map_location="cpu", weights_only=False)
gc_off, gc_on = ck_off["global_classifier_state"], ck_on["global_classifier_state"]
diffs = {
    k: (gc_off[k].float() - gc_on[k].float()).abs().max().item()
    for k in gc_off if gc_off[k].dtype.is_floating_point
}
max_diff = max(diffs.values())
n_diff = sum(1 for v in diffs.values() if v > 1e-9)
print(f"\nglobal_classifier: max |weight difference| = {max_diff:.3e}, params differing = {n_diff}/{len(diffs)}")
assert max_diff > 1e-9, (
    "PART B FAIL: domain_balanced_gc=True produced an IDENTICAL global classifier -- the flag is "
    "not reaching the objective."
)
print("PART B PASS: domain_balanced_gc genuinely changes the global classifier")

print("\nALL TESTS PASSED")

import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


sys.argv = ["smoke_test_information_oracle"]


def build_args(iters):
    a = args_parser()
    a.exp = 1
    a.mode = "ours"
    a.iters = iters
    a.lr = 0.01
    a.number_workers = 0
    a.dataset = "office"
    a.num_classes = 10
    a.size = 64
    a.batch = 32
    a.wk_iters = 2
    a.adcol_mu = 0.1
    a.adcol_beta = 0.1
    a.adcol_epoch = 1
    a.no_discriminator_fix = True
    a.device = "cpu"
    a.seed = 0
    return a


ORACLE_DOMAIN = "dslr"
failures = []


def check(cond, label):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    if not cond:
        failures.append(label)


args_ref = build_args(iters=2)
set_seed(args_ref)
ref_loaders, ref_domains, _ = prepare_data_office_multi_clients(args_ref, dslr_client_override=4)
ref_snapshot = {
    i: (type(l.sampler).__name__, list(l.sampler.indices), l.batch_size, len(l))
    for i, l in enumerate(ref_loaders)
}
ref_loader_size = [len(l.dataset) for l in ref_loaders]
ref_client_weights = [s / sum(ref_loader_size) for s in ref_loader_size]

args = build_args(iters=2)
set_seed(args)
train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=4)
args.num_users = len(train_loader_list)

oracle_idx = []
for i, dom in enumerate(client_domains):
    if dom != ORACLE_DOMAIN:
        continue
    ds = train_loader_list[i].dataset
    n_i = len(train_loader_list[i].sampler.indices)
    train_loader_list[i] = DataLoader(
        ds, batch_size=args.batch,
        sampler=RandomSampler(ds, replacement=False, num_samples=n_i),
        num_workers=args.number_workers, pin_memory=True,
    )
    oracle_idx.append(i)

print(f"client_domains: {client_domains}")
print(f"dslr client indices: {oracle_idx}")

print("\nPART A -- untouched domains identical to un-swapped baseline")
check(client_domains == ref_domains, "client_domains unchanged")
for i, dom in enumerate(client_domains):
    if dom == ORACLE_DOMAIN:
        continue
    ref_type, ref_indices, ref_bs, ref_nb = ref_snapshot[i]
    l = train_loader_list[i]
    ok = (
        type(l.sampler).__name__ == ref_type == "SubsetRandomSampler"
        and list(l.sampler.indices) == ref_indices
        and l.batch_size == ref_bs
        and len(l) == ref_nb
    )
    check(ok, f"client {i} ({dom}): sampler/indices/batch/batches identical "
              f"(n={len(ref_indices)}, batch={ref_bs}, batches={ref_nb})")

print("\nPART B -- dslr loaders have the intended shape")
for i in oracle_idx:
    l = train_loader_list[i]
    s = l.sampler
    ref_n = len(ref_snapshot[i][1])
    ok = (
        type(s).__name__ == "RandomSampler"
        and s.replacement is False
        and len(l.dataset) == 126
        and s.num_samples == ref_n
        and l.batch_size == 32
        and len(l) == 1
    )
    check(ok, f"client {i}: RandomSampler pool={len(l.dataset)} num_samples={s.num_samples} "
              f"(orig partition {ref_n}) batch={l.batch_size} batches/epoch={len(l)} "
              f"replacement={s.replacement}")

print("\nPART C -- the manipulation actually widens information coverage")
for i in oracle_idx:
    s = train_loader_list[i].sampler
    draws = [list(iter(s)) for _ in range(10)]
    sizes_ok = all(len(d) == s.num_samples for d in draws)
    no_dupes = all(len(set(d)) == len(d) for d in draws)
    differ = draws[0] != draws[1]
    union = len(set().union(*[set(d) for d in draws]))
    check(sizes_ok and no_dupes and differ and union > 2 * s.num_samples,
          f"client {i}: 10 draws x {s.num_samples} -> {union}/126 unique "
          f"(fixed-partition baseline would be {s.num_samples}/126); "
          f"per-draw distinct={no_dupes}, draws differ={differ}")

ref_i = oracle_idx[0]
ref_sampler = ref_loaders[ref_i].sampler
ref_draws = [set(iter(ref_sampler)) for _ in range(10)]
ref_union = len(set().union(*ref_draws))
check(ref_union == len(ref_snapshot[ref_i][1]),
      f"control: un-swapped client {ref_i} still reaches only {ref_union}/126 across 10 draws")

print("\nPART D -- not silently the ALB experiment")
check(getattr(args, "adaptive_local_batch", False) is False, "adaptive_local_batch is False")
check(all(train_loader_list[i].batch_size == 32 for i in oracle_idx), "dslr batch size still 32 (ALB would make it 16)")

print("\nPART E -- client_weights regression guard (non-key: dead code in ours())")
loader_size = [len(l.dataset) for l in train_loader_list]
client_weights = [s / sum(loader_size) for s in loader_size]
check(client_weights == ref_client_weights,
      "client_weights identical (len(dataset) is the whole domain, so the swap cannot move it)")

print("\nPART F -- end-to-end run + resume")
tmp_dir = "C:/Users/jiaya/AppData/Local/Temp/claude/smoke_test_information_oracle"
os.makedirs(tmp_dir, exist_ok=True)
ckpt_path = f"{tmp_dir}/checkpoint.pt"
if os.path.exists(ckpt_path):
    os.remove(ckpt_path)

train_loss, accuracy_list, datasets_name = ours(
    args, train_loader_list, test_loader_list, client_domains,
    checkpoint_path=ckpt_path, checkpoint_every=1,
    acc_log_path=f"{tmp_dir}/acc.csv", weights_dir=f"{tmp_dir}/weights/",
    lambda_G=0.0,
)
check(all(len(accuracy_list[d]) == 2 for d in datasets_name), "2 rounds completed")
for d in datasets_name:
    print(f"      {d:<8}: {[round(x, 4) for x in accuracy_list[d]]}")

ckpt = torch.load(ckpt_path, weights_only=False)
check(len(ckpt["local_models_state"]) == args.num_users,
      f"checkpoint holds {len(ckpt['local_models_state'])} local models (expect {args.num_users})")

args_resume = build_args(iters=3)
set_seed(args_resume)
rl, rd, rt = prepare_data_office_multi_clients(args_resume, dslr_client_override=4)
args_resume.num_users = len(rl)
for i, dom in enumerate(rd):
    if dom != ORACLE_DOMAIN:
        continue
    ds = rl[i].dataset
    n_i = len(rl[i].sampler.indices)
    rl[i] = DataLoader(ds, batch_size=args_resume.batch,
                       sampler=RandomSampler(ds, replacement=False, num_samples=n_i),
                       num_workers=args_resume.number_workers, pin_memory=True)

_, acc_resume, names_resume = ours(
    args_resume, rl, rt, rd,
    checkpoint_path=ckpt_path, checkpoint_every=1,
    acc_log_path=f"{tmp_dir}/acc.csv", weights_dir=f"{tmp_dir}/weights/",
    lambda_G=0.0,
)
check(all(len(acc_resume[d]) == 3 for d in names_resume), "resumed run reached 3 rounds")
check(all(np.allclose(acc_resume[d][:2], accuracy_list[d]) for d in names_resume),
      "resumed run kept rounds 0-1 from the checkpoint instead of retraining them")

print("\n" + "=" * 70)
if failures:
    print(f"SMOKE TEST FAILED -- {len(failures)} check(s):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("SMOKE TEST PASSED -- all checks green.")

import sys
import os
sys.path.insert(0, os.getcwd())

import random
import torch
import numpy as np
from torchvision import transforms as tvt
from torch.utils.data import DataLoader, SubsetRandomSampler

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients
from update import LocalUpdate
import data_utils


class PersistentRNGCompose:
    def __init__(self, transform, seed):
        self.transform = transform
        self._python_state = random.Random(seed).getstate()
        self._numpy_state = np.random.RandomState(seed).get_state()
        _torch_gen = torch.Generator(device="cpu")
        _torch_gen.manual_seed(seed)
        self._torch_state = _torch_gen.get_state().clone()

    def __call__(self, img):
        caller_python = random.getstate()
        caller_numpy = np.random.get_state()
        caller_torch = torch.get_rng_state().clone()

        try:
            random.setstate(self._python_state)
            np.random.set_state(self._numpy_state)
            torch.set_rng_state(self._torch_state)

            out = self.transform(img)

            self._python_state = random.getstate()
            self._numpy_state = np.random.get_state()
            self._torch_state = torch.get_rng_state().clone()

            return out
        finally:
            random.setstate(caller_python)
            np.random.set_state(caller_numpy)
            torch.set_rng_state(caller_torch)

sys.argv = ["smoke_test_stronglocalaug"]
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

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=4)
args.num_users = len(train_loader_list)

DATA_BASE_PATH = "../data/office_caltech_10"
transform_office_strong = tvt.Compose([
    tvt.Resize([72, 72]),
    tvt.RandomResizedCrop(args.size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    tvt.RandomHorizontalFlip(),
    tvt.RandomRotation((-30, 30)),
    tvt.RandomApply([tvt.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)], p=0.8),
    tvt.RandomGrayscale(p=0.1),
    tvt.RandomApply([tvt.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.1),
    tvt.ToTensor(),
])
AUG_DOMAIN = "dslr"
feature_loader_list = [None] * args.num_users
_strong_ds_by_client = {}
_gen_by_client = {}

_py_before = random.getstate()
_np_before = np.random.get_state()
_torch_before = torch.get_rng_state().clone()

for i, dom in enumerate(client_domains):
    if dom != AUG_DOMAIN:
        continue
    original_loader = train_loader_list[i]
    feature_loader_list[i] = original_loader
    idx = original_loader.sampler.indices
    persistent_transform = PersistentRNGCompose(transform_office_strong, seed=args.seed + 30000 + i)
    strong_ds_i = data_utils.OfficeDataset(DATA_BASE_PATH, "dslr", transform=persistent_transform)
    _strong_ds_by_client[i] = strong_ds_i
    g = torch.Generator()
    g.manual_seed(args.seed + 10000 + i)
    _gen_by_client[i] = g
    train_loader_list[i] = DataLoader(
        strong_ds_i, batch_size=args.batch, sampler=SubsetRandomSampler(idx, generator=g),
        num_workers=args.number_workers, pin_memory=True,
    )

print("=" * 78)
print("CHECK 9: constructing StrongLocalAug loaders leaves the global RNG untouched")
print("=" * 78)
_py_unchanged = random.getstate() == _py_before
_np_unchanged = np.array_equal(np.random.get_state()[1], _np_before[1])
_torch_unchanged = torch.equal(torch.get_rng_state(), _torch_before)
print(f"  python random state unchanged = {_py_unchanged}")
print(f"  numpy random state unchanged  = {_np_unchanged}")
print(f"  torch RNG state unchanged     = {_torch_unchanged}")
_check9_pass = _py_unchanged and _np_unchanged and _torch_unchanged
print("  " + ("PASS -- model init inside ours() will match D_matched's seed0 init" if _check9_pass else
              "FAIL -- constructor leaked into the global RNG, model init will NOT match D_matched"))

print()
print("=" * 78)
print("CHECK 1-5: loader/dataset construction invariants")
print("=" * 78)

all_pass = _check9_pass
for i, dom in enumerate(client_domains):
    if dom != AUG_DOMAIN:
        continue
    feat_loader = feature_loader_list[i]
    train_loader = train_loader_list[i]

    not_aliased = feat_loader.dataset is not train_loader.dataset
    print(f"client {i}: dataset objects distinct = {not_aliased}")
    all_pass &= not_aliased

    def _compose_op_names(t):
        inner = t if hasattr(t, "transforms") else t.transform
        return [type(op).__name__ for op in inner.transforms]

    feat_ops = _compose_op_names(feat_loader.dataset.transform)
    train_ops = _compose_op_names(train_loader.dataset.transform)
    feat_has_strong_only_ops = any(op in feat_ops for op in ("ColorJitter", "RandomResizedCrop", "RandomApply"))
    feat_has_baseline_ops = "RandomHorizontalFlip" in feat_ops and "RandomRotation" in feat_ops
    train_has_strong_ops = "RandomResizedCrop" in train_ops and "RandomApply" in train_ops
    print(f"client {i}: feature transform ops = {feat_ops}")
    print(f"client {i}: train   transform ops = {train_ops}")
    print(f"  feature_loader is baseline-only (no strong-only ops): {not feat_has_strong_only_ops}")
    print(f"  feature_loader has baseline flip+rotation: {feat_has_baseline_ops}")
    print(f"  train_loader has strong ops (crop+jitter): {train_has_strong_ops}")
    all_pass &= (not feat_has_strong_only_ops) and feat_has_baseline_ops and train_has_strong_ops

    feat_idx = set(feat_loader.sampler.indices.tolist() if hasattr(feat_loader.sampler.indices, "tolist") else feat_loader.sampler.indices)
    train_idx = set(train_loader.sampler.indices)
    same_indices = feat_idx == train_idx
    print(f"client {i}: identical index set (feature vs train) = {same_indices}  (n={len(feat_idx)})")
    all_pass &= same_indices

    sample_idx = next(iter(feat_idx))
    same_file = feat_loader.dataset.paths[sample_idx] == train_loader.dataset.paths[sample_idx]
    print(f"client {i}: paths[{sample_idx}] match across feature/strong dataset instances = {same_file}")
    all_pass &= same_file

print()
print("=" * 78)
print("CHECK 6: non-dslr clients untouched")
print("=" * 78)
for i, dom in enumerate(client_domains):
    if dom == AUG_DOMAIN:
        continue
    is_none = feature_loader_list[i] is None
    print(f"client {i} ({dom}): feature_loader_list entry is None = {is_none}")
    all_pass &= is_none

print()
print(f"CHECKS 1-6: {'ALL PASS' if all_pass else 'FAIL -- see above'}")

print()
print("=" * 78)
print("CHECK 8: PersistentRNGCompose survives LocalUpdate's per-round reseed")
print("=" * 78)
_probe_client = next(iter(_strong_ds_by_client.keys()))
_probe_loader = train_loader_list[_probe_client]


def _simulate_one_round():
    _ = LocalUpdate(args=args)
    return next(iter(_probe_loader))


imgs_r0, labs_r0 = _simulate_one_round()
imgs_r1, labs_r1 = _simulate_one_round()
identical = torch.equal(imgs_r0, imgs_r1) and torch.equal(labs_r0, labs_r1)
max_diff = (imgs_r0 - imgs_r1).abs().max().item()
print(f"client {_probe_client}: r0 vs r1 byte-identical = {identical}  (max pixel diff = {max_diff:.6f})")
print("  " + ("PASS -- genuinely fresh draws across simulated rounds" if not identical else
              "FAIL -- draws are repeating, PersistentRNGCompose fix did not take"))
all_pass &= not identical

print()
print("=" * 78)
print("CHECK 7: 2-round real run doesn't crash")
print("=" * 78)
_, acc, names = ours(
    args, train_loader_list, test_loader_list, client_domains,
    feature_loader_list=feature_loader_list,
    preserve_feature_probe_state=True,
)
for d in names:
    print(f"  {d:<8}: {[round(v, 4) for v in acc[d]]}")
print()
print(f"OVERALL (checks 1-6, 8, 9): {'ALL PASS' if all_pass else 'FAIL -- see above'}")
print("Smoke test finished.")

import sys
import os
sys.path.insert(0, os.getcwd())

import random
import subprocess
import numpy as np
import torch
from collections import Counter
from torchvision import transforms as tvt
from torch.utils.data import DataLoader

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients
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


class _StratifiedPoolSampler(torch.utils.data.Sampler):
    def __init__(self, pool_by_class, m_by_class, generator):
        self.pool_by_class = {c: list(idxs) for c, idxs in pool_by_class.items()}
        self.m_by_class = dict(m_by_class)
        self.generator = generator
        self.num_samples = sum(m_by_class.values())

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        batch = []
        for c, m in self.m_by_class.items():
            pool_c = self.pool_by_class[c]
            perm = torch.randperm(len(pool_c), generator=self.generator)[:m]
            batch.extend(pool_c[p] for p in perm.tolist())
        order = torch.randperm(len(batch), generator=self.generator)
        return iter(batch[o] for o in order.tolist())


EXPECTED_K = {"target2x": {6: 62, 7: 62, 8: 62, 9: 62},
              "maxeligible": {6: 108, 7: 126, 8: 126, 9: 120}}

sys.argv = ["smoke_test_ksupport_oracle"]
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
set_seed(args)

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=4)
DSLR_CLIENTS = [i for i, d in enumerate(client_domains) if d == "dslr"]
DATA_BASE_PATH = "../data/office_caltech_10"

_full_labels = train_loader_list[DSLR_CLIENTS[0]].dataset.labels
domain_class_indices = {}
for idx_, lab in enumerate(_full_labels):
    domain_class_indices.setdefault(lab, []).append(idx_)
domain_counts = {c: len(v) for c, v in domain_class_indices.items()}
client_own_indices = {i: list(train_loader_list[i].sampler.indices) for i in DSLR_CLIENTS}
client_own_by_class = {i: Counter(_full_labels[j] for j in client_own_indices[i]) for i in DSLR_CLIENTS}

all_pass = True
for dose in ("target2x", "maxeligible"):
    print("=" * 78)
    print(f"DOSE = {dose}")
    print("=" * 78)
    pool_rng = np.random.RandomState(args.seed + 40000)
    pool_by_class = {}
    achievable_K = {}
    for i in DSLR_CLIENTS:
        pool_by_class[i] = {}
        for cls, n_local in client_own_by_class[i].items():
            n_domain = domain_counts[cls]
            target = min(2 * n_local, n_domain) if dose == "target2x" else n_domain
            own_this_class = [j for j in client_own_indices[i] if _full_labels[j] == cls]
            n_extra_needed = target - n_local
            extra = []
            if n_extra_needed > 0:
                peer_candidates = [j for j in domain_class_indices[cls] if j not in client_own_indices[i]]
                chosen = pool_rng.choice(peer_candidates, size=n_extra_needed, replace=False)
                extra = [int(x) for x in chosen]
            class_pool = own_this_class + extra
            assert len(class_pool) == target
            assert len(class_pool) == len(set(class_pool))
            pool_by_class[i][cls] = class_pool
        achievable_K[i] = sum(len(v) for v in pool_by_class[i].values())

    for i in DSLR_CLIENTS:
        exp = EXPECTED_K[dose][i]
        ok = achievable_K[i] == exp
        print(f"  client {i}: achievable_K={achievable_K[i]}  expected={exp}  {'OK' if ok else 'MISMATCH!'}")
        all_pass &= ok
        pool_classes = set(pool_by_class[i].keys())
        own_classes = set(client_own_by_class[i].keys())
        class_ok = pool_classes == own_classes
        print(f"    class-set match: {class_ok}")
        all_pass &= class_ok

    i = DSLR_CLIENTS[0]
    g = torch.Generator()
    g.manual_seed(args.seed + 10000 + i)
    sampler = _StratifiedPoolSampler(pool_by_class[i], client_own_by_class[i], g)
    n_i = sum(client_own_by_class[i].values())
    print(f"  client {i} stratified-composition check (n_i={n_i}), 5 simulated epochs:")
    comp_ok = True
    for epoch in range(5):
        drawn = list(iter(sampler))
        drawn_labels = Counter(_full_labels[j] for j in drawn)
        matches = drawn_labels == client_own_by_class[i]
        comp_ok &= matches
        print(f"    epoch {epoch}: composition matches original K32 exactly = {matches}")
    print(f"  {'PASS' if comp_ok else 'FAIL'} -- stratified composition preserved every epoch")
    all_pass &= comp_ok

print()
print(f"OVERALL: {'ALL PASS' if all_pass else 'FAIL -- see above'}")

print()
print("=" * 78)
print("Pre-flight RNG + short real 2-round run (target2x)")
print("=" * 78)
args.iters = 2
args.wk_iters = 1
args.adcol_epoch = 1
args.adcol_mu = 0.1
args.adcol_beta = 0.1
set_seed(args)
train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=4)
args.num_users = len(train_loader_list)
DSLR_CLIENTS = [i for i, d in enumerate(client_domains) if d == "dslr"]
client_own_indices = {i: list(train_loader_list[i].sampler.indices) for i in DSLR_CLIENTS}
client_own_by_class = {i: Counter(_full_labels[j] for j in client_own_indices[i]) for i in DSLR_CLIENTS}

pool_rng = np.random.RandomState(args.seed + 40000)
pool_by_class = {}
for i in DSLR_CLIENTS:
    pool_by_class[i] = {}
    for cls, n_local in client_own_by_class[i].items():
        n_domain = domain_counts[cls]
        target = min(2 * n_local, n_domain)
        own_this_class = [j for j in client_own_indices[i] if _full_labels[j] == cls]
        n_extra_needed = target - n_local
        extra = []
        if n_extra_needed > 0:
            peer_candidates = [j for j in domain_class_indices[cls] if j not in client_own_indices[i]]
            chosen = pool_rng.choice(peer_candidates, size=n_extra_needed, replace=False)
            extra = [int(x) for x in chosen]
        pool_by_class[i][cls] = own_this_class + extra

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

_py_before = random.getstate()
_np_before = np.random.get_state()
_torch_before = torch.get_rng_state().clone()

feature_loader_list = [None] * args.num_users
for i in DSLR_CLIENTS:
    original_loader = train_loader_list[i]
    feature_loader_list[i] = original_loader
    persistent_transform = PersistentRNGCompose(transform_office_strong, seed=args.seed + 30000 + i)
    strong_ds_i = data_utils.OfficeDataset(DATA_BASE_PATH, "dslr", transform=persistent_transform)
    g = torch.Generator()
    g.manual_seed(args.seed + 10000 + i)
    train_loader_list[i] = DataLoader(
        strong_ds_i, batch_size=args.batch,
        sampler=_StratifiedPoolSampler(pool_by_class[i], client_own_by_class[i], g),
        num_workers=args.number_workers, pin_memory=True,
    )

rng_ok = (random.getstate() == _py_before and np.array_equal(np.random.get_state()[1], _np_before[1])
          and torch.equal(torch.get_rng_state(), _torch_before))
print(f"pre-flight global RNG unchanged: {rng_ok}")

_, acc, names = ours(
    args, train_loader_list, test_loader_list, client_domains,
    feature_loader_list=feature_loader_list,
    preserve_feature_probe_state=True,
)
for d in names:
    print(f"  {d:<8}: {[round(v, 4) for v in acc[d]]}")
print("Smoke test finished.")

import sys
import os
sys.path.insert(0, os.getcwd())
import tempfile

import random
import torch
import numpy as np
from torchvision import transforms as tvt
from torch.utils.data import DataLoader, SubsetRandomSampler

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_PACS_multi_clients
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


def run_condition(use_aug, seed=0, iters=2):
    sys.argv = ["smoke_test_stronglocalaug_pacs"]
    args = args_parser()
    args.exp = 1
    args.mode = "ours"
    args.iters = iters
    args.lr = 0.01
    args.number_workers = 0
    args.dataset = "PACS"
    args.num_classes = 7
    args.size = 64
    args.batch = 32
    args.wk_iters = 2
    args.adcol_mu = 0.1
    args.adcol_beta = 0.1
    args.adcol_epoch = 1
    args.device = "cpu"
    args.domain_keyed_proto = True
    args.no_discriminator_fix = False
    args.adaptive_local_batch = False
    LAMBDA_G = 0.0607

    args.seed = seed
    set_seed(args)

    train_loader_list, client_domains, test_loader_list = prepare_data_PACS_multi_clients(args)
    args.num_users = len(train_loader_list)

    print(f"[condition use_aug={use_aug}] client_domains = {client_domains}")
    assert all(hasattr(l.sampler, "indices") for l in train_loader_list), \
        "prepare_data_PACS_multi_clients loaders do not expose .sampler.indices"

    DATA_BASE_PATH = "../data/PACS"
    AUG_DOMAINS = {"art_painting", "photo", "sketch"}
    feature_loader_list = [None] * args.num_users
    _aug_report = []

    if use_aug:
        transform_pacs_strong = tvt.Compose([
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

        for i, dom in enumerate(client_domains):
            if dom not in AUG_DOMAINS:
                continue
            original_loader = train_loader_list[i]
            feature_loader_list[i] = original_loader
            idx = original_loader.sampler.indices
            persistent_transform = PersistentRNGCompose(transform_pacs_strong, seed=args.seed + 30000 + i)
            strong_ds_i = data_utils.PACSDataset(DATA_BASE_PATH, dom, transform=persistent_transform)
            _img, _lab = strong_ds_i[int(idx[0])]
            assert _img.shape[0] == 3, f"unexpected channel count from PACSDataset: {_img.shape}"
            g = torch.Generator()
            g.manual_seed(args.seed + 10000 + i)
            train_loader_list[i] = DataLoader(
                strong_ds_i, batch_size=args.batch,
                sampler=SubsetRandomSampler(idx, generator=g),
                num_workers=0, pin_memory=False,
            )
            _aug_report.append((i, dom, len(idx)))

        assert random.getstate() == _py_before, "global python RNG mutated by loader construction"
        assert np.array_equal(np.random.get_state()[1], _np_before[1]), "global numpy RNG mutated"
        assert torch.equal(torch.get_rng_state(), _torch_before), "global torch RNG mutated"

        touched = {dom for (_, dom, _) in _aug_report}
        assert touched == AUG_DOMAINS, f"expected touched={AUG_DOMAINS}, got {touched}"
        untouched = [dom for dom in client_domains if dom not in AUG_DOMAINS]
        assert set(untouched) == {"cartoon"}, f"expected only cartoon untouched, got {set(untouched)}"
        assert all(feature_loader_list[i] is None for i, d in enumerate(client_domains) if d == "cartoon"), \
            "cartoon client(s) unexpectedly got a feature_loader entry"
        print(f"  touched clients: {_aug_report}")
        print(f"  untouched domains: {set(untouched)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "ckpt.pt")
        acc_log_path = os.path.join(tmpdir, "acc.csv")
        weights_dir = os.path.join(tmpdir, "weights/")
        train_loss, accuracy_list, datasets_name = ours(
            args, train_loader_list, test_loader_list, client_domains,
            checkpoint_path=checkpoint_path, checkpoint_every=5,
            acc_log_path=acc_log_path, weights_dir=weights_dir,
            lambda_G=LAMBDA_G,
            lambda_mixup=0.0, domain_balanced_gc=False,
            feature_loader_list=feature_loader_list,
            preserve_feature_probe_state=True,
        )
    print(f"  ran {iters} round(s) OK. final accs: "
          f"{ {d: round(accuracy_list[d][-1], 4) for d in datasets_name} }")
    return accuracy_list


print("=" * 78)
print("SMOKE TEST: run_d_stronglocalaug_pacs.py -- baseline condition")
run_condition(use_aug=False)
print("=" * 78)
print("SMOKE TEST: run_d_stronglocalaug_pacs.py -- strongaug condition")
run_condition(use_aug=True)
print("=" * 78)
print("ALL SMOKE TESTS PASSED")

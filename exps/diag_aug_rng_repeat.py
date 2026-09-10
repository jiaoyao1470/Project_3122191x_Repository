import sys
import os
sys.path.insert(0, os.getcwd())

import random
import torch
import numpy as np
from torchvision import transforms as tvt
from torch.utils.data import DataLoader, SubsetRandomSampler

from option import args_parser
from federated_main import set_seed
from util import prepare_data_office_multi_clients
from update import LocalUpdate
import data_utils


class PersistentRNGCompose:
    def __init__(self, transform, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self._python_state = random.getstate()
        self._numpy_state = np.random.get_state()
        self._torch_state = torch.get_rng_state().clone()
        self.transform = transform

    def __call__(self, img):
        caller_python = random.getstate()
        caller_numpy = np.random.get_state()
        caller_torch = torch.get_rng_state().clone()

        random.setstate(self._python_state)
        np.random.set_state(self._numpy_state)
        torch.set_rng_state(self._torch_state)

        out = self.transform(img)

        self._python_state = random.getstate()
        self._numpy_state = np.random.get_state()
        self._torch_state = torch.get_rng_state().clone()

        random.setstate(caller_python)
        np.random.set_state(caller_numpy)
        torch.set_rng_state(caller_torch)

        return out


sys.argv = ["diag_aug_rng_repeat"]
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
set_seed(args)

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=4)

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

client_idx = 6
idx = train_loader_list[client_idx].sampler.indices

persistent_transform = PersistentRNGCompose(transform_office_strong, seed=args.seed + 30000 + client_idx)
strong_ds = data_utils.OfficeDataset(DATA_BASE_PATH, "dslr", transform=persistent_transform)
g = torch.Generator()
g.manual_seed(args.seed + 10000 + client_idx)
loader = DataLoader(strong_ds, batch_size=args.batch, sampler=SubsetRandomSampler(idx, generator=g),
                     num_workers=0, pin_memory=True)


def simulate_one_round():
    _ = LocalUpdate(args=args)
    images, labels = next(iter(loader))
    return images, labels


print("Simulating round r0 (LocalUpdate reseed -> strong-aug DataLoader draw)...")
images_r0, labels_r0 = simulate_one_round()

print("Simulating round r1 (LocalUpdate reseed AGAIN -> strong-aug DataLoader draw)...")
images_r1, labels_r1 = simulate_one_round()

print("Simulating round r2 (LocalUpdate reseed AGAIN -> strong-aug DataLoader draw)...")
images_r2, labels_r2 = simulate_one_round()

identical_r0_r1 = torch.equal(images_r0, images_r1) and torch.equal(labels_r0, labels_r1)
identical_r0_r2 = torch.equal(images_r0, images_r2) and torch.equal(labels_r0, labels_r2)
max_diff_r0_r1 = (images_r0 - images_r1).abs().max().item()
max_diff_r0_r2 = (images_r0 - images_r2).abs().max().item()

print(f"\nr0 vs r1: byte-identical = {identical_r0_r1}  (max pixel diff = {max_diff_r0_r1:.6f})")
print(f"r0 vs r2: byte-identical = {identical_r0_r2}  (max pixel diff = {max_diff_r0_r2:.6f})")
print(f"labels_r0[:8] = {labels_r0[:8].tolist()}")
print(f"labels_r1[:8] = {labels_r1[:8].tolist()}")
print(f"labels_r2[:8] = {labels_r2[:8].tolist()}")
print("EXPECT: byte-identical=False, nonzero max diff, label order differing across rounds (sampler "
      "generator is independent too) -- if any of this still shows repetition, the fix didn't take.")

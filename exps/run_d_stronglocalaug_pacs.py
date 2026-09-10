import sys
import os
sys.path.insert(0, os.getcwd())

import random
import shutil
import numpy as np
import torch
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


if len(sys.argv) not in (2, 3) or sys.argv[1] not in ("baseline", "strongaug"):
    raise RuntimeError("Usage: python run_d_stronglocalaug_pacs.py <baseline|strongaug> [seed]")
_condition = sys.argv[1]
_use_aug = (_condition == "strongaug")
_SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
sys.argv = ["run_d_stronglocalaug_pacs"]
args = args_parser()

args.exp = 1
args.mode = "ours"
args.iters = 100
args.lr = 0.01
args.number_workers = 0
args.dataset = "PACS"
args.num_classes = 7
args.size = 64
args.batch = 32
args.wk_iters = 10
args.adcol_mu = 0.1
args.adcol_beta = 0.1
args.adcol_epoch = 3
args.device = args.device if torch.cuda.is_available() else "cpu"

args.domain_keyed_proto = True
args.no_discriminator_fix = False
args.adaptive_local_batch = False
LAMBDA_G = 0.0607

args.seed = _SEED
set_seed(args)

train_loader_list, client_domains, test_loader_list = prepare_data_PACS_multi_clients(args)
args.num_users = len(train_loader_list)

DATA_BASE_PATH = "../data/PACS"
AUG_DOMAINS = {"art_painting", "photo", "sketch"}
feature_loader_list = [None] * args.num_users
_aug_report = []
_aug_generators = {}

if _use_aug:
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
        g = torch.Generator()
        g.manual_seed(args.seed + 10000 + i)
        _aug_generators[i] = g
        train_loader_list[i] = DataLoader(
            strong_ds_i,
            batch_size=args.batch,
            sampler=SubsetRandomSampler(idx, generator=g),
            num_workers=args.number_workers,
            pin_memory=True,
        )
        _aug_report.append((i, dom, len(idx)))

    assert random.getstate() == _py_before, (
        "constructing the StrongLocalAug loaders mutated the global Python random state -- model "
        "initialization inside ours() would no longer match D_matched_pacs's seed init."
    )
    assert np.array_equal(np.random.get_state()[1], _np_before[1]), (
        "constructing the StrongLocalAug loaders mutated the global numpy RNG state."
    )
    assert torch.equal(torch.get_rng_state(), _torch_before), (
        "constructing the StrongLocalAug loaders mutated the global torch RNG state -- ours()'s "
        "global_model/local_models/discriminator/global_classifier init would be seeded wrong."
    )

    if _aug_generators:
        _gen_seeds = [g.initial_seed() for g in _aug_generators.values()]
        assert len(_gen_seeds) == len(set(_gen_seeds)), (
            f"augmented clients do not have distinct sampler generator seeds -- {_gen_seeds}."
        )

    if not _aug_report:
        raise RuntimeError(f"No clients found in AUG_DOMAINS={AUG_DOMAINS} -- client_domains={client_domains}")
    _touched_domains = {dom for (_, dom, _) in _aug_report}
    _missing_domains = AUG_DOMAINS - _touched_domains
    if _missing_domains:
        raise RuntimeError(f"Domains in AUG_DOMAINS never matched any client: {_missing_domains}")

print("=" * 78)
print(f"D-MATCHED vs D+STRONGLOCALAUG ON PACS -- condition={_condition}  (augmentation={_use_aug})  "
      f"seed={args.seed}")
print(f"domain_keyed_proto = {args.domain_keyed_proto}   "
      f"no_discriminator_fix = {args.no_discriminator_fix}   "
      f"adaptive_local_batch = {getattr(args, 'adaptive_local_batch', False)}   "
      f"lambda_G = {LAMBDA_G}   preserve_feature_probe_state = True")
print("-" * 78)
if _use_aug:
    print("augmented clients (train_loader=strong-aug view of the SAME fixed images, feature_loader=fixed original):")
    for (ci, dom, n_i) in _aug_report:
        print(f"  client {ci:>2} | domain={dom:<12} | num_samples={n_i:>4} | "
              f"train_sampler={type(train_loader_list[ci].sampler).__name__} | "
              f"feature_sampler={type(feature_loader_list[ci].sampler).__name__}")
    print("untouched clients (cartoon control, feature_loader_list entry = None -> falls back to their own "
          "unmodified train_loader_list[i]):")
    for i, dom in enumerate(client_domains):
        if dom in AUG_DOMAINS:
            continue
        print(f"  client {i:>2} | domain={dom:<12} | num_samples={len(train_loader_list[i].sampler.indices):>4}")
else:
    print("baseline condition -- no augmentation, all clients use the original unmodified loaders "
          "(feature_loader_list stays all-None).")
print("reference points:")
print("  ceiling (FedPall paper reproduction, one-domain-one-client, 3-seed): 60.07%")
print("  Multi-Client baseline (fragmented, no repair, 3-seed):               54.43%")
print("  discriminator-fix alone (seed0):                                     56.09%")
print("  C = discriminator-fix + domain-keyed-proto, Layer1 only (seed0):     55.39%")
print("  D (old, WITHOUT preserve_feature_probe_state, 3-seed):               55.62%   <- NOT a valid")
print("                                                                                    denominator, see docstring")
print("  D_matched_pacs (this run, 'baseline' condition): what we're finding out")
print("  D_matched_pacs + StrongLocalAug (this run, 'strongaug' condition): what we're finding out")
print("=" * 78)

RESUME_INPUT_DIR = None
RESUME = False

if _use_aug and (RESUME or RESUME_INPUT_DIR is not None):
    raise RuntimeError(
        "StrongAug exact resume is disabled: PersistentRNGCompose's private transform RNG state "
        "is not checkpointed by ours(). Run the formal strongaug condition uninterrupted from "
        "round 0 in a single session; do not set RESUME=True or RESUME_INPUT_DIR for this "
        "condition. (The baseline condition has no augmentation and is unaffected by this guard.)"
    )


def align_acc_csv_to_checkpoint(checkpoint_path, acc_log_path):
    import csv
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_round = int(ckpt["round"])

    if not os.path.exists(acc_log_path):
        raise RuntimeError(
            f"Resume checkpoint is at round {ckpt_round}, but acc.csv is missing: {acc_log_path}"
        )

    with open(acc_log_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    rows = [r for r in rows if int(r["round"]) <= ckpt_round]
    dedup = {}
    for r in rows:
        dedup[(int(r["round"]), r["domain"])] = r
    rows = sorted(dedup.values(), key=lambda r: (int(r["round"]), r["domain"]))

    with open(acc_log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Aligned acc.csv to checkpoint round {ckpt_round}; training resumes at round {ckpt_round + 1}.")


_tag = f"d_stronglocalaug_pacs_seed{args.seed}_{_condition}"

if os.path.isdir("/kaggle/working"):
    drive_base = "/kaggle/working/FedPall/limitation_2/d_stronglocalaug_pacs"
elif os.path.isdir("/content/drive/MyDrive"):
    drive_base = "/content/drive/MyDrive/FedPall/limitation_2/d_stronglocalaug_pacs"
else:
    raise RuntimeError(
        "Neither /kaggle/working nor /content/drive/MyDrive is available.\n"
        "On Colab: mount Drive first --\n"
        "  from google.colab import drive\n"
        "  drive.mount('/content/drive')\n"
    )
os.makedirs(drive_base, exist_ok=True)

checkpoint_path = f"{drive_base}/{_tag}_checkpoint.pt"
acc_log_path = f"{drive_base}/{_tag}_acc.csv"
weights_dir = f"{drive_base}/weights_{_tag}/"

if RESUME_INPUT_DIR is not None:
    _resume_ckpt = f"{RESUME_INPUT_DIR}/{_tag}_checkpoint.pt"
    _resume_acc = f"{RESUME_INPUT_DIR}/{_tag}_acc.csv"
    if not os.path.exists(_resume_ckpt):
        raise RuntimeError(f"RESUME_INPUT_DIR is set but checkpoint is missing: {_resume_ckpt}")
    if not os.path.exists(_resume_acc):
        raise RuntimeError(f"RESUME_INPUT_DIR is set but acc.csv is missing: {_resume_acc}")
    shutil.copy(_resume_ckpt, checkpoint_path)
    print(f"Restored checkpoint from RESUME_INPUT_DIR ({_resume_ckpt}).")
    shutil.copy(_resume_acc, acc_log_path)
    print(f"Restored acc.csv from RESUME_INPUT_DIR ({_resume_acc}).")
    _resume_weights = f"{RESUME_INPUT_DIR}/weights_{_tag}"
    if os.path.isdir(_resume_weights):
        shutil.copytree(_resume_weights, weights_dir, dirs_exist_ok=True)
        print("Restored weights dir from RESUME_INPUT_DIR.")
    else:
        print(f"WARNING: RESUME_INPUT_DIR is set but {_resume_weights} does not exist -- the "
              f"diagnostic best-round weight files before this resume point will be missing.")

if os.path.exists(checkpoint_path) and (RESUME or RESUME_INPUT_DIR is not None):
    align_acc_csv_to_checkpoint(checkpoint_path, acc_log_path)

if os.path.exists(checkpoint_path) and not RESUME and RESUME_INPUT_DIR is None:
    raise RuntimeError(
        f"Checkpoint already exists: {checkpoint_path}\n"
        "ours() would silently resume from it. If this is a deliberate continuation of the SAME\n"
        "configuration, set RESUME = True. If anything in this driver changed since that checkpoint\n"
        "was written, delete or move it instead -- resuming would mix two configurations in one run."
    )

print(f"checkpoint: {checkpoint_path}  (RESUME={RESUME}, RESUME_INPUT_DIR={RESUME_INPUT_DIR})")
train_loss, accuracy_list, datasets_name = ours(
    args, train_loader_list, test_loader_list, client_domains,
    checkpoint_path=checkpoint_path, checkpoint_every=5,
    acc_log_path=acc_log_path, weights_dir=weights_dir,
    lambda_G=LAMBDA_G,
    lambda_mixup=0.0, domain_balanced_gc=False,
    feature_loader_list=feature_loader_list,
    preserve_feature_probe_state=True,
)
print(f"Done. acc log at {acc_log_path}")
for domain in datasets_name:
    accs = accuracy_list[domain]
    print(f"  {domain:<12}: first={accs[0]:.4f}  final={accs[-1]:.4f}  "
          f"own-peak={max(accs):.4f} @r{accs.index(max(accs))}")

n_rounds = len(accuracy_list[datasets_name[0]])
avg_by_round = [
    sum(accuracy_list[d][r] for d in datasets_name) / len(datasets_name)
    for r in range(n_rounds)
]
xdom_best_round = max(range(n_rounds), key=lambda r: avg_by_round[r])
_late_idx = [r for r in range(n_rounds) if 90 <= r <= 99]
print("-" * 78)
print(f"X-dom-best: round={xdom_best_round}  four-domain avg={avg_by_round[xdom_best_round]:.4f}")
for domain in datasets_name:
    accs = accuracy_list[domain]
    _x = accs[xdom_best_round]
    _late = sum(accs[r] for r in _late_idx) / len(_late_idx) if _late_idx else None
    print(f"    {domain:<12} @X-dom-best round = {_x:.4f}   late-window(r90-99) = "
          f"{_late:.4f}" if _late is not None else f"    {domain:<12} @X-dom-best round = {_x:.4f}")
print("Run reconstruct_robust_standard.py on this run's acc.csv for the robust-standard comparison "
      "against the OTHER condition's own acc.csv (baseline vs strongaug) -- do NOT compare against the "
      "old, unmatched D=55.62% number.")
print("-" * 78)

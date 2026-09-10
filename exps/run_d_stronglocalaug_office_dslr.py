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


_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0

sys.argv = ["run_d_stronglocalaug_office_dslr"]
args = args_parser()

args.exp = 1
args.mode = "ours"
args.iters = 100
args.lr = 0.01
args.number_workers = 0
args.dataset = "office"
args.num_classes = 10
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
LAMBDA_G = 0.05

args.seed = _SEED
set_seed(args)

_M = 4
train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=_M)
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

_py_before = random.getstate()
_np_before = np.random.get_state()
_torch_before = torch.get_rng_state().clone()

AUG_DOMAIN = "dslr"
feature_loader_list = [None] * args.num_users
_aug_report = []
_aug_generators = {}
for i, dom in enumerate(client_domains):
    if dom != AUG_DOMAIN:
        continue
    original_loader = train_loader_list[i]
    feature_loader_list[i] = original_loader
    idx = original_loader.sampler.indices
    persistent_transform = PersistentRNGCompose(transform_office_strong, seed=args.seed + 30000 + i)
    strong_ds_i = data_utils.OfficeDataset(DATA_BASE_PATH, "dslr", transform=persistent_transform)
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
    _aug_report.append((i, len(idx)))

assert random.getstate() == _py_before, (
    "constructing the StrongLocalAug loaders mutated the global Python random state -- model "
    "initialization inside ours() would no longer match D_matched's seed0 init."
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
        f"dslr clients do not have distinct sampler generator seeds -- {_gen_seeds}."
    )

if not _aug_report:
    raise RuntimeError(f"No '{AUG_DOMAIN}' clients found -- client_domains={client_domains}")

print("=" * 78)
print(f"D + STRONG LOCAL AUGMENTATION -- dslr train_loader gets richer views, feature_loader fixed, "
      f"M={_M}, seed={args.seed}")
print(f"domain_keyed_proto = {args.domain_keyed_proto}   "
      f"no_discriminator_fix = {args.no_discriminator_fix}   "
      f"adaptive_local_batch = {getattr(args, 'adaptive_local_batch', False)}   "
      f"lambda_G = {LAMBDA_G}")
print("-" * 78)
print("dslr clients (train_loader=strong-aug view of the SAME fixed images, feature_loader=fixed original):")
for (ci, n_i) in _aug_report:
    print(f"  client {ci:>2} | num_samples={n_i:>4} | "
          f"train_sampler={type(train_loader_list[ci].sampler).__name__} | "
          f"feature_sampler={type(feature_loader_list[ci].sampler).__name__}")
print("untouched clients (caltech/amazon/webcam, feature_loader_list entry = None -> falls back "
      "to their own unmodified train_loader_list[i]):")
for i, dom in enumerate(client_domains):
    if dom == AUG_DOMAIN:
        continue
    print(f"  client {i:>2} | domain={dom:<8} | num_samples={len(train_loader_list[i].sampler.indices):>4}")
print("reference points:")
print("  D(old, unmatched)        : X-dom-best=0.3387  late-window=0.2645   DIAGNOSTIC REFERENCE ONLY --")
print("                             NOT a valid R_aug denominator (predates preserve_feature_probe_state=True,")
print("                             see run_d_meandispersion_office_dslr.py's own header on why). Formal R_aug")
print("                             needs a D_matched run (preserve_feature_probe_state=True, no augmentation)")
print("                             -- not yet re-run for this specific comparison; use whatever the")
print("                             already-locked D_matched numbers are once available.")
print("  A (local expand, oracle) : X-dom-best=0.7500          late-window=0.7685   (upper bound for R_aug)")
print("  Resampling (matched)     : X-dom-best=0.7823          late-window=0.7290")
print("=" * 78)

RESUME_INPUT_DIR = None
RESUME = False


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


_tag = f"d_stronglocalaug_M4_seed{args.seed}"

if os.path.isdir("/kaggle/working"):
    drive_base = "/kaggle/working/FedPall/limitation_2/d_stronglocalaug_dslr"
elif os.path.isdir("/content/drive/MyDrive"):
    drive_base = "/content/drive/MyDrive/FedPall/limitation_2/d_stronglocalaug_dslr"
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
    print(f"  {domain:<8}: first={accs[0]:.4f}  final={accs[-1]:.4f}  "
          f"own-peak={max(accs):.4f} @r{accs.index(max(accs))}")

n_rounds = len(accuracy_list[datasets_name[0]])
avg_by_round = [
    sum(accuracy_list[d][r] for d in datasets_name) / len(datasets_name)
    for r in range(n_rounds)
]
xdom_best_round = max(range(n_rounds), key=lambda r: avg_by_round[r])
_x = accuracy_list["dslr"][xdom_best_round]
_late_idx = [r for r in range(n_rounds) if 90 <= r <= 99]
_late = sum(accuracy_list["dslr"][r] for r in _late_idx) / len(_late_idx) if _late_idx else None
print("-" * 78)
print(f"X-dom-best (PRIMARY): round={xdom_best_round}  four-domain avg={avg_by_round[xdom_best_round]:.4f}")
for domain in datasets_name:
    print(f"    {domain:<8} @that round = {accuracy_list[domain][xdom_best_round]:.4f}")
print(f"  -> dslr X-dom-best [StrongLocalAug] = {_x:.4f}   (A oracle=0.7500; D(old,unmatched)=0.3387 is a "
      f"diagnostic reference only, NOT a valid R_aug denominator -- see header)")
if _late is not None:
    print(f"  -> dslr late-window [StrongLocalAug] = {_late:.4f}   (A oracle=0.7685; D(old,unmatched)=0.2645 "
          f"is a diagnostic reference only, NOT a valid R_aug denominator -- see header)")
print("  R_aug NOT auto-computed here -- needs D_matched (preserve_feature_probe_state=True, no "
      "augmentation), not the old unmatched D. Compute once that reference is available:")
print("    R_aug        = (X_aug - X_D_matched) / (0.7500 - X_D_matched)")
print("    R_aug_late   = (L_aug - L_D_matched) / (0.7685 - L_D_matched)")
print("-" * 78)

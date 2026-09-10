import sys
import os
sys.path.insert(0, os.getcwd())

import random
import shutil
import csv
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
        for c, m in self.m_by_class.items():
            assert len(self.pool_by_class[c]) >= m, (
                f"class {c}: pool has only {len(self.pool_by_class[c])} but needs to draw {m} "
                "per epoch -- accessible pool is smaller than the original per-class exposure."
            )

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


if len(sys.argv) != 2 or sys.argv[1] not in ("target2x", "maxeligible"):
    raise RuntimeError("Usage: python run_d_ksupport_oracle_office_dslr.py <target2x|maxeligible>")
_DOSE = sys.argv[1]
sys.argv = ["run_d_ksupport_oracle_office_dslr"]
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

args.seed = 0
set_seed(args)

_M = 4
train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=_M)
args.num_users = len(train_loader_list)

DATA_BASE_PATH = "../data/office_caltech_10"
CLASS_NAMES = ["back_pack", "bike", "calculator", "headphones", "keyboard",
               "laptop_computer", "monitor", "mouse", "mug", "projector"]

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
DSLR_CLIENTS = [i for i, d in enumerate(client_domains) if d == AUG_DOMAIN]

_full_labels = train_loader_list[DSLR_CLIENTS[0]].dataset.labels
domain_class_indices = {}
for idx_, lab in enumerate(_full_labels):
    domain_class_indices.setdefault(lab, []).append(idx_)
domain_counts = {c: len(v) for c, v in domain_class_indices.items()}

client_own_indices = {i: list(train_loader_list[i].sampler.indices) for i in DSLR_CLIENTS}
client_own_by_class = {}
for i in DSLR_CLIENTS:
    labels_i = [_full_labels[j] for j in client_own_indices[i]]
    client_own_by_class[i] = Counter(labels_i)

pool_rng = np.random.RandomState(args.seed + 40000)
pool_by_class = {}
achievable_K = {}
for i in DSLR_CLIENTS:
    pool_by_class[i] = {}
    for cls, n_local in client_own_by_class[i].items():
        n_domain = domain_counts[cls]
        target = min(2 * n_local, n_domain) if _DOSE == "target2x" else n_domain
        own_this_class = [j for j in client_own_indices[i] if _full_labels[j] == cls]
        assert len(own_this_class) == n_local, (
            f"client {i} class {cls}: Counter said {n_local} but re-derived list has "
            f"{len(own_this_class)} -- indexing bug."
        )
        n_extra_needed = target - n_local
        extra = []
        if n_extra_needed > 0:
            peer_candidates = [j for j in domain_class_indices[cls] if j not in client_own_indices[i]]
            assert len(peer_candidates) >= n_extra_needed, (
                f"client {i} class {cls}: need {n_extra_needed} extra but only "
                f"{len(peer_candidates)} peer candidates available -- logic error."
            )
            chosen = pool_rng.choice(peer_candidates, size=n_extra_needed, replace=False)
            extra = [int(x) for x in chosen]
        class_pool = own_this_class + extra
        assert len(class_pool) == target, (
            f"client {i} class {cls}: pool size {len(class_pool)} != target {target}."
        )
        assert len(class_pool) == len(set(class_pool)), (
            f"client {i} class {cls}: duplicate indices in pool -- overlap between own+extra."
        )
        pool_by_class[i][cls] = class_pool
    achievable_K[i] = sum(len(v) for v in pool_by_class[i].values())

for i in DSLR_CLIENTS:
    pool_classes = set(pool_by_class[i].keys())
    own_classes = set(client_own_by_class[i].keys())
    assert pool_classes == own_classes, (
        f"client {i}: accessible pool has classes {pool_classes} but own K32 partition has "
        f"{own_classes} -- C_i^dose != C_i^32, constraint violated."
    )

_py_before = random.getstate()
_np_before = np.random.get_state()
_torch_before = torch.get_rng_state().clone()

feature_loader_list = [None] * args.num_users
_report = []
for i in DSLR_CLIENTS:
    original_loader = train_loader_list[i]
    feature_loader_list[i] = original_loader
    n_i = len(original_loader.sampler.indices)
    assert n_i == sum(client_own_by_class[i].values()), (
        f"client {i}: per-epoch exposure budget {n_i} != sum of original per-class counts "
        f"{sum(client_own_by_class[i].values())} -- budget-preservation assumption broken."
    )
    persistent_transform = PersistentRNGCompose(transform_office_strong, seed=args.seed + 30000 + i)
    strong_ds_i = data_utils.OfficeDataset(DATA_BASE_PATH, "dslr", transform=persistent_transform)
    g = torch.Generator()
    g.manual_seed(args.seed + 10000 + i)
    train_loader_list[i] = DataLoader(
        strong_ds_i,
        batch_size=args.batch,
        sampler=_StratifiedPoolSampler(pool_by_class[i], client_own_by_class[i], g),
        num_workers=args.number_workers,
        pin_memory=True,
    )
    _report.append((i, n_i, achievable_K[i], sorted(client_own_by_class[i].keys())))

assert random.getstate() == _py_before, (
    "constructing the K-support-oracle loaders mutated the global Python random state -- model "
    "initialization inside ours() would no longer match D_matched's seed0 init."
)
assert np.array_equal(np.random.get_state()[1], _np_before[1]), (
    "constructing the K-support-oracle loaders mutated the global numpy RNG state."
)
assert torch.equal(torch.get_rng_state(), _torch_before), (
    "constructing the K-support-oracle loaders mutated the global torch RNG state -- ours()'s "
    "global_model/local_models/discriminator/global_classifier init would be seeded wrong."
)

print("=" * 100)
print(f"ACCESSIBLE SUPPORT-BREADTH DOSE ORACLE -- dose={_DOSE} (diagnostic oracle, NOT deployable), "
      f"same StrongAug recipe as StrongLocalAug, M={_M}, seed={args.seed}")
print(f"domain_keyed_proto = {args.domain_keyed_proto}   no_discriminator_fix = {args.no_discriminator_fix}   "
      f"lambda_G = {LAMBDA_G}")
print("-" * 100)
print("dslr clients (train_loader = StrongAug over an EXPANDED, class-constrained accessible pool, "
      "feature_loader = fixed K32 baseline-aug, unchanged from D):")
for (ci, n_i, k_i, classes) in _report:
    own_n = len(client_own_indices[ci])
    class_names_owned = [CLASS_NAMES[c] for c in classes]
    print(f"  client {ci:>2} | per-epoch exposure={n_i:>3} | own K32={own_n:>3} | "
          f"accessible pool K_i={k_i:>4} | owns {len(classes)}/10 classes: {class_names_owned}")
print("reference points:")
print("  Original(=StrongLocalAug, ALREADY DONE): X-dom-best=0.4919  own-peak=0.5242(r35)  late-window=0.4903")
print("  D(old, unmatched)         : X-dom-best=0.3387  late-window=0.2645   diagnostic reference only")
print("  A (full resampling oracle, NO StrongAug): X-dom-best=0.7500  late-window=0.7685")
print("  Resampling (matched, NO StrongAug)      : X-dom-best=0.7823  late-window=0.7290")
print("=" * 100)

RESUME_INPUT_DIR = None
RESUME = False


def align_acc_csv_to_checkpoint(checkpoint_path, acc_log_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_round = int(ckpt["round"])
    if not os.path.exists(acc_log_path):
        raise RuntimeError(f"Resume checkpoint is at round {ckpt_round}, but acc.csv is missing: {acc_log_path}")
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


_tag = f"d_ksupport_{_DOSE}_M4_seed0"

if os.path.isdir("/kaggle/working"):
    drive_base = "/kaggle/working/FedPall/limitation_2/d_ksupport_oracle_dslr"
elif os.path.isdir("/content/drive/MyDrive"):
    drive_base = "/content/drive/MyDrive/FedPall/limitation_2/d_ksupport_oracle_dslr"
else:
    raise RuntimeError(
        "Neither /kaggle/working nor /content/drive/MyDrive is available.\n"
        "On Colab: mount Drive first --\n  from google.colab import drive\n  drive.mount('/content/drive')\n"
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
    print(f"  {domain:<8}: first={accs[0]:.4f}  final={accs[-1]:.4f}  own-peak={max(accs):.4f} @r{accs.index(max(accs))}")

n_rounds = len(accuracy_list[datasets_name[0]])
avg_by_round = [sum(accuracy_list[d][r] for d in datasets_name) / len(datasets_name) for r in range(n_rounds)]
xdom_best_round = max(range(n_rounds), key=lambda r: avg_by_round[r])
_x = accuracy_list["dslr"][xdom_best_round]
_late_idx = [r for r in range(n_rounds) if 90 <= r <= 99]
_late = sum(accuracy_list["dslr"][r] for r in _late_idx) / len(_late_idx) if _late_idx else None
print("-" * 100)
print(f"X-dom-best (PRIMARY): round={xdom_best_round}  four-domain avg={avg_by_round[xdom_best_round]:.4f}")
for domain in datasets_name:
    print(f"    {domain:<8} @that round = {accuracy_list[domain][xdom_best_round]:.4f}")
print(f"  -> dslr X-dom-best [{_DOSE}+StrongAug] = {_x:.4f}   (Original+StrongAug=0.4919, A oracle=0.7500)")
if _late is not None:
    print(f"  -> dslr late-window [{_DOSE}+StrongAug] = {_late:.4f}   (Original+StrongAug=0.4903, A oracle=0.7685)")
print("-" * 100)

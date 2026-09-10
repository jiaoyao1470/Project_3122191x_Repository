import sys
import os
sys.path.insert(0, os.getcwd())

import shutil
import torch

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


sys.argv = ["run_d_mixup_office_dslr"]
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

MIXUP_DOMAIN = "dslr"
LAMBDA_INPUT_MIXUP = 1.0
INPUT_MIXUP_ALPHA = 0.2

_mixup_clients = [i for i, dom in enumerate(client_domains) if dom == MIXUP_DOMAIN]
if not _mixup_clients:
    raise RuntimeError(f"No '{MIXUP_DOMAIN}' clients found -- client_domains={client_domains}")

print("=" * 78)
print(f"D + GENUINE INPUT-SPACE MIXUP -- dslr's CE term becomes a convex blend with cross-sample "
      f"MixUp CE, M={_M}, seed={args.seed}")
print(f"domain_keyed_proto = {args.domain_keyed_proto}   "
      f"no_discriminator_fix = {args.no_discriminator_fix}   "
      f"adaptive_local_batch = {getattr(args, 'adaptive_local_batch', False)}   "
      f"lambda_G = {LAMBDA_G}")
print(f"input_mixup_domains = {{'{MIXUP_DOMAIN}'}}   lambda_input_mixup = {LAMBDA_INPUT_MIXUP}   "
      f"input_mixup_alpha = {INPUT_MIXUP_ALPHA}")
print(f"old lambda_mixup (missing-class, detached, classifier-only) = 0.0 (unchanged, unused here)")
print("-" * 78)
print("dslr clients (MixUp on, loader/partition unchanged from D_matched):")
for i in _mixup_clients:
    print(f"  client {i:>2} | num_samples={len(train_loader_list[i].sampler.indices):>4}")
print("untouched clients (caltech/amazon/webcam, lambda_input_mixup_client=0.0 by domain-gating):")
for i, dom in enumerate(client_domains):
    if dom == MIXUP_DOMAIN:
        continue
    print(f"  client {i:>2} | domain={dom:<8} | num_samples={len(train_loader_list[i].sampler.indices):>4}")
print("reference points:")
print("  D_matched (preserve_feature_probe_state=True, no augmentation/MixUp/MD/FeatureBank/mixing):")
print("    X-dom-best=0.3790   late-window=0.2895")
print("  StrongLocalAug (candidate #1, confirmed positive, single variable):")
print("    X-dom-best=0.4919   late-window=0.4903   R_aug~=30.4%/41.9%")
print("  A (local expand, oracle, upper bound)    : X-dom-best=0.7500   late-window=0.7685")
print("  Resampling (matched, oracle, upper bound) : X-dom-best=0.7823   late-window=0.7290")
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


_tag = "d_mixup_M4_seed0"

if os.path.isdir("/kaggle/working"):
    drive_base = "/kaggle/working/FedPall/limitation_2/d_mixup_dslr"
elif os.path.isdir("/content/drive/MyDrive"):
    drive_base = "/content/drive/MyDrive/FedPall/limitation_2/d_mixup_dslr"
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
    input_mixup_domains={MIXUP_DOMAIN}, lambda_input_mixup=LAMBDA_INPUT_MIXUP, input_mixup_alpha=INPUT_MIXUP_ALPHA,
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
print(f"  -> dslr X-dom-best [MixUp] = {_x:.4f}")
if _late is not None:
    print(f"  -> dslr late-window [MixUp] = {_late:.4f}")
print("R_mixup vs D_matched (X_D_matched=0.3790, L_D_matched=0.2895, A=0.7500/0.7685):")
print(f"    R_mixup      = (X_mixup - 0.3790) / (0.7500 - 0.3790) = {(_x - 0.3790) / (0.7500 - 0.3790):.4f}")
if _late is not None:
    print(f"    R_mixup_late = (L_mixup - 0.2895) / (0.7685 - 0.2895) = {(_late - 0.2895) / (0.7685 - 0.2895):.4f}")
print("Standalone repair comparison -- StrongLocalAug and MixUp are two")
print("PARALLEL, independent single-variable arms off D_matched, NOT a chain. MixUp is NOT applied")
print("on top of StrongLocalAug in this run -- 'MixUp - StrongLocalAug' below is a head-to-head")
print("difference between two separate standalone experiments, not an incremental gain on top of")
print("StrongLocalAug's own result. Do not read it as 'S->MixUp'.):")
print(f"    D_matched -> StrongLocalAug (X-dom-best) : delta = {0.4919 - 0.3790:.4f}  "
      f"({0.3790:.4f} -> {0.4919:.4f})")
print(f"    D_matched -> MixUp          (X-dom-best) : delta = {_x - 0.3790:.4f}  "
      f"({0.3790:.4f} -> {_x:.4f})")
print(f"    MixUp - StrongLocalAug      (X-dom-best) : {_x - 0.4919:.4f}   [head-to-head only, NOT incremental]")
if _late is not None:
    print(f"    D_matched -> StrongLocalAug (late-window): delta = {0.4903 - 0.2895:.4f}  "
          f"({0.2895:.4f} -> {0.4903:.4f})")
    print(f"    D_matched -> MixUp          (late-window): delta = {_late - 0.2895:.4f}  "
          f"({0.2895:.4f} -> {_late:.4f})")
    print(f"    MixUp - StrongLocalAug      (late-window): {_late - 0.4903:.4f}   [head-to-head only, NOT incremental]")
print("-" * 78)

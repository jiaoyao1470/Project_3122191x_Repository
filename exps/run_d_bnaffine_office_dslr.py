import sys
import os
sys.path.insert(0, os.getcwd())

import shutil
import torch

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


sys.argv = ["run_d_bnaffine_office_dslr"]
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

BN_AFFINE_SYNC_DOMAINS = {"dslr"}

print("=" * 78)
print(f"D + BN-AFFINE SYNC -- lightweight representation coordination oracle, M={_M}, seed={args.seed}")
print(f"domain_keyed_proto = {args.domain_keyed_proto}   "
      f"no_discriminator_fix = {args.no_discriminator_fix}   "
      f"adaptive_local_batch = {getattr(args, 'adaptive_local_batch', False)}   "
      f"lambda_G = {LAMBDA_G}")
print(f"bn_affine_sync_domains = {BN_AFFINE_SYNC_DOMAINS}   lambda_FB = 0.0 (FeatureBank OFF)")
print(f"client_domains = {client_domains}")
print("-" * 78)
print("reference to beat (M4_D_seed0, same seed/config, no BN sync): "
      "X-dom-best=0.3387 (r48)  own-peak=0.3790 (r20)  late-window=0.2645 (r90-99)")
print("ceiling for R_D (D+Resampling, different base condition -- an upper reference, not a target): "
      "X-dom-best=0.7823  late-window=0.7290")
print("R_D = (X_b - 0.3387)/0.4436     R_D_late = (L_b - 0.2645)/0.4645     "
      "thresholds: >=0.7 strong, 0.3-0.7 partial, <0.3 weak")
print("=" * 78)

RESUME_INPUT_DIR = None
RESUME = False

_tag = "d_bnaffine_M4_seed0"

if os.path.isdir("/kaggle/working"):
    drive_base = "/kaggle/working/FedPall/limitation_2/d_bnaffine_dslr"
elif os.path.isdir("/content/drive/MyDrive"):
    drive_base = "/content/drive/MyDrive/FedPall/limitation_2/d_bnaffine_dslr"
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
        print("Restored weights dir from RESUME_INPUT_DIR -- preserves the true all-round best "
              "weights even if no round after the resume point beats best_acc.")
    else:
        print(f"WARNING: RESUME_INPUT_DIR is set but {_resume_weights} does not exist -- the "
              f"diagnostic best-round weight files before this resume point will be missing (does "
              f"not affect the primary X-dom-best/late-window metrics).")

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
    lambda_FB=0.0,
    bn_affine_sync_domains=BN_AFFINE_SYNC_DOMAINS,
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
_x_b = accuracy_list["dslr"][xdom_best_round]
print("-" * 78)
print(f"X-dom-best (PRIMARY): round={xdom_best_round}  four-domain avg={avg_by_round[xdom_best_round]:.4f}")
for domain in datasets_name:
    print(f"    {domain:<8} @that round = {accuracy_list[domain][xdom_best_round]:.4f}")
print(f"  -> dslr X-dom-best = {_x_b:.4f}   (reference M4_D_seed0 = 0.3387)")
print(f"  -> dslr own-peak    = {max(accuracy_list['dslr']):.4f}   "
      f"(reference M4_D_seed0 = 0.3790, secondary metric)")
print(f"  -> delta vs D-control (X-dom-best) = {_x_b - 0.3387:+.4f}   "
      f"R_D = {(_x_b - 0.3387)/0.4436:.4f}")
print("     (R_D_late still to be computed from acc.csv rounds 90-99: (L_b - 0.2645)/0.4645)")
print("-" * 78)

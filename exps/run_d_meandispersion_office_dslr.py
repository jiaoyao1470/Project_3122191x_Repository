import sys
import os
sys.path.insert(0, os.getcwd())

import shutil
import torch

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


if len(sys.argv) not in (2, 3) or sys.argv[1] not in ("baseline", "meandispersion"):
    raise RuntimeError(
        "Usage: python run_d_meandispersion_office_dslr.py <baseline|meandispersion> [seed]"
    )
_condition = sys.argv[1]
_use_md = (_condition == "meandispersion")
_SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
sys.argv = ["run_d_meandispersion_office_dslr"]
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
LAMBDA_MD = 4.9297

args.seed = _SEED
set_seed(args)

_M = 4
train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=_M)
args.num_users = len(train_loader_list)

print("=" * 78)
print(f"D-MATCHED vs D+MEAN-DISPERSION -- condition={_condition}  (use_mean_dispersion={_use_md})  "
      f"M={_M}, seed={args.seed}")
print(f"domain_keyed_proto = {args.domain_keyed_proto}   "
      f"no_discriminator_fix = {args.no_discriminator_fix}   "
      f"adaptive_local_batch = {getattr(args, 'adaptive_local_batch', False)}   "
      f"lambda_G = {LAMBDA_G}   preserve_feature_probe_state = True")
print("-" * 78)
print("dslr clients (train_loader=fixed original, feature_loader=SAME fixed original -- no resampling):")
for i, dom in enumerate(client_domains):
    if dom != "dslr":
        continue
    print(f"  client {i:>2} | num_samples={len(train_loader_list[i].sampler.indices):>4} | "
          f"sampler={type(train_loader_list[i].sampler).__name__}")
print("reference points (A/B already locked, this run answers the repair question):")
print("  D(old, unmatched)      : X-dom-best=0.3387          late-window=0.2645")
print("  A (local expand)       : X-dom-best=0.7500 (pos=0.9272)  late-window=0.7685 (pos=1.0851)")
print("  B (server/upload expand): X-dom-best=0.4194 (pos=0.1818) late-window=0.3476 (pos=0.1789)")
print("  Resampling (matched)   : X-dom-best=0.7823          late-window=0.7290")
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


if _condition == "baseline":
    _tag = f"d_meandispersion_M4_seed{args.seed}_{_condition}_v2calib"
else:
    _tag = f"d_meandispersion_M4_seed{args.seed}_{_condition}_v3calib_rho05"

if os.path.isdir("/kaggle/working"):
    drive_base = "/kaggle/working/FedPall/limitation_2/d_meandispersion_dslr"
elif os.path.isdir("/content/drive/MyDrive"):
    drive_base = "/content/drive/MyDrive/FedPall/limitation_2/d_meandispersion_dslr"
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
    feature_loader_list=None,
    preserve_feature_probe_state=True,
    use_mean_dispersion=_use_md,
    lambda_md=LAMBDA_MD,
    mean_dispersion_shrinkage_tau=10.0,
    mean_dispersion_var_floor=1e-4,
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
print(f"  -> dslr X-dom-best [{_condition}] = {_x:.4f}")
if _late is not None:
    print(f"  -> dslr late-window [{_condition}] = {_late:.4f}")
print("     (run BOTH conditions, then: R_MD = (X_meandispersion - X_baseline) / (0.7500 - X_baseline), "
      "R_MD_late = (L_meandispersion - L_baseline) / (0.7685 - L_baseline))")
print("-" * 78)

import sys
import os
sys.path.insert(0, os.getcwd())

import torch
from torch.utils.data import DataLoader, RandomSampler

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


sys.argv = ["run_information_oracle_office_dslr"]
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

args.no_discriminator_fix = True
LAMBDA_G = 0.0

args.seed = 0
set_seed(args)

_M = 4
train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=_M)
args.num_users = len(train_loader_list)

ORACLE_DOMAIN = "dslr"
_oracle_report = []
_oracle_generators = {}
for i, dom in enumerate(client_domains):
    if dom != ORACLE_DOMAIN:
        continue
    ds = train_loader_list[i].dataset
    n_i = len(train_loader_list[i].sampler.indices)
    g = torch.Generator()
    g.manual_seed(args.seed + 10000 + i)
    _oracle_generators[i] = g
    train_loader_list[i] = DataLoader(
        ds,
        batch_size=args.batch,
        sampler=RandomSampler(ds, replacement=False, num_samples=n_i, generator=g),
        num_workers=args.number_workers,
        pin_memory=True,
    )
    _oracle_report.append((i, len(ds), n_i, args.batch, len(train_loader_list[i])))

if not _oracle_report:
    raise RuntimeError(f"No '{ORACLE_DOMAIN}' clients found -- client_domains={client_domains}")

print("=" * 78)
print(f"DSLR RESAMPLING INFORMATION ORACLE -- M={_M}, baseline arm, seed={args.seed}")
print(f"adaptive_local_batch = {getattr(args, 'adaptive_local_batch', False)}   "
      f"no_discriminator_fix = {args.no_discriminator_fix}   lambda_G = {LAMBDA_G}")
print("-" * 78)
print("resampled clients (static reads only, no draws -- see RNG NOTE above):")
for (ci, ds_size, n_i, bs, n_batches) in _oracle_report:
    print(f"  client {ci:>2} | domain={ORACLE_DOMAIN} | pool={ds_size} | num_samples={n_i} | "
          f"batch={bs} | batches/epoch={n_batches} | replacement=False | "
          f"sampler={type(train_loader_list[ci].sampler).__name__} | "
          f"generator_seed={_oracle_generators[ci].initial_seed()}")
_gen_seeds = [_oracle_generators[ci].initial_seed() for ci, *_ in _oracle_report]
assert len(_gen_seeds) == len(set(_gen_seeds)), (
    f"dslr clients do not have distinct generator seeds -- {_gen_seeds}. This would silently "
    "recreate the exact collision this fix exists to remove."
)
print("untouched clients:")
for i, dom in enumerate(client_domains):
    if dom == ORACLE_DOMAIN:
        continue
    print(f"  client {i:>2} | domain={dom:<8} | num_samples={len(train_loader_list[i].sampler.indices):>4} | "
          f"batch={train_loader_list[i].batch_size} | batches/epoch={len(train_loader_list[i])} | "
          f"sampler={type(train_loader_list[i].sampler).__name__}")
print("reference to beat (M4_baseline, same seed/config): peak 39.52%, late-window 37.23%")
print("=" * 78)

RESUME_INPUT_DIR = None

RESUME = False

if RESUME or RESUME_INPUT_DIR is not None:
    print("WARNING: resuming this driver restarts the 4 dslr clients' resampling generators from their "
          "own round-0 draws (generator state isn't checkpointed) -- see RESUME CAVEAT comment above. "
          "The run is still valid, just not a single unbroken 100-round stream for those 4 clients.")

_tag = "oracle_resample_M4_baseline_seed0"

if os.path.isdir("/kaggle/working"):
    drive_base = "/kaggle/working/FedPall/limitation_2/information_oracle_dslr"
    os.makedirs(drive_base, exist_ok=True)
    if RESUME_INPUT_DIR is not None:
        import shutil
        _resume_ckpt = f"{RESUME_INPUT_DIR}/{_tag}_checkpoint.pt"
        if os.path.exists(_resume_ckpt):
            shutil.copy(_resume_ckpt, f"{drive_base}/{_tag}_checkpoint.pt")
            print(f"Copied checkpoint from RESUME_INPUT_DIR ({_resume_ckpt}) into working dir for resume.")
        else:
            print(f"WARNING: RESUME_INPUT_DIR is set but {_resume_ckpt} does not exist -- starting fresh.")
elif os.path.isdir("/content/drive/MyDrive"):
    drive_base = "/content/drive/MyDrive/FedPall/limitation_2/information_oracle_dslr"
else:
    raise RuntimeError(
        "Neither /kaggle/working nor /content/drive/MyDrive is available.\n"
        "On Colab: mount Drive first --\n"
        "  from google.colab import drive\n"
        "  drive.mount('/content/drive')\n"
    )

checkpoint_path = f"{drive_base}/{_tag}_checkpoint.pt"
acc_log_path = f"{drive_base}/{_tag}_acc.csv"
weights_dir = f"{drive_base}/weights_{_tag}/"

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
print("-" * 78)
print(f"X-dom-best (PRIMARY): round={xdom_best_round}  four-domain avg={avg_by_round[xdom_best_round]:.4f}")
for domain in datasets_name:
    print(f"    {domain:<8} @that round = {accuracy_list[domain][xdom_best_round]:.4f}")
print(f"  -> dslr X-dom-best = {accuracy_list['dslr'][xdom_best_round]:.4f}   "
      f"(reference M4_baseline = 0.3871)")
print(f"  -> dslr own-peak    = {max(accuracy_list['dslr']):.4f}   "
      f"(reference M4_baseline = 0.3952, secondary metric)")
print("-" * 78)

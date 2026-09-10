import sys
import os
sys.path.insert(0, os.getcwd())

import json
import numpy as np

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


if len(sys.argv) < 2:
    raise SystemExit("usage: python run_mixup_broadened_office.py <mixup_support_threshold> [seed]")
_threshold_arg = int(sys.argv[1])
_seed_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 0
sys.argv = ["run_mixup_broadened_office"]
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
args.domain_keyed_proto = True
args.device = args.device if __import__("torch").cuda.is_available() else "cpu"

LAMBDA_G = 0.05
LAMBDA_MIXUP = 0.1
args.mixup_support_threshold = _threshold_arg

RESUME = False

RESUME_INPUT_DIR = None

if os.path.isdir("/kaggle/working"):
    drive_base = f"/kaggle/working/FedPall/limitation_2/mixup_broadened_office_thr{_threshold_arg}"
elif os.path.isdir("/content/drive/MyDrive"):
    drive_base = f"/content/drive/MyDrive/FedPall/limitation_2/mixup_broadened_office_thr{_threshold_arg}"
else:
    raise RuntimeError(
        "Neither /kaggle/working nor /content/drive/MyDrive is available.\n"
        "On Colab: mount Drive first --\n"
        "  from google.colab import drive\n"
        "  drive.mount('/content/drive')\n"
    )

seed = _seed_arg
args.seed = seed
set_seed(args)

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args)
args.num_users = len(train_loader_list)

trigger_by_domain = {}
firing_clients = []
for client_idx, domain in enumerate(client_domains):
    idx = train_loader_list[client_idx].sampler.indices
    labs = np.asarray(train_loader_list[client_idx].dataset.labels)[idx]
    class_counts = np.bincount(labs, minlength=args.num_classes)
    n_trigger = int((class_counts <= _threshold_arg).sum())
    trigger_by_domain[domain] = trigger_by_domain.get(domain, 0) + n_trigger
    if n_trigger >= 2:
        firing_clients.append((client_idx, domain, n_trigger))

print(f"\n[MixUp trigger coverage at mixup_support_threshold={_threshold_arg}]")
for d in sorted(trigger_by_domain, key=trigger_by_domain.get, reverse=True):
    print(f"  {d:10s} {trigger_by_domain[d]:3d} triggering (client, class) cells")
print(f"  clients that will ACTUALLY fire the MixUp term (needs >= 2 triggering classes): "
      f"{[(c, d, n) for c, d, n in firing_clients]}")
if not firing_clients:
    raise RuntimeError(
        f"mixup_support_threshold={_threshold_arg} triggers on no client with >= 2 classes -- the "
        f"MixUp term would be a silent no-op for this entire run. Raise the threshold."
    )
print()

checkpoint_path = f"{drive_base}/seed{seed}_checkpoint.pt"
acc_log_path = f"{drive_base}/seed{seed}_acc.csv"
weights_dir = f"{drive_base}/weights_seed{seed}/"
config_path = f"{drive_base}/seed{seed}_config.json"
this_config = {
    "mixup_support_threshold": _threshold_arg,
    "lambda_mixup": LAMBDA_MIXUP,
    "lambda_G": LAMBDA_G,
}

if RESUME_INPUT_DIR is not None:
    if not os.path.isdir(RESUME_INPUT_DIR):
        raise RuntimeError(f"RESUME_INPUT_DIR does not exist: {RESUME_INPUT_DIR}")
    src_ckpt = f"{RESUME_INPUT_DIR}/seed{seed}_checkpoint.pt"
    if not os.path.exists(src_ckpt):
        raise RuntimeError(
            f"No checkpoint at {src_ckpt} -- check RESUME_INPUT_DIR points at the directory that "
            f"directly contains seed{seed}_checkpoint.pt, and that the interrupted session was "
            f"actually saved (Save Version) before it ended."
        )
    os.makedirs(drive_base, exist_ok=True)
    import shutil
    for fname in [f"seed{seed}_checkpoint.pt", f"seed{seed}_config.json", f"seed{seed}_acc.csv"]:
        src = f"{RESUME_INPUT_DIR}/{fname}"
        if os.path.exists(src):
            shutil.copy(src, f"{drive_base}/{fname}")
            print(f"staged for resume: {src} -> {drive_base}/{fname}")
    if not RESUME:
        raise RuntimeError(
            "RESUME_INPUT_DIR is set but RESUME is still False -- set RESUME = True to actually "
            "continue the staged run (the config sidecar is verified either way, so a changed "
            "threshold/lambda still cannot slip through)."
        )

if os.path.exists(checkpoint_path):
    if not RESUME:
        raise RuntimeError(
            f"Checkpoint already exists: {checkpoint_path}\n"
            "This run is meant to start from round 0. Set RESUME=True only if nothing about the "
            "config changed since it was written (e.g. resuming after a disconnect); otherwise "
            "delete/move the old checkpoint."
        )
    if not os.path.exists(config_path):
        raise RuntimeError(f"Checkpoint exists but config sidecar missing at {config_path}.")
    with open(config_path) as f:
        recorded = json.load(f)
    if recorded != this_config:
        raise RuntimeError(
            f"Config changed since this checkpoint was written -- resuming would mix two configs "
            f"in one run. recorded={recorded} current={this_config}"
        )
    print(f"config verified unchanged since checkpoint was written: {recorded}")
else:
    os.makedirs(drive_base, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(this_config, f)
    print(f"recorded config for this fresh run -> {config_path}")

print(f"Running D+MixUp-broadened(threshold={_threshold_arg}, lambda_mixup={LAMBDA_MIXUP}) "
      f"seed={seed} (checkpoint: {checkpoint_path})")
train_loss, accuracy_list, datasets_name = ours(
    args, train_loader_list, test_loader_list, client_domains,
    checkpoint_path=checkpoint_path, checkpoint_every=5,
    acc_log_path=acc_log_path, weights_dir=weights_dir,
    lambda_G=LAMBDA_G, lambda_mixup=LAMBDA_MIXUP,
)
print(f"seed={seed} done. acc log at {acc_log_path}")
for domain in datasets_name:
    print(f"  {domain}: final acc={accuracy_list[domain][-1]:.4f}, first acc={accuracy_list[domain][0]:.4f}")

print("\nCompare (robust standard AND late-window) against D+MixUp(thr=0) seed0, the single-variable "
      "control -- NOT against D:")
print("  D+MixUp(thr=0) seed0 robust:      55.90%  (caltech 42.22, amazon 65.89, webcam 77.59, dslr 37.90)")
print("  D+MixUp(thr=0) seed0 late-window: 52.67%  (caltech 40.96, amazon 63.31, webcam 75.10, dslr 31.31)")

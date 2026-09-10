import sys
import os
sys.path.insert(0, os.getcwd())

import json

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


_seed_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 0
sys.argv = ["run_domain_balanced_gc_office"]
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
DOMAIN_BALANCED_GC = True

RESUME = False

RESUME_INPUT_DIR = None

if os.path.isdir("/kaggle/working"):
    drive_base = "/kaggle/working/FedPall/limitation_2/domain_balanced_gc_office"
elif os.path.isdir("/content/drive/MyDrive"):
    drive_base = "/content/drive/MyDrive/FedPall/limitation_2/domain_balanced_gc_office"
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

client_counts = [len(train_loader_list[i].sampler.indices) for i in range(args.num_users)]
domain_counts = {}
for ci, d in enumerate(client_domains):
    domain_counts[d] = domain_counts.get(d, 0) + client_counts[ci]
total = sum(domain_counts.values())
print("\n[domain-balanced global classifier] influence share on the shared head:")
print(f"  {'domain':10s}{'samples':>9s}{'default':>12s}{'balanced':>12s}")
for d in sorted(domain_counts, key=domain_counts.get, reverse=True):
    print(f"  {d:10s}{domain_counts[d]:>9d}{domain_counts[d] / total:>11.1%}{1 / len(domain_counts):>12.1%}")
print()

checkpoint_path = f"{drive_base}/seed{seed}_checkpoint.pt"
acc_log_path = f"{drive_base}/seed{seed}_acc.csv"
weights_dir = f"{drive_base}/weights_seed{seed}/"
config_path = f"{drive_base}/seed{seed}_config.json"
this_config = {"domain_balanced_gc": DOMAIN_BALANCED_GC, "lambda_G": LAMBDA_G}

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
            "continue the staged run."
        )

if os.path.exists(checkpoint_path):
    if not RESUME:
        raise RuntimeError(
            f"Checkpoint already exists: {checkpoint_path}\n"
            "This run is meant to start from round 0. Set RESUME=True only if nothing about the "
            "config changed since it was written; otherwise delete/move the old checkpoint."
        )
    if not os.path.exists(config_path):
        raise RuntimeError(f"Checkpoint exists but config sidecar missing at {config_path}.")
    with open(config_path) as f:
        recorded = json.load(f)
    if recorded != this_config:
        raise RuntimeError(
            f"Config changed since this checkpoint was written -- resuming would mix two configs. "
            f"recorded={recorded} current={this_config}"
        )
    print(f"config verified unchanged since checkpoint was written: {recorded}")
else:
    os.makedirs(drive_base, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(this_config, f)
    print(f"recorded config for this fresh run -> {config_path}")

print(f"Running D-BGC(domain_balanced_gc={DOMAIN_BALANCED_GC}) seed={seed} "
      f"(checkpoint: {checkpoint_path})")
train_loss, accuracy_list, datasets_name = ours(
    args, train_loader_list, test_loader_list, client_domains,
    checkpoint_path=checkpoint_path, checkpoint_every=5,
    acc_log_path=acc_log_path, weights_dir=weights_dir,
    lambda_G=LAMBDA_G, domain_balanced_gc=DOMAIN_BALANCED_GC,
)
print(f"seed={seed} done. acc log at {acc_log_path}")
for domain in datasets_name:
    print(f"  {domain}: final acc={accuracy_list[domain][-1]:.4f}, first acc={accuracy_list[domain][0]:.4f}")

print("\nCompare (robust standard AND late-window) against D seed0 -- domain_balanced_gc is the "
      "single variable relative to D:")
print("  D seed0 robust:      55.64%  (caltech 39.70, amazon 63.80, webcam 82.76, dslr 36.29)")
print("  D seed0 late-window: 50.69%  (caltech 39.10, amazon 61.74, webcam 76.45, dslr 25.48)")

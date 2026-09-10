import sys
import os
sys.path.insert(0, os.getcwd())

import json
import math

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


_seed_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 0
sys.argv = ["run_adaptive_batch_office"]
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
args.adaptive_local_batch = True

RESUME = False
RESUME_INPUT_DIR = None

if os.path.isdir("/kaggle/working"):
    drive_base = "/kaggle/working/FedPall/limitation_2/adaptive_batch_office"
elif os.path.isdir("/content/drive/MyDrive"):
    drive_base = "/content/drive/MyDrive/FedPall/limitation_2/adaptive_batch_office"
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

print("\n[fragmentation-aware local batch] B_i = min(32, ceil(n_i/2)):")
print(f"  {'client':<7}{'domain':<9}{'n_i':>5}{'B_i':>6}{'batches':>9}{'steps/round':>13}")
n_changed = 0
for i, d in enumerate(client_domains):
    n_i = len(train_loader_list[i].sampler.indices)
    b_i = train_loader_list[i].batch_size
    n_batches = len(train_loader_list[i])
    flag = ""
    if b_i != args.batch:
        flag = "  <-- CHANGED"
        n_changed += 1
    print(f"  {i:<7}{d:<9}{n_i:>5}{b_i:>6}{n_batches:>9}{n_batches * args.wk_iters:>13}{flag}")
if n_changed == 0:
    raise RuntimeError(
        "adaptive_local_batch=True but no client's batch size actually changed -- the flag is not "
        "reaching prepare_data_office_multi_clients, or no client is smaller than args.batch."
    )
for i in range(args.num_users):
    if len(train_loader_list[i]) < 2:
        raise RuntimeError(
            f"client {i} still has {len(train_loader_list[i])} batch(es) per epoch after applying "
            f"the rule -- the >=2-minibatch guarantee is violated."
        )
    n_i = len(train_loader_list[i].sampler.indices)
    b_i = train_loader_list[i].batch_size
    last_batch = n_i % b_i or b_i
    if last_batch < 2:
        raise RuntimeError(
            f"client {i} (n={n_i}, B={b_i}) would end its epoch on a batch of {last_batch} sample(s); "
            f"BatchNorm requires >1 sample per batch in train mode."
        )
print(f"  ({n_changed} client(s) changed; every client now has >= 2 minibatches per local epoch)\n")

checkpoint_path = f"{drive_base}/seed{seed}_checkpoint.pt"
acc_log_path = f"{drive_base}/seed{seed}_acc.csv"
weights_dir = f"{drive_base}/weights_seed{seed}/"
config_path = f"{drive_base}/seed{seed}_config.json"
this_config = {"adaptive_local_batch": True, "batch": args.batch, "lambda_G": LAMBDA_G}

if RESUME_INPUT_DIR is not None:
    if not os.path.isdir(RESUME_INPUT_DIR):
        raise RuntimeError(f"RESUME_INPUT_DIR does not exist: {RESUME_INPUT_DIR}")
    src_ckpt = f"{RESUME_INPUT_DIR}/seed{seed}_checkpoint.pt"
    if not os.path.exists(src_ckpt):
        raise RuntimeError(
            f"No checkpoint at {src_ckpt} -- check RESUME_INPUT_DIR points at the directory that "
            f"directly contains seed{seed}_checkpoint.pt, and that the interrupted session was "
            f"saved (Save Version) before it ended."
        )
    os.makedirs(drive_base, exist_ok=True)
    import shutil
    for fname in [f"seed{seed}_checkpoint.pt", f"seed{seed}_config.json", f"seed{seed}_acc.csv"]:
        src = f"{RESUME_INPUT_DIR}/{fname}"
        if os.path.exists(src):
            shutil.copy(src, f"{drive_base}/{fname}")
            print(f"staged for resume: {src} -> {drive_base}/{fname}")
    if not RESUME:
        raise RuntimeError("RESUME_INPUT_DIR is set but RESUME is still False.")

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

print(f"Running D-ALB(adaptive_local_batch=True) seed={seed} (checkpoint: {checkpoint_path})")
train_loss, accuracy_list, datasets_name = ours(
    args, train_loader_list, test_loader_list, client_domains,
    checkpoint_path=checkpoint_path, checkpoint_every=5,
    acc_log_path=acc_log_path, weights_dir=weights_dir,
    lambda_G=LAMBDA_G,
)
print(f"seed={seed} done. acc log at {acc_log_path}")
for domain in datasets_name:
    print(f"  {domain}: final acc={accuracy_list[domain][-1]:.4f}, first acc={accuracy_list[domain][0]:.4f}")

print("\nCompare (robust standard AND late-window) against D seed0 -- adaptive_local_batch is the "
      "single variable relative to D:")
print("  D seed0 robust:      55.64%  (caltech 39.70, amazon 63.80, webcam 82.76, dslr 36.29)")
print("  D seed0 late-window: 50.69%  (caltech 39.10, amazon 61.74, webcam 76.45, dslr 25.48)")
print("Prediction under test: dslr improves noticeably; the other three barely move.")

import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients

sys.argv = ["run_baseline_relog_office"]
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
args.device = args.device if __import__("torch").cuda.is_available() else "cpu"

if not os.path.isdir("/content/drive/MyDrive"):
    raise RuntimeError(
        "Google Drive is not mounted at /content/drive/MyDrive. "
        "Run this in a Colab cell BEFORE this script:\n"
        "  from google.colab import drive\n"
        "  drive.mount('/content/drive')\n"
    )

drive_base = "/content/drive/MyDrive/FedPall/limitation_2/baseline_relog"
seeds = [0, 1, 2]

for seed in seeds:
    args.seed = seed
    set_seed(args)

    train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args)
    args.num_users = len(train_loader_list)

    checkpoint_path = f"{drive_base}/seed{seed}_checkpoint.pt"
    acc_log_path = f"{drive_base}/seed{seed}_acc.csv"
    weights_dir = f"{drive_base}/weights_seed{seed}/"

    print(f"Running setting=multi_client seed={seed} (checkpoint: {checkpoint_path})")
    train_loss, accuracy_list, datasets_name = ours(
        args, train_loader_list, test_loader_list, client_domains,
        checkpoint_path=checkpoint_path, checkpoint_every=5,
        acc_log_path=acc_log_path, weights_dir=weights_dir,
    )
    print(f"seed={seed} done. acc log at {acc_log_path}")

print("All seeds done.")

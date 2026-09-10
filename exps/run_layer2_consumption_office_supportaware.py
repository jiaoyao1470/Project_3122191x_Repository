import sys
import os
sys.path.insert(0, os.getcwd())

import json
from collections import defaultdict

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients

_seed_arg = sys.argv[1] if len(sys.argv) > 1 else None
sys.argv = ["run_layer2_consumption_office_supportaware"]
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

LAMBDA_G_BASE = 0.05

if os.path.isdir("/kaggle/working"):
    drive_base = "/kaggle/working/FedPall/limitation_2/layer2_consumption_office_supportaware"
elif os.path.isdir("/content/drive/MyDrive"):
    drive_base = "/content/drive/MyDrive/FedPall/limitation_2/layer2_consumption_office_supportaware"
else:
    raise RuntimeError(
        "Neither /kaggle/working nor /content/drive/MyDrive is available.\n"
        "On Colab: mount Drive first --\n"
        "  from google.colab import drive\n"
        "  drive.mount('/content/drive')\n"
        "On Kaggle: /kaggle/working should always exist -- check you're running inside a Kaggle "
        "notebook environment."
    )

RESUME = False

seeds_to_run = [0]
seeds = [int(_seed_arg)] if _seed_arg is not None else seeds_to_run

for seed in seeds:
    args.seed = seed
    set_seed(args)

    train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args)
    args.num_users = len(train_loader_list)

    N_d = defaultdict(int)
    M_d = defaultdict(int)
    for client_idx, domain in enumerate(client_domains):
        M_d[domain] += 1
        N_d[domain] += len(train_loader_list[client_idx].sampler.indices)
    S_d = {d: N_d[d] / M_d[d] for d in N_d}
    S_max = max(S_d.values())
    lambda_G_by_domain = {d: LAMBDA_G_BASE * S_d[d] / S_max for d in S_d}

    print("\n[Support-aware lambda_G]")
    for d in sorted(lambda_G_by_domain, key=lambda_G_by_domain.get, reverse=True):
        print(f"  {d:10s} N={N_d[d]:4d}  M={M_d[d]}  support={S_d[d]:7.2f}  "
              f"lambda_G={lambda_G_by_domain[d]:.5f}")
    print()

    checkpoint_path = f"{drive_base}/seed{seed}_checkpoint.pt"
    acc_log_path = f"{drive_base}/seed{seed}_acc.csv"
    weights_dir = f"{drive_base}/weights_seed{seed}/"
    g_log_path = f"{drive_base}/seed{seed}_g_log.csv"
    provenance_path = f"{drive_base}/seed{seed}_lambda_G_provenance.json"

    if os.path.exists(checkpoint_path):
        if not RESUME:
            raise RuntimeError(
                f"Checkpoint already exists: {checkpoint_path}\n"
                "D-SA is intended to start from round 0 for a clean single-variable comparison "
                "against D. Set RESUME=True only if you've confirmed nothing about this script's "
                "lambda_G formula or config changed since this checkpoint was written (e.g. "
                "resuming after a disconnect with unmodified code) -- delete/move the checkpoint "
                "otherwise."
            )
        if not os.path.exists(provenance_path):
            raise RuntimeError(
                f"Checkpoint exists at {checkpoint_path} but its lambda_G provenance sidecar is "
                f"missing at {provenance_path} -- cannot confirm this checkpoint was produced "
                f"under the CURRENT lambda_G_by_domain values. Do not resume from it blindly."
            )
        with open(provenance_path) as f:
            recorded_lambda_G = json.load(f)
        mismatch = {
            d: (recorded_lambda_G.get(d), lambda_G_by_domain[d])
            for d in lambda_G_by_domain
            if d not in recorded_lambda_G or abs(recorded_lambda_G[d] - lambda_G_by_domain[d]) > 1e-9
        }
        if mismatch:
            raise RuntimeError(
                f"lambda_G_by_domain has changed since checkpoint {checkpoint_path} was written -- "
                f"resuming would silently mix two different configs in one run. Mismatches "
                f"(domain: (recorded, current)): {mismatch}"
            )
        print(f"lambda_G provenance verified unchanged since checkpoint was written: {recorded_lambda_G}")
    else:
        os.makedirs(os.path.dirname(provenance_path), exist_ok=True)
        with open(provenance_path, "w") as f:
            json.dump(lambda_G_by_domain, f)
        print(f"recorded lambda_G provenance for this fresh run -> {provenance_path}")

    print(f"Running setting=D-SA(D + support-aware lambda_G,d) seed={seed} "
          f"(checkpoint: {checkpoint_path})")
    train_loss, accuracy_list, datasets_name = ours(
        args, train_loader_list, test_loader_list, client_domains,
        checkpoint_path=checkpoint_path, checkpoint_every=5,
        acc_log_path=acc_log_path, weights_dir=weights_dir,
        g_log_path=g_log_path,
        lambda_G=lambda_G_by_domain,
    )
    print(f"seed={seed} done. acc log at {acc_log_path}")
    for domain in datasets_name:
        print(f"  {domain}: final acc={accuracy_list[domain][-1]:.4f}, first acc={accuracy_list[domain][0]:.4f}")

print("All seeds done.")
print("\nMATCHED-SEED comparison point (do not compare against D's 3-seed mean yet):")
print("  D seed0 (robust standard): 55.64% (caltech=39.70%, amazon=63.80%, webcam=82.76%, dslr=36.29%)")

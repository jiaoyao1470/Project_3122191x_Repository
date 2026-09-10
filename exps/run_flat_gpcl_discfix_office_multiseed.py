import sys
import os
sys.path.insert(0, os.getcwd())

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


sys.argv = ["run_flat_gpcl_discfix_office_multiseed"]

if not os.path.isdir("/content/drive/MyDrive"):
    raise RuntimeError(
        "Google Drive is not mounted at /content/drive/MyDrive. "
        "Run this in a Colab cell BEFORE this script:\n"
        "  from google.colab import drive\n"
        "  drive.mount('/content/drive')\n"
        "Without this, checkpoints/logs would silently write to Colab's ephemeral "
        "/content filesystem and be lost on disconnect."
    )

drive_base = "/content/drive/MyDrive/FedPall/limitation_2/flat_gpcl_discfix_office"
os.makedirs(drive_base, exist_ok=True)
seeds = [0, 1, 2]

for seed in seeds:
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
    args.use_flat_gpcl = True
    args.device = args.device if __import__("torch").cuda.is_available() else "cpu"
    args.seed = seed
    set_seed(args)

    train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args)
    args.num_users = len(train_loader_list)

    checkpoint_path = f"{drive_base}/seed{seed}_checkpoint.pt"
    acc_log_path = f"{drive_base}/seed{seed}_acc.csv"
    weights_dir = f"{drive_base}/weights_seed{seed}/"

    print(f"\n{'='*20} seed={seed} (target iters={args.iters}) {'='*20}")
    train_loss, accuracy_list, datasets_name = ours(
        args, train_loader_list, test_loader_list, client_domains,
        checkpoint_path=checkpoint_path, checkpoint_every=5,
        acc_log_path=acc_log_path, weights_dir=weights_dir,
    )
    print(f"=== seed={seed} finished without crashing ===")
    for domain in datasets_name:
        print(f"  {domain}: final acc={accuracy_list[domain][-1]:.4f}, first acc={accuracy_list[domain][0]:.4f}")

print(f"\n{'='*20} all {len(seeds)} seeds complete {'='*20}")
for seed in seeds:
    print(f"seed={seed}: weights under {drive_base}/weights_seed{seed}/, "
          f"acc log at {drive_base}/seed{seed}_acc.csv")
print("\nreference points (robust standard, 3-seed unless noted):")
print("  baseline(neither fix):                          52.11%")
print("  A = flat GPCL alone (no disc-fix):                54.17%")
print("  discriminator-fix alone:                          54.81%")
print("  D (two-layer + disc-fix, full method):            55.13%")
print("checkpoint is on Drive -- safe across Colab session restarts, just re-run this same cell.")

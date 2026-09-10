import sys
import os
sys.path.insert(0, os.getcwd())

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


_seed_arg = sys.argv[1] if len(sys.argv) > 1 else None
sys.argv = ["run_mixup_missing_class_office"]
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

if os.path.isdir("/kaggle/working"):
    drive_base = "/kaggle/working/FedPall/limitation_2/mixup_missing_class_office"
elif os.path.isdir("/content/drive/MyDrive"):
    drive_base = "/content/drive/MyDrive/FedPall/limitation_2/mixup_missing_class_office"
else:
    raise RuntimeError(
        "Neither /kaggle/working nor /content/drive/MyDrive is available.\n"
        "On Colab: mount Drive first --\n"
        "  from google.colab import drive\n"
        "  drive.mount('/content/drive')\n"
    )

seeds_to_run = [0]
seeds = [int(_seed_arg)] if _seed_arg is not None else seeds_to_run

for seed in seeds:
    args.seed = seed
    set_seed(args)

    train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args)
    args.num_users = len(train_loader_list)

    checkpoint_path = f"{drive_base}/seed{seed}_checkpoint.pt"
    acc_log_path = f"{drive_base}/seed{seed}_acc.csv"
    weights_dir = f"{drive_base}/weights_seed{seed}/"

    print(f"Running setting=D+MixUp(lambda_mixup={LAMBDA_MIXUP}) seed={seed} (checkpoint: {checkpoint_path})")
    train_loss, accuracy_list, datasets_name = ours(
        args, train_loader_list, test_loader_list, client_domains,
        checkpoint_path=checkpoint_path, checkpoint_every=5,
        acc_log_path=acc_log_path, weights_dir=weights_dir,
        lambda_G=LAMBDA_G, lambda_mixup=LAMBDA_MIXUP,
    )
    print(f"seed={seed} done. acc log at {acc_log_path}")
    for domain in datasets_name:
        print(f"  {domain}: final acc={accuracy_list[domain][-1]:.4f}, first acc={accuracy_list[domain][0]:.4f}")

print("Done.")
print("\nreference points (robust standard):")
print("  baseline(3-seed):                  52.11%")
print("  discriminator-fix alone(3-seed):    54.81%")
print("  D (no MixUp, 3-seed):               55.13%")

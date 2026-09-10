import sys
import os
import shutil
sys.path.insert(0, os.getcwd())

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_PACS_multi_clients


sys.argv = ["calib_run_for_lambda_G_pacs"]
args = args_parser()

args.exp = 1
args.mode = "ours"
args.lr = 0.01
args.number_workers = 0
args.dataset = "PACS"
args.num_classes = 7
args.size = 64
args.batch = 32
args.wk_iters = 10
args.adcol_mu = 0.1
args.adcol_beta = 0.1
args.adcol_epoch = 3
args.domain_keyed_proto = True
args.device = args.device if __import__("torch").cuda.is_available() else "cpu"
args.seed = 0
set_seed(args)

train_loader_list, client_domains, test_loader_list = prepare_data_PACS_multi_clients(args)
args.num_users = len(train_loader_list)

if os.path.isdir("/kaggle/working"):
    base_dir = "/kaggle/working/result/calib_lambda_G_run_pacs"
elif os.path.isdir("/content/drive/MyDrive"):
    base_dir = "/content/drive/MyDrive/FedPall/limitation_2/calib_lambda_G_run_pacs"
else:
    base_dir = "calib_lambda_G_run_pacs"
os.makedirs(base_dir, exist_ok=True)

checkpoint_path = f"{base_dir}/seed0_checkpoint.pt"
acc_log_path = f"{base_dir}/seed0_acc.csv"
weights_dir = f"{base_dir}/weights_seed0/"
g_log_path = f"{base_dir}/seed0_g_log.csv"

probe_rounds = [5, 10, 15, 20]
for target_iters in probe_rounds:
    args.iters = target_iters
    print(f"\n=== stage: training up to round {target_iters - 1} (checkpoint_every=5) ===")
    train_loss, accuracy_list, datasets_name = ours(
        args, train_loader_list, test_loader_list, client_domains,
        checkpoint_path=checkpoint_path, checkpoint_every=5,
        acc_log_path=acc_log_path, weights_dir=weights_dir,
        g_log_path=g_log_path,
    )
    snapshot_path = f"{base_dir}/seed0_checkpoint_round{target_iters - 1}.pt"
    shutil.copy(checkpoint_path, snapshot_path)
    print(f"snapshot saved: {snapshot_path}")
    for d in datasets_name:
        print(f"  {d}: acc={accuracy_list[d][-1]:.4f}")

print("\n=== all 4 probe-point snapshots ready ===")
print(f"snapshots under: {base_dir}")
print("download these 4 checkpoint_round*.pt files for Phase 2's gradient probe.")
if base_dir.startswith("/kaggle/"):
    print("\nIMPORTANT (Kaggle): click 'Save Version' when done so /kaggle/working becomes durable Output.")

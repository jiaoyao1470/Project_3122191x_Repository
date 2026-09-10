import sys
import os
sys.path.insert(0, os.getcwd())
import shutil

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


CONDITIONS = {
    "0": (True,  False, 0.0),
    "1": (False, False, 0.0),
    "2": (True,  True,  0.0),
    "4": (False, True,  0.0),
    "6": (True,  True,  0.05),
    "7": (False, True,  0.05),
}
STAGE_ROUNDS = [10, 20, 30]

if len(sys.argv) < 2 or sys.argv[1] not in CONDITIONS:
    raise SystemExit(f"Usage: run_component_ablation_office_m2.py <{'|'.join(CONDITIONS.keys())}> [final_iters]")
_cond = sys.argv[1]
_final_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
no_disc_fix, domain_keyed, lambda_g = CONDITIONS[_cond]
sys.argv = ["run_component_ablation_office_m2"]
args = args_parser()

args.exp = 1
args.mode = "ours"
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

if no_disc_fix:
    args.no_discriminator_fix = True
if domain_keyed:
    args.domain_keyed_proto = True

args.seed = 0
set_seed(args)

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=2)
args.num_users = len(train_loader_list)

RESUME_INPUT_DIR = None

if os.path.isdir("/kaggle/working"):
    drive_base = f"/kaggle/working/FedPall/limitation_2/component_ablation_m2/cond{_cond}"
    os.makedirs(drive_base, exist_ok=True)
    if RESUME_INPUT_DIR is not None:
        _resume_ckpt = f"{RESUME_INPUT_DIR}/cond{_cond}_seed0_checkpoint.pt"
        if os.path.exists(_resume_ckpt):
            shutil.copy(_resume_ckpt, f"{drive_base}/cond{_cond}_seed0_checkpoint.pt")
            print(f"Copied checkpoint from RESUME_INPUT_DIR for resume.")
        else:
            print(f"WARNING: RESUME_INPUT_DIR set but {_resume_ckpt} not found -- starting fresh.")
elif os.path.isdir("/content/drive/MyDrive"):
    drive_base = f"/content/drive/MyDrive/FedPall/limitation_2/component_ablation_m2/cond{_cond}"
else:
    raise RuntimeError(
        "Neither /kaggle/working nor /content/drive/MyDrive is available.\n"
        "On Colab: mount Drive first --\n"
        "  from google.colab import drive\n"
        "  drive.mount('/content/drive')\n"
    )

checkpoint_path = f"{drive_base}/cond{_cond}_seed0_checkpoint.pt"
acc_log_path = f"{drive_base}/cond{_cond}_seed0_acc.csv"
weights_dir = f"{drive_base}/weights_seed0/"

print(f"Component ablation condition {_cond}: no_discriminator_fix={no_disc_fix}, "
      f"domain_keyed_proto={domain_keyed}, lambda_G={lambda_g}")

for stage_round in STAGE_ROUNDS:
    if stage_round > _final_iters:
        break
    args.iters = stage_round
    print(f"\n=== Running to round {stage_round} ===")
    train_loss, accuracy_list, datasets_name = ours(
        args, train_loader_list, test_loader_list, client_domains,
        checkpoint_path=checkpoint_path, checkpoint_every=5,
        acc_log_path=acc_log_path, weights_dir=weights_dir,
        lambda_G=lambda_g,
    )
    stage_snapshot = f"{drive_base}/cond{_cond}_seed0_round{stage_round}_checkpoint.pt"
    shutil.copy(checkpoint_path, stage_snapshot)
    print(f"Stage complete (round {stage_round}). Snapshot saved: {stage_snapshot}")
    for domain in datasets_name:
        print(f"  {domain}: acc={accuracy_list[domain][-1]:.4f}")

print(f"\nDone. Final acc log at {acc_log_path}")
print(f"Round-specific snapshots: cond{_cond}_seed0_round{{10,20,30}}_checkpoint.pt in {drive_base}")

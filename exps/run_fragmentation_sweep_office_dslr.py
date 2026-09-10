import sys
import os
sys.path.insert(0, os.getcwd())

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


if len(sys.argv) < 3 or sys.argv[1] not in ("1", "2", "4") or sys.argv[2] not in ("baseline", "D"):
    raise SystemExit("Usage: run_fragmentation_sweep_office_dslr.py <1|2|4> <baseline|D> [seed]")
_M = int(sys.argv[1])
_condition = sys.argv[2]
_SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
sys.argv = ["run_fragmentation_sweep_office_dslr"]
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

if _condition == "baseline":
    args.no_discriminator_fix = True
    LAMBDA_G = 0.0
else:
    args.domain_keyed_proto = True
    LAMBDA_G = 0.05

args.seed = _SEED
set_seed(args)

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=_M)
args.num_users = len(train_loader_list)

RESUME_INPUT_DIR = None

if os.path.isdir("/kaggle/working"):
    drive_base = f"/kaggle/working/FedPall/limitation_2/fragmentation_sweep_dslr/M{_M}_{_condition}"
    os.makedirs(drive_base, exist_ok=True)
    if RESUME_INPUT_DIR is not None:
        import shutil
        _resume_ckpt = f"{RESUME_INPUT_DIR}/M{_M}_{_condition}_seed{args.seed}_checkpoint.pt"
        if os.path.exists(_resume_ckpt):
            shutil.copy(_resume_ckpt, f"{drive_base}/M{_M}_{_condition}_seed{args.seed}_checkpoint.pt")
            print(f"Copied checkpoint from RESUME_INPUT_DIR ({_resume_ckpt}) into working dir for resume.")
        else:
            print(f"WARNING: RESUME_INPUT_DIR is set but {_resume_ckpt} does not exist -- starting fresh.")
elif os.path.isdir("/content/drive/MyDrive"):
    drive_base = f"/content/drive/MyDrive/FedPall/limitation_2/fragmentation_sweep_dslr/M{_M}_{_condition}"
else:
    raise RuntimeError(
        "Neither /kaggle/working nor /content/drive/MyDrive is available.\n"
        "On Colab: mount Drive first --\n"
        "  from google.colab import drive\n"
        "  drive.mount('/content/drive')\n"
    )

checkpoint_path = f"{drive_base}/M{_M}_{_condition}_seed{args.seed}_checkpoint.pt"
acc_log_path = f"{drive_base}/M{_M}_{_condition}_seed{args.seed}_acc.csv"
weights_dir = f"{drive_base}/weights_seed{args.seed}/"

print(f"Running fragmentation sweep: M(dslr)={_M}, condition={_condition}, lambda_G={LAMBDA_G} "
      f"(checkpoint: {checkpoint_path})")
train_loss, accuracy_list, datasets_name = ours(
    args, train_loader_list, test_loader_list, client_domains,
    checkpoint_path=checkpoint_path, checkpoint_every=5,
    acc_log_path=acc_log_path, weights_dir=weights_dir,
        lambda_G=LAMBDA_G,
        preserve_feature_probe_state=(_condition == "D"),
)
print(f"Done. acc log at {acc_log_path}")
for domain in datasets_name:
    print(f"  {domain}: final acc={accuracy_list[domain][-1]:.4f}, first acc={accuracy_list[domain][0]:.4f}")

import sys
import os
sys.path.insert(0, os.getcwd())
import shutil

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


if len(sys.argv) < 4 or sys.argv[1] not in ("cond6", "cond7") or sys.argv[2] not in ("kldown", "ceup"):
    raise SystemExit("Usage: run_intervention_kl_ce_balance.py <cond6|cond7> <kldown|ceup> <multiplier> [final_iters]")
_base = sys.argv[1]
_arm = sys.argv[2]
_mult = float(sys.argv[3])
_final_iters = int(sys.argv[4]) if len(sys.argv) > 4 else 30
_kl_mult = _mult if _arm == "kldown" else 1.0
_ce_mult = _mult if _arm == "ceup" else 1.0
_run_name = f"{_base}_{_arm}_{_mult}"
_disc_fix_on = (_base == "cond7")
sys.argv = ["run_intervention_kl_ce_balance"]
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
if not _disc_fix_on:
    args.no_discriminator_fix = True
args.domain_keyed_proto = True

args.seed = 0
set_seed(args)

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=2)
args.num_users = len(train_loader_list)

if os.path.isdir("/kaggle/working"):
    raise RuntimeError("This intervention resumes from a Drive-resident checkpoint -- run on Colab with "
                        "Drive mounted, not Kaggle.")
elif os.path.isdir("/content/drive/MyDrive"):
    drive_root_candidate = f"/content/drive/MyDrive/{_base}_seed0_round10_checkpoint.pt"
    original_path_candidate = f"/content/drive/MyDrive/FedPall/limitation_2/component_ablation_m2/{_base}/{_base}_seed0_round10_checkpoint.pt"
    if os.path.exists(drive_root_candidate):
        source_round10_ckpt = drive_root_candidate
    elif os.path.exists(original_path_candidate):
        source_round10_ckpt = original_path_candidate
    else:
        raise FileNotFoundError(
            f"Could not find {_base}'s round10 checkpoint at either:\n  {drive_root_candidate}\n  {original_path_candidate}\n"
            f"Upload {_base}_seed0_round10_checkpoint.pt (from C:\\Users\\jiaya\\Downloads\\l2_component_ablation\\) "
            f"to your Drive root via the browser (large-file uploads via files.upload() are unreliable)."
        )
    intervention_base = f"/content/drive/MyDrive/FedPall/limitation_2/intervention_kl_ce_balance/{_run_name}"
else:
    raise RuntimeError(
        "Drive not mounted. On Colab:\n"
        "  from google.colab import drive\n"
        "  drive.mount('/content/drive')\n"
    )

os.makedirs(intervention_base, exist_ok=True)
checkpoint_path = f"{intervention_base}/{_run_name}_seed0_checkpoint.pt"
if not os.path.exists(checkpoint_path):
    shutil.copy(source_round10_ckpt, checkpoint_path)
    print(f"Seeded {checkpoint_path} from {_base}'s round10 checkpoint ({source_round10_ckpt}).")
else:
    print(f"{checkpoint_path} already exists -- resuming this arm's own in-progress run, not reseeding.")

acc_log_path = f"{intervention_base}/{_run_name}_seed0_acc.csv"
weights_dir = f"{intervention_base}/weights_seed0/"

print(f"Intervention base={_base} arm={_arm} multiplier={_mult}  (kl_weight_multiplier={_kl_mult}, "
      f"ce_weight_multiplier={_ce_mult}), resuming from {_base} round10, target final round={_final_iters-1}")

args.iters = _final_iters
train_loss, accuracy_list, datasets_name = ours(
    args, train_loader_list, test_loader_list, client_domains,
    checkpoint_path=checkpoint_path, checkpoint_every=5,
    acc_log_path=acc_log_path, weights_dir=weights_dir,
    lambda_G=0.05,
    kl_weight_multiplier=_kl_mult, ce_weight_multiplier=_ce_mult,
)

stage_snapshot = f"{intervention_base}/{_run_name}_seed0_round{_final_iters}_checkpoint.pt"
shutil.copy(checkpoint_path, stage_snapshot)
print(f"\nDone. Final round={_final_iters-1}. Snapshot: {stage_snapshot}")
for domain in datasets_name:
    print(f"  {domain}: acc={accuracy_list[domain][-1]:.4f}")

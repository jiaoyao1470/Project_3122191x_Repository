import sys
import os
import shutil
import torch
sys.path.insert(0, os.getcwd())

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


sys.argv = ["calib_run_for_lambda_FB"]
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
args.device = args.device if torch.cuda.is_available() else "cpu"

args.domain_keyed_proto = True
args.no_discriminator_fix = False
args.adaptive_local_batch = False
LAMBDA_G = 0.05

args.seed = 0
set_seed(args)

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=4)
args.num_users = len(train_loader_list)

FEATURE_BANK_DOMAINS = {"dslr"}
PROBE_ROUNDS = [5, 10, 15, 20]

RESUME_INPUT_DIR = None

base_dir = "calib_lambda_FB_run"
if RESUME_INPUT_DIR is None and os.path.exists(base_dir):
    stale = [
        os.path.join(base_dir, "seed0_checkpoint.pt"),
        os.path.join(base_dir, "seed0_acc.csv"),
    ] + [
        os.path.join(base_dir, f"seed0_checkpoint_round{r}.pt") for r in [4, 9, 14, 19]
    ]
    existing = [p for p in stale if os.path.exists(p)]
    if existing:
        raise RuntimeError(
            f"{base_dir}/ contains old calibration outputs. Delete or rename the directory before "
            f"starting a fresh calibration run, or set RESUME_INPUT_DIR if this is an intentional "
            f"resume. Found: {existing}"
        )
os.makedirs(base_dir, exist_ok=True)
checkpoint_path = f"{base_dir}/seed0_checkpoint.pt"
acc_log_path = f"{base_dir}/seed0_acc.csv"
weights_dir = f"{base_dir}/weights_seed0/"

if RESUME_INPUT_DIR is not None:
    if not os.path.isdir(RESUME_INPUT_DIR):
        raise RuntimeError(f"RESUME_INPUT_DIR does not exist: {RESUME_INPUT_DIR}")
    for fname in ["seed0_checkpoint.pt", "seed0_acc.csv"]:
        src = os.path.join(RESUME_INPUT_DIR, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(base_dir, fname))
            print(f"Restored {fname} from RESUME_INPUT_DIR.")
    for _r in [t - 1 for t in PROBE_ROUNDS]:
        fname = f"seed0_checkpoint_round{_r}.pt"
        src = os.path.join(RESUME_INPUT_DIR, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(base_dir, fname))
            print(f"Restored {fname} from RESUME_INPUT_DIR.")
    src_weights = os.path.join(RESUME_INPUT_DIR, "weights_seed0")
    if os.path.isdir(src_weights):
        shutil.copytree(src_weights, weights_dir, dirs_exist_ok=True)
        print("Restored weights_seed0/ from RESUME_INPUT_DIR.")
    if not os.path.exists(checkpoint_path):
        print("WARNING: RESUME_INPUT_DIR is set but no seed0_checkpoint.pt found there -- "
              "starting fresh (any already-completed snapshots restored above will still be reused).")

for target_iters in PROBE_ROUNDS:
    args.iters = target_iters
    snapshot_path = f"{base_dir}/seed0_checkpoint_round{target_iters - 1}.pt"
    if os.path.exists(snapshot_path):
        print(f"\n=== stage (iters={target_iters}) already complete -- {snapshot_path} exists, skipping ===")
        continue
    print(f"\n=== stage: training up to round {target_iters - 1} (checkpoint_every=5) ===")
    train_loss, accuracy_list, datasets_name = ours(
        args, train_loader_list, test_loader_list, client_domains,
        checkpoint_path=checkpoint_path, checkpoint_every=5,
        acc_log_path=acc_log_path, weights_dir=weights_dir,
        lambda_G=LAMBDA_G,
        lambda_mixup=0.0, domain_balanced_gc=False,
        feature_bank_domains=FEATURE_BANK_DOMAINS, lambda_FB=0.0,
    )
    _verify_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _actual_round = _verify_ckpt["round"]
    if _actual_round != target_iters - 1:
        raise RuntimeError(
            f"Refusing to snapshot: {checkpoint_path} is at round {_actual_round}, expected round "
            f"{target_iters - 1} for this stage. Inspect {base_dir}/ by hand (existing snapshots "
            f"and the growing checkpoint's actual round) before retrying."
        )
    shutil.copy(checkpoint_path, snapshot_path)
    print(f"snapshot saved: {snapshot_path}")
    for d in datasets_name:
        print(f"  {d}: acc={accuracy_list[d][-1]:.4f}")

print("\n=== all 4 probe-point snapshots ready ===")

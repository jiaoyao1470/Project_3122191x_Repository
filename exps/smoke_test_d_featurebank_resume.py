import sys
import os
import shutil
import csv
import torch
sys.path.insert(0, os.getcwd())

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


sys.argv = ["smoke_test_d_featurebank_resume"]


def build_args():
    args = args_parser()
    args.exp = 1
    args.mode = "ours"
    args.lr = 0.01
    args.number_workers = 0
    args.device = "cpu"
    args.dataset = "office"
    args.num_classes = 10
    args.size = 64
    args.batch = 32
    args.wk_iters = 1
    args.adcol_epoch = 1
    args.adcol_mu = 0.1
    args.adcol_beta = 0.1
    args.domain_keyed_proto = True
    args.no_discriminator_fix = False
    args.adaptive_local_batch = False
    args.seed = 0
    set_seed(args)
    return args


LAMBDA_G = 0.05
FEATURE_BANK_DOMAINS = {"dslr"}
LAMBDA_FB = 0.0268

session_dir = "smoke_dfb_resume_run"
mock_input_dir = "smoke_dfb_resume_MOCK_INPUT"
for d in [session_dir, mock_input_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)

print("=" * 20 + " Session A: fresh run, 2 rounds " + "=" * 20)
argsA = build_args()
tlA, cdA, telA = prepare_data_office_multi_clients(argsA, dslr_client_override=4)
argsA.num_users = len(tlA)
os.makedirs(session_dir, exist_ok=True)
ckptA = f"{session_dir}/seed0_checkpoint.pt"
accA = f"{session_dir}/seed0_acc.csv"
weightsA = f"{session_dir}/weights_seed0/"
argsA.iters = 2
_, accdictA, namesA = ours(
    argsA, tlA, telA, cdA, checkpoint_path=ckptA, checkpoint_every=1,
    acc_log_path=accA, weights_dir=weightsA, lambda_G=LAMBDA_G,
    feature_bank_domains=FEATURE_BANK_DOMAINS, lambda_FB=LAMBDA_FB,
)
with open(accA) as f:
    rowsA = list(csv.reader(f))
n_data_rowsA = len(rowsA) - 1
print(f"Session A complete. acc.csv has {n_data_rowsA} data rows (expect 2 rounds x 4 domains = 8).")
assert n_data_rowsA == 8, f"FAIL: expected 8 rows after 2 rounds, got {n_data_rowsA}"
weightsA_files = sorted(os.listdir(weightsA)) if os.path.isdir(weightsA) else []
print(f"Session A weights_seed0/ contains: {weightsA_files}")
assert len(weightsA_files) > 0, "FAIL: weights_seed0/ is empty after session A (best_acc started at 0, should have saved something)"

print("\n" + "=" * 20 + " Simulating full Kaggle session restart " + "=" * 20)
shutil.copytree(session_dir, mock_input_dir)
shutil.rmtree(session_dir)
assert not os.path.exists(session_dir)
print(f"Copied {session_dir}/ -> {mock_input_dir}/ (mock RESUME_INPUT_DIR), deleted local {session_dir}/.")

print("\n" + "=" * 20 + " Session B: RESUME_INPUT_DIR set, restore 3 things, continue to round2 " + "=" * 20)
RESUME_INPUT_DIR = mock_input_dir
_tag_dir = session_dir
os.makedirs(_tag_dir, exist_ok=True)
checkpoint_path = f"{_tag_dir}/seed0_checkpoint.pt"
acc_log_path = f"{_tag_dir}/seed0_acc.csv"
weights_dir = f"{_tag_dir}/weights_seed0/"

_resume_ckpt = f"{RESUME_INPUT_DIR}/seed0_checkpoint.pt"
shutil.copy(_resume_ckpt, checkpoint_path)
print(f"Restored checkpoint from RESUME_INPUT_DIR.")
_resume_acc = f"{RESUME_INPUT_DIR}/seed0_acc.csv"
shutil.copy(_resume_acc, acc_log_path)
print(f"Restored acc.csv from RESUME_INPUT_DIR.")
_resume_weights = f"{RESUME_INPUT_DIR}/weights_seed0"
shutil.copytree(_resume_weights, weights_dir, dirs_exist_ok=True)
print(f"Restored weights_seed0/ from RESUME_INPUT_DIR.")

with open(acc_log_path) as f:
    rows_after_restore = list(csv.reader(f))
assert len(rows_after_restore) - 1 == 8, (
    f"FAIL: acc.csv right after restore (before any new training) should have 8 data rows, "
    f"got {len(rows_after_restore) - 1}"
)
weights_after_restore = sorted(os.listdir(weights_dir))
assert weights_after_restore == weightsA_files, (
    f"FAIL: restored weights_seed0/ contents differ from session A's: "
    f"{weights_after_restore} vs {weightsA_files}"
)
print("Restore verified: acc.csv has session A's 8 rows, weights_seed0/ matches session A's files.")

argsB = build_args()
tlB, cdB, telB = prepare_data_office_multi_clients(argsB, dslr_client_override=4)
argsB.num_users = len(tlB)
argsB.iters = 3
_, accdictB, namesB = ours(
    argsB, tlB, telB, cdB, checkpoint_path=checkpoint_path, checkpoint_every=1,
    acc_log_path=acc_log_path, weights_dir=weights_dir, lambda_G=LAMBDA_G,
    feature_bank_domains=FEATURE_BANK_DOMAINS, lambda_FB=LAMBDA_FB,
)
print("Session B complete (trained round2, resuming from restored round1 state).")

with open(acc_log_path) as f:
    rowsB = list(csv.reader(f))
n_data_rowsB = len(rowsB) - 1
print(f"Final acc.csv has {n_data_rowsB} data rows (expect 3 rounds x 4 domains = 12, "
      f"NOT just round2's 4).")
assert n_data_rowsB == 12, (
    f"FAIL: expected 12 rows (full r0-r2 continuity), got {n_data_rowsB} -- acc.csv restore is not "
    f"preserving pre-resume rounds."
)

weightsB_files = sorted(os.listdir(weights_dir))
print(f"Final weights_seed0/ contains: {weightsB_files}")
assert len(weightsB_files) > 0, "FAIL: weights_seed0/ is empty after session B"
for fname in weightsA_files:
    assert fname in weightsB_files, (
        f"FAIL: session A's weight file {fname} is missing after session B -- the restored "
        f"weights_dir got silently overwritten/emptied instead of preserved."
    )
print("PASS: pre-resume weight files are still present after session B.")

print("\n" + "=" * 20 + " Checkpoint feature_bank_prev continuity check " + "=" * 20)
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
fb = ckpt.get("feature_bank_prev", {})
assert "dslr" in fb and len(fb["dslr"]) == 4, (
    f"FAIL: feature_bank_prev['dslr'] should have 4 populated clients after round2, got {fb.get('dslr', {})}"
)
print(f"feature_bank_prev['dslr'] populated for clients: {sorted(fb['dslr'].keys())} -- "
      f"confirms the bank did not cold-start on resume.")

print("\n=== ALL CHECKS COMPLETE ===")

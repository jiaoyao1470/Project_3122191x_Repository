import sys
import os
import shutil
import torch
sys.path.insert(0, os.getcwd())

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


sys.argv = ["smoke_test_calib_resume"]


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

session1_dir = "calib_smoke_resume_run"
mock_input_dir = "calib_smoke_resume_MOCK_INPUT"
for d in [session1_dir, mock_input_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)

print("=" * 20 + " Session 1: fresh run, PROBE_ROUNDS=[1] " + "=" * 20)
args1 = build_args()
tl1, cd1, tel1 = prepare_data_office_multi_clients(args1, dslr_client_override=4)
args1.num_users = len(tl1)
os.makedirs(session1_dir, exist_ok=True)
ckpt1 = f"{session1_dir}/seed0_checkpoint.pt"
args1.iters = 1
_, acc1, names1 = ours(
    args1, tl1, tel1, cd1, checkpoint_path=ckpt1, checkpoint_every=1,
    acc_log_path=f"{session1_dir}/seed0_acc.csv", weights_dir=f"{session1_dir}/weights_seed0/",
    lambda_G=LAMBDA_G, feature_bank_domains=FEATURE_BANK_DOMAINS, lambda_FB=0.0,
)
snap1 = f"{session1_dir}/seed0_checkpoint_round0.pt"
shutil.copy(ckpt1, snap1)
print(f"Session 1 complete. snapshot: {snap1}")

print("\n" + "=" * 20 + " Simulating Kaggle 'Add Input' + new session " + "=" * 20)
shutil.copytree(session1_dir, mock_input_dir)
shutil.rmtree(session1_dir)
print(f"Copied {session1_dir}/ -> {mock_input_dir}/ (stand-in for a mounted Kaggle Input), "
      f"then deleted the local {session1_dir}/ (stand-in for a fresh session with nothing local).")
assert not os.path.exists(session1_dir)
assert os.path.exists(os.path.join(mock_input_dir, "seed0_checkpoint_round0.pt"))

print("\n" + "=" * 20 + " Session 2: RESUME_INPUT_DIR set, PROBE_ROUNDS=[1, 2] " + "=" * 20)
RESUME_INPUT_DIR = mock_input_dir
base_dir = session1_dir
PROBE_ROUNDS = [1, 2]

if RESUME_INPUT_DIR is None and os.path.exists(base_dir):
    raise RuntimeError("should not happen in this test -- fresh-start guard path")
os.makedirs(base_dir, exist_ok=True)
checkpoint_path = f"{base_dir}/seed0_checkpoint.pt"
acc_log_path = f"{base_dir}/seed0_acc.csv"
weights_dir = f"{base_dir}/weights_seed0/"

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

args2 = build_args()
tl2, cd2, tel2 = prepare_data_office_multi_clients(args2, dslr_client_override=4)
args2.num_users = len(tl2)

stage_ran = {}
for target_iters in PROBE_ROUNDS:
    args2.iters = target_iters
    snapshot_path = f"{base_dir}/seed0_checkpoint_round{target_iters - 1}.pt"
    if os.path.exists(snapshot_path):
        print(f"\n=== stage (iters={target_iters}) already complete -- {snapshot_path} exists, skipping ===")
        stage_ran[target_iters] = False
        continue
    print(f"\n=== stage: training up to round {target_iters - 1} ===")
    train_loss, accuracy_list, datasets_name = ours(
        args2, tl2, tel2, cd2, checkpoint_path=checkpoint_path, checkpoint_every=1,
        acc_log_path=acc_log_path, weights_dir=weights_dir, lambda_G=LAMBDA_G,
        feature_bank_domains=FEATURE_BANK_DOMAINS, lambda_FB=0.0,
    )
    _verify_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _actual_round = _verify_ckpt["round"]
    if _actual_round != target_iters - 1:
        raise RuntimeError(
            f"Refusing to snapshot: {checkpoint_path} is at round {_actual_round}, expected round "
            f"{target_iters - 1}."
        )
    shutil.copy(checkpoint_path, snapshot_path)
    print(f"snapshot saved: {snapshot_path}")
    stage_ran[target_iters] = True

print(f"\nstage_ran = {stage_ran}")
assert stage_ran[1] is False, "FAIL: stage 1 (already complete) should have been skipped, but it re-ran"
assert stage_ran[2] is True, "FAIL: stage 2 (new) should have actually trained, but it didn't"
assert os.path.exists(f"{base_dir}/seed0_checkpoint_round1.pt"), "FAIL: stage 2 snapshot missing"
print("PASS: stage 1 skipped, stage 2 trained and snapshotted correctly.")

print("\n" + "=" * 20 + " Corrupted-checkpoint verification check (no training) " + "=" * 20)
fake_ckpt_path = f"{base_dir}/fake_checkpoint_for_verify_test.pt"
torch.save({"round": 99}, fake_ckpt_path)
_verify_ckpt = torch.load(fake_ckpt_path, map_location="cpu", weights_only=False)
_actual_round = _verify_ckpt["round"]
_target_iters = 2
try:
    if _actual_round != _target_iters - 1:
        raise RuntimeError(
            f"Refusing to snapshot: fake checkpoint is at round {_actual_round}, expected round "
            f"{_target_iters - 1}."
        )
    print("FAIL: expected RuntimeError, but the verification did not fire")
except RuntimeError as e:
    print(f"PASS: RuntimeError raised as expected: {e}")

print("\n=== ALL CHECKS COMPLETE ===")

import sys
import os
sys.path.insert(0, os.getcwd())

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_PACS_multi_clients

_seed_arg = sys.argv[1] if len(sys.argv) > 1 else None
sys.argv = ["run_domain_keyed_proto_alone_pacs"]
args = args_parser()

args.exp = 1
args.mode = "ours"
args.iters = 100
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
args.no_discriminator_fix = True
args.device = args.device if __import__("torch").cuda.is_available() else "cpu"

seeds_to_run = [0]
seeds = [int(_seed_arg)] if _seed_arg is not None else seeds_to_run

if os.path.isdir("/kaggle/working"):
    base_dir = "/kaggle/working/result"
elif os.path.isdir("/content/drive/MyDrive"):
    base_dir = "/content/drive/MyDrive/FedPall/limitation_2/domain_keyed_proto_alone_pacs"
else:
    raise RuntimeError(
        "Neither /kaggle/working nor /content/drive/MyDrive is available.\n"
        "On Colab: mount Drive first --\n"
        "  from google.colab import drive\n"
        "  drive.mount('/content/drive')\n"
        "On Kaggle: /kaggle/working should always exist -- check you're running inside a Kaggle "
        "notebook environment."
    )
os.makedirs(base_dir, exist_ok=True)

RESUME_INPUT_DIR = None

for seed in seeds:
    args.seed = seed
    set_seed(args)

    train_loader_list, client_domains, test_loader_list = prepare_data_PACS_multi_clients(args)
    args.num_users = len(train_loader_list)

    tag = f"domain_keyed_proto_alone_pacs_seed{seed}"
    checkpoint_path = f"{base_dir}/{tag}_checkpoint.pt"
    acc_log_path = f"{base_dir}/{tag}_acc.csv"
    weights_dir = f"{base_dir}/weights_{tag}/"

    if RESUME_INPUT_DIR is not None:
        resume_ckpt = f"{RESUME_INPUT_DIR}/{tag}_checkpoint.pt"
        if not os.path.exists(resume_ckpt):
            raise RuntimeError(f"RESUME_INPUT_DIR is set but checkpoint not found at: {resume_ckpt}")
        import shutil
        shutil.copy(resume_ckpt, checkpoint_path)
        print(f"Copied checkpoint from mounted input dataset: {resume_ckpt} -> {checkpoint_path}")
        resume_acc = f"{RESUME_INPUT_DIR}/{tag}_acc.csv"
        if os.path.exists(resume_acc):
            shutil.copy(resume_acc, acc_log_path)
            print(f"Copied acc log from mounted input dataset: {resume_acc}")

    if not os.path.exists(checkpoint_path) and os.path.exists(acc_log_path):
        os.remove(acc_log_path)

    print(f"Running setting=domain_keyed_proto ALONE (PACS) seed={seed} (checkpoint: {checkpoint_path})")
    train_loss, accuracy_list, datasets_name = ours(
        args, train_loader_list, test_loader_list, client_domains,
        checkpoint_path=checkpoint_path, checkpoint_every=5,
        acc_log_path=acc_log_path, weights_dir=weights_dir,
    )
    print(f"\n=== seed={seed} finished without crashing ===\n")
    for domain in datasets_name:
        print(f"  {domain}: final acc={accuracy_list[domain][-1]:.4f}, first acc={accuracy_list[domain][0]:.4f}")
    print(f"weights saved under {weights_dir}")

print("\nAll seeds done.")
print("\nreference points (PACS, robust standard where noted, seed0-only where noted):")
print("  FedPall paper reproduction (one-client):        60.07% (3-seed)")
print("  Multi-Client baseline (neither fix):             54.43% (3-seed)")
print("  C (discriminator-fix+domain_keyed_proto):        55.39% (seed0 only)")
print("  discriminator-fix ALONE:                          56.09% (seed0 only, robust standard)")
if base_dir.startswith("/kaggle/"):
    print("\nIMPORTANT (Kaggle): click 'Save Version' / commit this notebook PERIODICALLY during the run")
    print("(not just at the end) -- /kaggle/working/ does NOT persist automatically, and a session")
    print("restart mid-training (10+ hour ETA) is a real possibility, not just an edge case.")
else:
    print("\ncheckpoint is on Drive -- safe across Colab session restarts, no extra step needed.")

import sys
import os
sys.path.insert(0, os.getcwd())

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_PACS_multi_clients

_seed_arg = sys.argv[1] if len(sys.argv) > 1 else None
sys.argv = ["run_discriminator_domain_keyed_pacs"]
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
args.device = args.device if __import__("torch").cuda.is_available() else "cpu"
args.seed = int(_seed_arg) if _seed_arg is not None else 0
set_seed(args)

train_loader_list, client_domains, test_loader_list = prepare_data_PACS_multi_clients(args)
args.num_users = len(train_loader_list)

RESUME_INPUT_DIR = None

if os.path.isdir("/kaggle/working"):
    base_dir = "/kaggle/working/result"
elif os.path.isdir("/content/drive/MyDrive"):
    base_dir = "/content/drive/MyDrive/FedPall/limitation_2/discriminator_domain_keyed_pacs"
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
tag = f"discriminator_domain_keyed_pacs_seed{args.seed}"
checkpoint_path = f"{base_dir}/{tag}_checkpoint.pt"

if RESUME_INPUT_DIR is not None:
    resume_ckpt = f"{RESUME_INPUT_DIR}/{tag}_checkpoint.pt"
    if not os.path.exists(resume_ckpt):
        raise RuntimeError(f"RESUME_INPUT_DIR is set but checkpoint not found at: {resume_ckpt}")
    import shutil
    shutil.copy(resume_ckpt, checkpoint_path)
    print(f"Copied checkpoint from mounted input dataset: {resume_ckpt} -> {checkpoint_path}")
    resume_acc = f"{RESUME_INPUT_DIR}/{tag}_acc.csv"
    if os.path.exists(resume_acc):
        shutil.copy(resume_acc, f"{base_dir}/{tag}_acc.csv")
        print(f"Copied acc log from mounted input dataset: {resume_acc}")

acc_log_path = f"{base_dir}/{tag}_acc.csv"
weights_dir = f"{base_dir}/weights_{tag}/"

if not os.path.exists(checkpoint_path) and os.path.exists(acc_log_path):
    os.remove(acc_log_path)

print(f"Running setting=discriminator+domain_keyed_proto(PACS) seed={args.seed} (checkpoint: {checkpoint_path})")
train_loss, accuracy_list, datasets_name = ours(
    args, train_loader_list, test_loader_list, client_domains,
    checkpoint_path=checkpoint_path, checkpoint_every=5,
    acc_log_path=acc_log_path, weights_dir=weights_dir,
)
print(f"\n=== training finished without crashing ===\n")
print(f"seed={args.seed} done. acc log at {acc_log_path}")
for domain in datasets_name:
    print(f"  {domain}: final acc={accuracy_list[domain][-1]:.4f}, first acc={accuracy_list[domain][0]:.4f}")

print(f"\nweights saved under {weights_dir}")
print(f"checkpoint at: {checkpoint_path}")
if base_dir.startswith("/kaggle/"):
    print("\nIMPORTANT (Kaggle): click 'Save Version' / commit this notebook PERIODICALLY during the run")
    print("(not just at the end) -- /kaggle/working/ does NOT persist automatically, and this run is long")
    print("enough that a session restart mid-training is a real possibility, not just a remote edge case.")
else:
    print("\ncheckpoint is on Drive -- safe across Colab session restarts, no extra step needed.")

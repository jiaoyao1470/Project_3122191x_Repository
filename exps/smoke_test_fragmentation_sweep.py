import sys
import os
sys.path.insert(0, os.getcwd())

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


sys.argv = ["smoke_test_fragmentation_sweep"]
args = args_parser()
args.exp = 1
args.mode = "ours"
args.iters = 1
args.lr = 0.01
args.number_workers = 0
args.dataset = "office"
args.num_classes = 10
args.size = 64
args.batch = 32
args.wk_iters = 2
args.adcol_mu = 0.1
args.adcol_beta = 0.1
args.adcol_epoch = 1
args.domain_keyed_proto = True
args.device = "cpu"
args.seed = 0
set_seed(args)

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=4)
args.num_users = len(train_loader_list)
print(f"client_domains: {client_domains}")
print(f"num_users: {args.num_users}")

tmp_dir = "C:/Users/jiaya/AppData/Local/Temp/claude/smoke_test_fragmentation_sweep"
os.makedirs(tmp_dir, exist_ok=True)

train_loss, accuracy_list, datasets_name = ours(
    args, train_loader_list, test_loader_list, client_domains,
    checkpoint_path=f"{tmp_dir}/checkpoint.pt", checkpoint_every=1,
    acc_log_path=f"{tmp_dir}/acc.csv", weights_dir=f"{tmp_dir}/weights/",
    lambda_G=0.05,
)
print("Smoke test training completed without error.")
for domain in datasets_name:
    print(f"  {domain}: acc={accuracy_list[domain][-1]:.4f}")

import torch
ckpt = torch.load(f"{tmp_dir}/checkpoint.pt", weights_only=False)
print(f"\nCheckpoint keys: {list(ckpt.keys())}")
print(f"local_models_state length: {len(ckpt['local_models_state'])} (expect {args.num_users})")
dslr_indices = [i for i, d in enumerate(client_domains) if d == 'dslr']
print(f"dslr client indices: {dslr_indices} (expect 4 of them)")
print("\nSmoke test PASSED.")

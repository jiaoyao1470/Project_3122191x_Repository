import sys
import os
sys.path.insert(0, os.getcwd())

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


sys.argv = ["smoke_test_domain_keyed_proto"]

def build_args(domain_keyed):
    args = args_parser()
    args.exp = 1
    args.mode = "ours"
    args.iters = 2
    args.lr = 0.01
    args.number_workers = 0
    args.device = "cpu"
    args.dataset = "office"
    args.seed = 0
    set_seed(args)
    args.num_classes = 10
    args.size = 64
    args.batch = 32
    args.wk_iters = 2
    args.adcol_epoch = 1
    args.adcol_mu = 0.1
    args.adcol_beta = 0.1
    args.domain_keyed_proto = domain_keyed
    return args

scratch = os.environ.get("TEMP", ".") + "/l2_domain_keyed_smoke_test/"
os.makedirs(scratch, exist_ok=True)

print("=" * 20 + " Phase (a): domain_keyed_proto=False (regression control) " + "=" * 20)
args_false = build_args(domain_keyed=False)
train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args_false)
args_false.num_users = len(train_loader_list)
train_loss_f, acc_f, datasets_name_f = ours(
    args_false, train_loader_list, test_loader_list, client_domains,
    checkpoint_path=None, acc_log_path=scratch + "false_acc.csv",
    weights_dir=scratch + "weights_false/",
)
print("Phase (a) complete, no crash.")
for d in datasets_name_f:
    print(f"  {d}: acc history = {[round(a,4) for a in acc_f[d]]}")

print("\n" + "=" * 20 + " Phase (b): domain_keyed_proto=True (new path) " + "=" * 20)
args_true = build_args(domain_keyed=True)
train_loader_list2, client_domains2, test_loader_list2 = prepare_data_office_multi_clients(args_true)
args_true.num_users = len(train_loader_list2)
train_loss_t, acc_t, datasets_name_t = ours(
    args_true, train_loader_list2, test_loader_list2, client_domains2,
    checkpoint_path=None, acc_log_path=scratch + "true_acc.csv",
    weights_dir=scratch + "weights_true/",
)
print("Phase (b) complete, no crash.")
for d in datasets_name_t:
    print(f"  {d}: acc history = {[round(a,4) for a in acc_t[d]]}")

print("\n" + "=" * 20 + " Phase (c): checkpoint save + resume, domain_keyed_proto=True " + "=" * 20)
args_resume = build_args(domain_keyed=True)
train_loader_list3, client_domains3, test_loader_list3 = prepare_data_office_multi_clients(args_resume)
args_resume.num_users = len(train_loader_list3)
ckpt_path = scratch + "resume_checkpoint.pt"
if os.path.exists(ckpt_path):
    os.remove(ckpt_path)
args_resume.iters = 2
ours(args_resume, train_loader_list3, test_loader_list3, client_domains3,
     checkpoint_path=ckpt_path, checkpoint_every=1,
     acc_log_path=scratch + "resume_acc.csv", weights_dir=scratch + "weights_resume/")
print("First 2 rounds + checkpoint save: OK")

args_resume2 = build_args(domain_keyed=True)
args_resume2.iters = 4
train_loader_list4, client_domains4, test_loader_list4 = prepare_data_office_multi_clients(args_resume2)
args_resume2.num_users = len(train_loader_list4)
train_loss_r, acc_r, datasets_name_r = ours(
    args_resume2, train_loader_list4, test_loader_list4, client_domains4,
    checkpoint_path=ckpt_path, checkpoint_every=1,
    acc_log_path=scratch + "resume_acc.csv", weights_dir=scratch + "weights_resume/",
)
print("Resume + 2 more rounds: OK (tuple-keyed global_proto survived torch.save/load)")
for d in datasets_name_r:
    print(f"  {d}: acc history after resume = {[round(a,4) for a in acc_r[d]]}")

print("\n=== ALL PHASES COMPLETE ===")

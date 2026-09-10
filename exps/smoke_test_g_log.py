import sys
import os
sys.path.insert(0, os.getcwd())

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


sys.argv = ["smoke_test_g_log"]

def build_args():
    args = args_parser()
    args.exp = 1
    args.mode = "ours"
    args.iters = 1
    args.lr = 0.01
    args.number_workers = 0
    args.device = "cpu"
    args.dataset = "office"
    args.seed = 0
    set_seed(args)
    args.num_classes = 10
    args.size = 64
    args.batch = 32
    args.wk_iters = 1
    args.adcol_epoch = 1
    args.adcol_mu = 0.1
    args.adcol_beta = 0.1
    args.domain_keyed_proto = True
    return args

scratch = os.environ.get("TEMP", ".") + "/l2_g_log_smoke_test_lean/"
os.makedirs(scratch, exist_ok=True)

print("=" * 20 + " Phase (a): 1 round with g_log_path set " + "=" * 20)
args_a = build_args()
tl, cd, tel = prepare_data_office_multi_clients(args_a)
args_a.num_users = len(tl)
g_log_path = scratch + "g_log.csv"
if os.path.exists(g_log_path):
    os.remove(g_log_path)
_, acc, names = ours(
    args_a, tl, tel, cd, checkpoint_path=None, acc_log_path=scratch + "a_acc.csv",
    weights_dir=scratch + "weights_a/", g_log_path=g_log_path,
)
print("Phase (a): 1 round completed without crashing -> PASS")

with open(g_log_path) as f:
    lines = f.readlines()
print(f"g_log.csv: {len(lines)} lines")
print(f"header: {lines[0].strip()}")
header_ok = lines[0].strip() == "round,domain,avg_weight_across_classes,n_classes,G_avg_norm"
print(f"header matches expected -> {'PASS' if header_ok else 'FAIL'}")
values_ok = True
for line in lines[1:]:
    round_s, domain_s, w_s, n_s, norm_s = line.strip().split(",")
    w, n, norm = float(w_s), int(n_s), float(norm_s)
    if not (0.0 <= w <= 1.0 + 1e-6):
        values_ok = False
        print(f"  suspicious weight value: {line.strip()}")
    if not (norm == norm) or norm <= 0:
        values_ok = False
        print(f"  suspicious G_avg_norm value: {line.strip()}")
print(f"all weight/norm values in plausible ranges -> {'PASS' if values_ok else 'FAIL'}")
print(f"content:\n{''.join(lines)}")

print("\n" + "=" * 20 + " Phase (b): checkpoint save + resume (crash check only) " + "=" * 20)
args_b1 = build_args()
tl2, cd2, tel2 = prepare_data_office_multi_clients(args_b1)
args_b1.num_users = len(tl2)
ckpt_path = scratch + "resume_ckpt.pt"
if os.path.exists(ckpt_path):
    os.remove(ckpt_path)
resume_glog = scratch + "resume_g_log.csv"
if os.path.exists(resume_glog):
    os.remove(resume_glog)
ours(args_b1, tl2, tel2, cd2, checkpoint_path=ckpt_path, checkpoint_every=1,
     acc_log_path=scratch + "b_acc.csv", weights_dir=scratch + "weights_b/", g_log_path=resume_glog)
print("Round 1 + checkpoint save: OK")

args_b2 = build_args()
args_b2.iters = 2
tl3, cd3, tel3 = prepare_data_office_multi_clients(args_b2)
args_b2.num_users = len(tl3)
ours(args_b2, tl3, tel3, cd3, checkpoint_path=ckpt_path, checkpoint_every=1,
     acc_log_path=scratch + "b_acc.csv", weights_dir=scratch + "weights_b/", g_log_path=resume_glog)
print("Resume + 1 more round: OK (prev_G_state survived torch.save/load without crashing) -> PASS")

print("\n=== ALL PHASES COMPLETE ===")

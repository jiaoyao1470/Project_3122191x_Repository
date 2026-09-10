import sys
import os
sys.path.insert(0, os.getcwd())

import shutil
import torch

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients


scratch = os.environ.get("TEMP", ".") + "/l2_bn_affine_smoke_test/"
if os.path.isdir(scratch):
    shutil.rmtree(scratch)
os.makedirs(scratch, exist_ok=True)


def make_args():
    sys.argv = ["smoke_test_bn_affine_sync"]
    args = args_parser()
    args.exp = 1
    args.mode = "ours"
    args.iters = 2
    args.lr = 0.01
    args.number_workers = 0
    args.device = "cpu"
    args.dataset = "office"
    args.seed = 0
    args.num_classes = 10
    args.size = 64
    args.batch = 32
    args.wk_iters = 1
    args.adcol_epoch = 1
    args.adcol_mu = 0.1
    args.adcol_beta = 0.1
    args.domain_keyed_proto = True
    args.no_discriminator_fix = False
    set_seed(args)
    return args


def build(args):
    tl, cd, testl = prepare_data_office_multi_clients(args, dslr_client_override=4)
    args.num_users = len(tl)
    return tl, cd, testl


def bn_affine(model):
    out = []
    for m in model.features.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.weight is not None:
            out.append((m.weight.data.clone(), m.bias.data.clone()))
    return out


def max_pair_diff(models):
    refs = [bn_affine(m) for m in models]
    worst = 0.0
    for layer in range(len(refs[0])):
        for j in range(1, len(refs)):
            worst = max(
                worst,
                (refs[j][layer][0] - refs[0][layer][0]).abs().max().item(),
                (refs[j][layer][1] - refs[0][layer][1]).abs().max().item(),
            )
    return worst


print("=" * 78)
print("PHASE (a): baseline -- new argument not passed at all")
args = make_args()
tl, cd, testl = build(args)
_, acc_a, names = ours(args, tl, testl, cd, checkpoint_path=scratch + "a/a_ckpt.pt")
print("Phase (a) complete.")

print("=" * 78)
print("PHASE (b): bn_affine_sync_domains=None explicitly")
args = make_args()
tl, cd, testl = build(args)
_, acc_b, _ = ours(args, tl, testl, cd, checkpoint_path=scratch + "b/b_ckpt.pt",
                   bn_affine_sync_domains=None)
worst = max(abs(acc_a[d][r] - acc_b[d][r]) for d in names for r in range(len(acc_a[d])))
print(f"  max |acc(a) - acc(b)| over all rounds/domains = {worst:.10f}")
print("  " + ("PASS -- accuracy trajectories identical" if worst == 0.0
              else "FAIL -- passing None changed the accuracy trajectory"))

ckpt_a = torch.load(scratch + "a/a_ckpt.pt", map_location="cpu", weights_only=False)
ckpt_b = torch.load(scratch + "b/b_ckpt.pt", map_location="cpu", weights_only=False)
state_identical = True
for sa, sb in zip(ckpt_a["local_models_state"], ckpt_b["local_models_state"]):
    for k in sa:
        if not torch.equal(sa[k], sb[k]):
            state_identical = False
for k in ckpt_a["global_classifier_state"]:
    if not torch.equal(ckpt_a["global_classifier_state"][k], ckpt_b["global_classifier_state"][k]):
        state_identical = False
for k in ckpt_a["discriminator_state"]:
    if not torch.equal(ckpt_a["discriminator_state"][k], ckpt_b["discriminator_state"][k]):
        state_identical = False
print("  " + ("PASS -- explicit None is byte-identical to omitting the argument (local_models, "
              "global_classifier, discriminator all torch.equal)" if state_identical
              else "FAIL -- passing None changed model state despite identical accuracy"))

print("=" * 78)
print("PHASE (c): bn_affine_sync_domains={'dslr'} -- sync must fire on dslr only")
args = make_args()
tl, cd, testl = build(args)
dslr_idx = [i for i, d in enumerate(cd) if d == "dslr"]
caltech_idx = [i for i, d in enumerate(cd) if d == "caltech"]
_, acc_c, _ = ours(args, tl, testl, cd, checkpoint_path=scratch + "c/c_ckpt.pt",
                   bn_affine_sync_domains={"dslr"})

ckpt = torch.load(scratch + "c/c_ckpt.pt", map_location="cpu", weights_only=False)
states = ckpt["local_models_state"]
from models import adcol_model


def load(i):
    m = adcol_model(num_classes=args.num_classes)
    m.load_state_dict(states[i])
    return m


d_worst = max_pair_diff([load(i) for i in dslr_idx])
c_worst = max_pair_diff([load(i) for i in caltech_idx])
print(f"  dslr clients {dslr_idx}: max pairwise BN-affine difference = {d_worst:.3e}")
print(f"  caltech clients {caltech_idx} (must NOT be synced): max pairwise diff = {c_worst:.3e}")
ok_c = d_worst == 0.0 and c_worst > 0.0
print("  " + ("PASS -- dslr synced exactly, caltech untouched" if ok_c else
              "FAIL -- dslr not identical, and/or caltech was wrongly synced"))

dslr_moved = any(acc_c["dslr"][r] != acc_c["dslr"][0] for r in range(len(acc_c["dslr"])))
print(f"  dslr accuracy across rounds: {acc_c['dslr']}")
print("  " + ("PASS -- dslr accuracy moved between rounds" if dslr_moved
              else "FAIL -- dslr accuracy identical every round, sync may have frozen training"))

rm_key = next(k for k in states[dslr_idx[0]] if k.startswith("features.") and k.endswith("running_mean"))
rm_diff = (states[dslr_idx[1]][rm_key] - states[dslr_idx[0]][rm_key]).abs().max().item()
print(f"  running_mean still client-specific (must be > 0): {rm_diff:.3e}")
print("  " + ("PASS" if rm_diff > 0.0 else "FAIL -- running stats were touched, they must not be"))

print("=" * 78)
print("PHASE (d): guard -- bn_affine_sync_domains without client_domains")
args = make_args()
tl, cd, testl = build(args)
try:
    ours(args, tl, testl, None, checkpoint_path=scratch + "d/d_ckpt.pt",
         bn_affine_sync_domains={"dslr"})
    print("  FAIL: expected ValueError, but ours() returned normally")
except ValueError as e:
    print(f"  PASS -- raised ValueError: {e}")

print("=" * 78)
print("Smoke test finished. Scratch dir:", scratch)

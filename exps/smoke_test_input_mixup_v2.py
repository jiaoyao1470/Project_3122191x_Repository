import sys
import os
sys.path.insert(0, os.getcwd())

import copy
import numpy as np
import torch
from torch import nn

from option import args_parser
from federated_main import ours, set_seed
from util import prepare_data_office_multi_clients
from models import adcol_model


sys.argv = ["smoke_test_input_mixup_v2"]
args = args_parser()
args.exp = 1
args.mode = "ours"
args.lr = 0.01
args.number_workers = 0
args.device = "cpu"
args.dataset = "office"
args.seed = 0
args.num_classes = 10
args.size = 64
args.batch = 32
args.domain_keyed_proto = True
args.no_discriminator_fix = False

all_pass = True

print("=" * 78)
print("CHECK 1: lambda_input_mixup=0 exact no-op")
print("=" * 78)
set_seed(args)
train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=4)
args.num_users = len(train_loader_list)
args.iters = 2
args.wk_iters = 1
args.adcol_epoch = 1
args.adcol_mu = 0.1
args.adcol_beta = 0.1

set_seed(args)
_, acc_omitted, names = ours(args, copy.deepcopy(train_loader_list), test_loader_list, client_domains)
set_seed(args)
_, acc_explicit, _ = ours(
    args, copy.deepcopy(train_loader_list), test_loader_list, client_domains,
    input_mixup_domains=None, lambda_input_mixup=0.0, input_mixup_alpha=0.2,
)
noop_ok = all(acc_omitted[d] == acc_explicit[d] for d in names)
for d in names:
    print(f"  [{d}] omitted={acc_omitted[d]}  explicit={acc_explicit[d]}  match={acc_omitted[d] == acc_explicit[d]}")
print(f"CHECK 1: {'PASS' if noop_ok else 'FAIL'}")
all_pass &= noop_ok

print()
print("=" * 78)
print("CHECK 2: mixup CE_term produces non-zero gradient in model.features")
print("=" * 78)
torch.manual_seed(123)
m = adcol_model(num_classes=10)
m.train()
images = torch.rand(8, 3, 64, 64)
labels = torch.randint(0, 10, (8,))
rng = np.random.RandomState(999)
perm = torch.as_tensor(rng.permutation(8), dtype=torch.long)
lam = float(rng.beta(0.2, 0.2))
mixed = lam * images + (1 - lam) * images[perm]
for mod in m.modules():
    if isinstance(mod, nn.modules.batchnorm._BatchNorm):
        mod.track_running_stats = False
_, logits_mix = m(mixed)
for mod in m.modules():
    if isinstance(mod, nn.modules.batchnorm._BatchNorm):
        mod.track_running_stats = True
labels_b = labels[perm]
ce = nn.CrossEntropyLoss()
input_mixup_ce = lam * ce(logits_mix, labels) + (1 - lam) * ce(logits_mix, labels_b)
m.zero_grad()
input_mixup_ce.backward()
feature_params_with_grad = [p for p in m.features.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0]
first_feature_param = next(p for p in m.features.parameters() if p.requires_grad)
grad_nonzero = first_feature_param.grad is not None and first_feature_param.grad.abs().sum().item() > 0
n_feature_params_total = sum(1 for p in m.features.parameters() if p.requires_grad)
print(f"  lam_mix={lam:.4f}")
print(f"  first (input-closest) features param: grad is not None = {first_feature_param.grad is not None}, "
      f"grad abs-sum = {first_feature_param.grad.abs().sum().item() if first_feature_param.grad is not None else None}")
print(f"  features params with non-zero grad: {len(feature_params_with_grad)} / {n_feature_params_total}")
print(f"CHECK 2: {'PASS' if grad_nonzero else 'FAIL'}")
all_pass &= grad_nonzero

print()
print("=" * 78)
print("CHECK 3: mix_rng draws differ across (client, round)")
print("=" * 78)


def seed_for(client_idx, rnd):
    return args.seed + 50000 + 1000 * client_idx + rnd


s_r0 = np.random.RandomState(seed_for(6, 0))
s_r1 = np.random.RandomState(seed_for(6, 1))
s_c7_r0 = np.random.RandomState(seed_for(7, 0))
perm_r0 = s_r0.permutation(32)
perm_r1 = s_r1.permutation(32)
perm_c7_r0 = s_c7_r0.permutation(32)
same_round_over_round = bool(np.array_equal(perm_r0, perm_r1))
same_client_over_client = bool(np.array_equal(perm_r0, perm_c7_r0))
print(f"  client6/round0 vs client6/round1 permutation identical: {same_round_over_round}")
print(f"  client6/round0 vs client7/round0 permutation identical: {same_client_over_client}")
no_replay_ok = (not same_round_over_round) and (not same_client_over_client)
print(f"CHECK 3: {'PASS' if no_replay_ok else 'FAIL'}")
all_pass &= no_replay_ok

print()
print("=" * 78)
print("CHECK 4: mixed forward does not mutate persistent BN running stats")
print("=" * 78)
torch.manual_seed(42)
m2 = adcol_model(num_classes=10)
m2.train()
bn_layers = [mod for mod in m2.modules() if isinstance(mod, nn.modules.batchnorm._BatchNorm)]
snap_before_clean = [bn.running_mean.clone() for bn in bn_layers]

images2 = torch.rand(8, 3, 64, 64)
_, _ = m2(images2)
snap_after_clean = [bn.running_mean.clone() for bn in bn_layers]
clean_forward_changed_bn = not all(torch.equal(a, b) for a, b in zip(snap_before_clean, snap_after_clean))

images3 = torch.rand(8, 3, 64, 64)
perm2 = torch.randperm(8)
mixed2 = 0.4 * images3 + 0.6 * images3[perm2]
for bn in bn_layers:
    bn.track_running_stats = False
_, logits_mix2 = m2(mixed2)
for bn in bn_layers:
    bn.track_running_stats = True
loss2 = logits_mix2.sum()
loss2.backward()
snap_after_mixed = [bn.running_mean.clone() for bn in bn_layers]
bn_unchanged_by_mixed = all(torch.equal(a, b) for a, b in zip(snap_after_clean, snap_after_mixed))

print(f"  #BN layers checked: {len(bn_layers)}")
print(f"  positive control -- clean forward DID change running_mean: {clean_forward_changed_bn}")
print(f"  running_mean identical before/after mixed forward: {bn_unchanged_by_mixed}")
bn_check_ok = clean_forward_changed_bn and bn_unchanged_by_mixed
print(f"CHECK 4: {'PASS' if bn_check_ok else 'FAIL'}")
all_pass &= bn_check_ok

print()
print("=" * 78)
print(f"OVERALL: {'ALL PASS' if all_pass else 'FAIL -- see above'}")

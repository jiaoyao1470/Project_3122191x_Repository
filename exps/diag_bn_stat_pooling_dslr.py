import sys
import os
sys.path.insert(0, os.getcwd())

import copy
import torch

from option import args_parser
from federated_main import set_seed
from models import adcol_model, Discriminator
from util import prepare_data_office_multi_clients
from update import LocalTest


if len(sys.argv) < 2:
    raise SystemExit("Usage: diag_bn_stat_pooling_dslr.py <checkpoint.pt> [weights_seed0_dir]")
_ckpt_path = sys.argv[1]
_weights_dir = sys.argv[2] if len(sys.argv) > 2 else None
sys.argv = ["diag_bn_stat_pooling_dslr"]
args = args_parser()

args.dataset = "office"
args.num_classes = 10
args.size = 64
args.batch = 32
args.number_workers = 0
args.device = args.device if torch.cuda.is_available() else "cpu"
args.seed = 0
set_seed(args)

_M = 4

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(
    args, dslr_client_override=_M
)
args.num_users = len(train_loader_list)

unique_domains = list(dict.fromkeys(client_domains))
dslr_test_loader = test_loader_list[unique_domains.index("dslr")]
dslr_idx = [i for i, d in enumerate(client_domains) if d == "dslr"]


def _n_train(loader):
    idxs = getattr(loader.sampler, "indices", None)
    return len(idxs) if idxs is not None else len(loader.dataset)


dslr_n = {i: _n_train(train_loader_list[i]) for i in dslr_idx}

print("=" * 78)
print("POST-HOC BN-STAT DIAGNOSTIC -- M=4 DSLR, experiment 'a' (evaluation-path only)")
print("client_domains = {}".format(client_domains))
print("dslr client indices = {}   n_train = {}   total = {}".format(
    dslr_idx, [dslr_n[i] for i in dslr_idx], sum(dslr_n.values())))
print("dslr test set size = {}".format(len(dslr_test_loader.dataset)))
print("=" * 78)

checkpoint = torch.load(_ckpt_path, weights_only=False, map_location=args.device)
accuracy_list = checkpoint["accuracy_list"]
final_round = checkpoint["round"]

domains_in_log = list(accuracy_list.keys())
n_rounds = len(accuracy_list["dslr"])
xdom_round = max(
    range(n_rounds),
    key=lambda r: sum(accuracy_list[d][r] for d in domains_in_log) / len(domains_in_log),
)
ref_final = accuracy_list["dslr"][final_round]
ref_xdom = accuracy_list["dslr"][xdom_round]
xdom_avg = sum(accuracy_list[d][xdom_round] for d in domains_in_log) / len(domains_in_log)
print("checkpoint round = {}   logged dslr acc @ that round = {:.4f}".format(final_round, ref_final))
print("X-dom-best round = {}   logged dslr acc @ that round = {:.4f}   (four-domain avg = {:.4f})".format(
    xdom_round, ref_xdom, xdom_avg))
print()

_disc_state = checkpoint["discriminator_state"]
_last_w = [v for k, v in _disc_state.items() if k.endswith("weight") and v.dim() == 2][-1]
num_domains = _last_w.shape[0]
print("discriminator output width in checkpoint = {} ({}-indexed arm)".format(
    num_domains, "client" if num_domains == args.num_users else "domain"))
discriminator = Discriminator(
    adcol_model(num_classes=args.num_classes), num_domains
).to(args.device)
discriminator.load_state_dict(_disc_state)


def load_states(source):
    if source == "final":
        return {i: copy.deepcopy(checkpoint["local_models_state"][i]) for i in dslr_idx}
    states = {}
    for i in dslr_idx:
        p = os.path.join(source, "best_local_model_client{}_{}.pth".format(i, client_domains[i]))
        if not os.path.exists(p):
            return None
        states[i] = torch.load(p, weights_only=False, map_location=args.device)
    return states


def eval_states(states):
    accs = {}
    for i, sd in states.items():
        model = adcol_model(num_classes=args.num_classes).to(args.device)
        model.load_state_dict(sd)
        with torch.no_grad():
            _, _, acc = LocalTest(args=args).test_inference_ours(
                args, model, dslr_test_loader, copy.deepcopy(discriminator),
                num_domains=num_domains,
            )
        accs[i] = acc
    return accs, sum(accs.values()) / len(accs)


def bn_stat_keys(state):
    return [k for k in state if k.startswith("features.") and k.endswith(".running_mean")]


def pool_bn_stats(states):
    pooled = {}
    N = float(sum(dslr_n[i] for i in states))
    ref = next(iter(states.values()))
    for mkey in bn_stat_keys(ref):
        vkey = mkey.replace(".running_mean", ".running_var")
        mu = sum(dslr_n[i] * states[i][mkey].double() for i in states) / N
        ex2 = sum(
            dslr_n[i] * (states[i][vkey].double() + states[i][mkey].double() ** 2)
            for i in states
        ) / N
        var = torch.clamp(ex2 - mu ** 2, min=0.0)
        pooled[mkey] = mu.to(ref[mkey].dtype)
        pooled[vkey] = var.to(ref[vkey].dtype)
    out = {}
    for i, sd in states.items():
        new = copy.deepcopy(sd)
        for k, v in pooled.items():
            new[k] = v.clone()
        out[i] = new
    return out, len(bn_stat_keys(ref))


def mismatch_report(states):
    ref = next(iter(states.values()))
    m_rel, v_rel = [], []
    for mkey in bn_stat_keys(ref):
        vkey = mkey.replace(".running_mean", ".running_var")
        mus = torch.stack([states[i][mkey].double() for i in states])
        vs = torch.stack([states[i][vkey].double() for i in states])
        m_rel.append((mus.std(dim=0) / (mus.abs().mean(dim=0) + 1e-8)).mean().item())
        v_rel.append((vs.std(dim=0) / (vs.abs().mean(dim=0) + 1e-8)).mean().item())
    return sum(m_rel) / len(m_rel), sum(v_rel) / len(v_rel)


def run(label, states, reference):
    print("-" * 78)
    print("[{}]".format(label))
    m_spread, v_spread = mismatch_report(states)
    print("  pre-pooling cross-client spread (mean over BN layers of per-channel std/|mean|): "
          "running_mean {:.4f}   running_var {:.4f}".format(m_spread, v_spread))

    base_accs, base_mean = eval_states(states)
    print("  no pooling : per-client {}   domain mean = {:.4f}".format(
        ["{:.4f}".format(base_accs[i]) for i in dslr_idx], base_mean))
    diff = base_mean - reference
    verdict = "MATCH" if abs(diff) < 1e-6 else "MISMATCH -- investigate before trusting the delta"
    print("  sanity vs logged accuracy_list: reference {:.4f}   diff {:+.6f}   {}".format(
        reference, diff, verdict))

    pooled_states, n_bn = pool_bn_stats(states)
    pool_accs, pool_mean = eval_states(pooled_states)
    print("  pooled BN  : per-client {}   domain mean = {:.4f}   ({} BN layers pooled)".format(
        ["{:.4f}".format(pool_accs[i]) for i in dslr_idx], pool_mean, n_bn))
    print("  DELTA      : {:+.2f} pp".format(100 * (pool_mean - base_mean)))
    return base_mean, pool_mean


results = {}
results["final r{}".format(final_round)] = run(
    "final round r{}".format(final_round), load_states("final"), ref_final)

if _weights_dir is not None:
    xdom_states = load_states(_weights_dir)
    if xdom_states is None:
        print("-" * 78)
        print("[X-dom-best r{}] SKIPPED -- weights dir given but the per-client .pth files were not "
              "all found under {}".format(xdom_round, _weights_dir))
    else:
        results["X-dom-best r{}".format(xdom_round)] = run(
            "X-dom-best round r{}".format(xdom_round), xdom_states, ref_xdom)
else:
    print("-" * 78)
    print("[X-dom-best r{}] NOT TESTED -- no weights_seed0 dir supplied. Only the final state was "
          "diagnosed; this is a single-representative-state result, not a sweep over representative "
          "states.".format(xdom_round))

print("=" * 78)
for k, (b, p) in results.items():
    print("{:<16s}  no-pooling {:.4f}  ->  pooled {:.4f}   ({:+.2f} pp)".format(
        k, b, p, 100 * (p - b)))
print("Interpretation bound: a null result licenses only 'BN running-stat / evaluation mismatch is "
      "not the main explanation'. It does not bear on encoder-space incompatibility.")
print("=" * 78)

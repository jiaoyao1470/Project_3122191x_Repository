import sys
import os
sys.path.insert(0, os.getcwd())

from collections import Counter

from option import args_parser
from federated_main import set_seed
from util import prepare_data_office_multi_clients


sys.argv = ["check_k64_class_feasibility"]
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
set_seed(args)

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=4)

DSLR_CLIENTS = [i for i, d in enumerate(client_domains) if d == "dslr"]
CLASS_NAMES = ["back_pack", "bike", "calculator", "headphones", "keyboard",
               "laptop_computer", "monitor", "mouse", "mug", "projector"]

domain_counts = Counter()
client_counts = {}
for i in DSLR_CLIENTS:
    loader = train_loader_list[i]
    labels_all = loader.dataset.labels
    idx = loader.sampler.indices
    c = Counter(labels_all[j] for j in idx)
    client_counts[i] = c
    domain_counts.update(c)

print("=" * 100)
print(f"DSLR domain-wide per-class totals (n=126 total, {len(DSLR_CLIENTS)} clients: {DSLR_CLIENTS})")
print("=" * 100)
for cls in range(args.num_classes):
    print(f"  class {cls} ({CLASS_NAMES[cls]:<16}): domain total = {domain_counts.get(cls, 0)}")
print(f"  TOTAL = {sum(domain_counts.values())}")

print()
print("=" * 100)
print("Per-client K64 feasibility (target = 2x local count per already-present class, capped at "
      "domain total; classes NOT present in this client stay absent -- C_i^K64 = C_i^K32)")
print("=" * 100)

summary_rows = []
for i in DSLR_CLIENTS:
    n_local_total = sum(client_counts[i].values())
    target_total = 0
    achievable_total = 0
    capped_classes = []
    print(f"\nclient {i} (own partition size = {n_local_total}):")
    print(f"  {'class':<18} {'n_local':>8} {'n_domain':>9} {'peer_avail':>11} {'target(2x)':>11} {'achievable':>11} {'capped?':>8}")
    for cls, n_local in sorted(client_counts[i].items()):
        n_dom = domain_counts[cls]
        peer_avail = n_dom - n_local
        target = 2 * n_local
        achievable = min(target, n_dom)
        capped = achievable < target
        if capped:
            capped_classes.append(CLASS_NAMES[cls])
        target_total += target
        achievable_total += achievable
        print(f"  {CLASS_NAMES[cls]:<18} {n_local:>8} {n_dom:>9} {peer_avail:>11} {target:>11} "
              f"{achievable:>11} {'YES' if capped else '':>8}")
    print(f"  -> K32={n_local_total}  target K64={target_total}  achievable K_i={achievable_total}"
          f"  (capped classes: {capped_classes if capped_classes else 'none'})")
    summary_rows.append((i, n_local_total, target_total, achievable_total, capped_classes))

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
for i, k32, target, achievable, capped in summary_rows:
    status = "OK, target fully reachable" if achievable == target else f"CAPPED -- real K_i={achievable}, not {target}"
    print(f"  client {i}: K32={k32:>3}  target_K64={target:>3}  achievable_K_i={achievable:>3}   {status}")

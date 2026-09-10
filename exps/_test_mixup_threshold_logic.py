import sys
sys.argv = ["_test_mixup_threshold_logic"]

import numpy as np

from option import args_parser
from federated_main import set_seed
from util import prepare_data_office_multi_clients

args = args_parser()
args.exp = 1
args.mode = "ours"
args.dataset = "office"
args.seed = 0
args.num_classes = 10
args.size = 64
args.batch = 32
args.device = "cpu"
set_seed(args)

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args)


def original_logic(labels, num_classes):
    classes_present = set(np.unique(labels).tolist())
    return set(range(num_classes)) - classes_present


def new_logic(labels, num_classes, threshold):
    class_counts = np.bincount(labels, minlength=num_classes)
    return {c for c in range(num_classes) if class_counts[c] <= threshold}


per_client_labels = []
for client_idx in range(len(train_loader_list)):
    idx = train_loader_list[client_idx].sampler.indices
    per_client_labels.append(np.asarray(train_loader_list[client_idx].dataset.labels)[idx])

for client_idx, labels in enumerate(per_client_labels):
    old = original_logic(labels, args.num_classes)
    new = new_logic(labels, args.num_classes, 0)
    assert old == new, (
        f"TEST 1 FAIL: client{client_idx} threshold=0 mismatch -- original={sorted(old)}, "
        f"new={sorted(new)}"
    )
print(f"TEST 1 PASS: threshold=0 reproduces the original n=0 logic exactly on all "
      f"{len(per_client_labels)} clients (backward compatible)")

for client_idx, labels in enumerate(per_client_labels):
    prev = new_logic(labels, args.num_classes, 0)
    for thr in [1, 2, 3, 5, 8]:
        cur = new_logic(labels, args.num_classes, thr)
        assert prev.issubset(cur), (
            f"TEST 2 FAIL: client{client_idx} threshold={thr} set is not a superset of the "
            f"threshold={thr-1 if thr <= 3 else 'lower'} set"
        )
        prev = cur
print("TEST 2 PASS: the triggering class set grows monotonically with threshold on every client")

print("\nActually-firing clients per threshold (update.py needs >= 2 triggering classes per client):")
for thr in [0, 1, 2, 3, 5, 8]:
    firing = []
    for client_idx, labels in enumerate(per_client_labels):
        n = len(new_logic(labels, args.num_classes, thr))
        if n >= 2:
            firing.append(f"c{client_idx}({client_domains[client_idx]},{n})")
    print(f"  threshold={thr}: {len(firing)} client(s) fire -> {firing}")

firing_at_0 = [
    client_idx for client_idx, labels in enumerate(per_client_labels)
    if len(new_logic(labels, args.num_classes, 0)) >= 2
]
assert len(firing_at_0) == 1, (
    f"TEST 3 FAIL: expected exactly 1 client to fire at threshold=0 (the documented situation), "
    f"got {firing_at_0}"
)
assert client_domains[firing_at_0[0]] == "dslr", "TEST 3 FAIL: the single firing client should be a dslr client"
print(f"\nTEST 3 PASS: at threshold=0 exactly ONE client (client{firing_at_0[0]}, dslr) actually fires "
      f"-- confirming the original mechanism was active on only 2 of dslr's 40 (client, class) cells")

for thr in [0, 1, 2, 3, 5, 8]:
    for client_idx, labels in enumerate(per_client_labels):
        if client_domains[client_idx] == "dslr":
            continue
        n = len(new_logic(labels, args.num_classes, thr))
        assert n == 0, (
            f"TEST 4 FAIL: {client_domains[client_idx]} client{client_idx} triggers {n} classes at "
            f"threshold={thr} -- the self-targeting claim (dslr only, up to thr=8) does not hold"
        )
print("TEST 4 PASS: no caltech/amazon/webcam client triggers a single class at any threshold up to 8 "
      "-- the rule is self-targeting to dslr with no domain-specific hardcoding")

print("\nALL MIXUP-THRESHOLD LOGIC TESTS PASSED (4/4)")

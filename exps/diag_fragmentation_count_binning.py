import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
from collections import Counter, defaultdict

from option import args_parser
from federated_main import set_seed
from models import adcol_model
from util import prepare_data_office_multi_clients


if len(sys.argv) < 4 or sys.argv[1] not in ("1", "2", "4") or sys.argv[2] not in ("baseline", "D"):
    raise SystemExit("Usage: diag_fragmentation_count_binning.py <1|2|4> <baseline|D> <path/to/checkpoint.pt>")
_M = int(sys.argv[1])
_condition = sys.argv[2]
_checkpoint_path_arg = sys.argv[3]
sys.argv = ["diag_fragmentation_count_binning"]
args = args_parser()

args.dataset = "office"
args.num_classes = 10
args.size = 64
args.batch = 32
args.number_workers = 0
args.device = args.device if __import__("torch").cuda.is_available() else "cpu"
args.seed = 0
set_seed(args)

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=_M)
unique_domains = list(dict.fromkeys(client_domains))
dslr_test_loader = test_loader_list[unique_domains.index('dslr')]
dslr_client_indices = [i for i, d in enumerate(client_domains) if d == 'dslr']

checkpoint = torch.load(_checkpoint_path_arg, weights_only=False, map_location=args.device)
local_models_state = checkpoint["local_models_state"]
print(f"Loaded checkpoint from round {checkpoint['round']} (M={_M}, condition={_condition})")


def eval_model(model, loader):
    model.eval()
    class_correct = Counter()
    class_total = Counter()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(args.device).float(), labels.to(args.device).long()
            _, logits = model(images)
            pred = logits.max(1)[1]
            for c, p in zip(labels.tolist(), pred.tolist()):
                class_total[c] += 1
                if c == p:
                    class_correct[c] += 1
    return class_correct, class_total


def bin_for(n):
    if n == 0:
        return "n=0"
    elif n <= 2:
        return "n=1-2"
    elif n <= 5:
        return "n=3-5"
    else:
        return "n>5"


bin_correct = defaultdict(int)
bin_total = defaultdict(int)
bin_cells = defaultdict(int)

for ci in dslr_client_indices:
    model = adcol_model(args.num_classes).to(args.device)
    model.load_state_dict(local_models_state[ci])

    train_idx = train_loader_list[ci].sampler.indices
    train_labels = np.array(train_loader_list[ci].dataset.labels)[train_idx]
    train_counts = Counter(train_labels.tolist())

    class_correct, class_total = eval_model(model, dslr_test_loader)

    for c in range(10):
        n_train_c = train_counts.get(c, 0)
        b = bin_for(n_train_c)
        bin_correct[b] += class_correct.get(c, 0)
        bin_total[b] += class_total.get(c, 0)
        bin_cells[b] += 1

print(f"\nM={_M}, condition={_condition} -- test accuracy by n_train bin (pooled across all dslr clients/classes)")
print(f"{'bin':10s}{'n_cells':10s}{'correct/total':16s}{'acc':10s}")
for b in ["n=0", "n=1-2", "n=3-5", "n>5"]:
    if bin_cells[b] == 0:
        print(f"{b:10s}{'0':10s}{'(no cells)':16s}")
        continue
    acc = bin_correct[b] / bin_total[b] if bin_total[b] > 0 else float('nan')
    print(f"{b:10s}{bin_cells[b]:<10d}{f'{bin_correct[b]}/{bin_total[b]}':16s}{acc*100:.1f}%")

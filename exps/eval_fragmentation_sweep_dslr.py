import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
from collections import Counter

from option import args_parser
from federated_main import set_seed
from models import adcol_model
from util import prepare_data_office_multi_clients


if len(sys.argv) < 4 or sys.argv[1] not in ("1", "2", "4") or sys.argv[2] not in ("baseline", "D"):
    raise SystemExit("Usage: eval_fragmentation_sweep_dslr.py <1|2|4> <baseline|D> <path/to/checkpoint.pt>")
_M = int(sys.argv[1])
_condition = sys.argv[2]
_checkpoint_path_arg = sys.argv[3]
sys.argv = ["eval_fragmentation_sweep_dslr"]
args = args_parser()

args.dataset = "office"
args.num_classes = 10
args.size = 64
args.batch = 32
args.number_workers = 0
args.device = args.device if __import__("torch").cuda.is_available() else "cpu"
args.seed = 0
set_seed(args)

LABEL_NAMES = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard',
               'laptop_computer', 'monitor', 'mouse', 'mug', 'projector']

train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=_M)

unique_domains = list(dict.fromkeys(client_domains))
dslr_test_loader = test_loader_list[unique_domains.index('dslr')]
dslr_client_indices = [i for i, d in enumerate(client_domains) if d == 'dslr']

checkpoint = torch.load(_checkpoint_path_arg, weights_only=False, map_location=args.device)
local_models_state = checkpoint["local_models_state"]
loaded_round = checkpoint["round"]
print(f"Loaded checkpoint from round {loaded_round} (M={_M}, condition={_condition}, path={_checkpoint_path_arg})")
EXPECTED_FINAL_ROUND = 99
if loaded_round != EXPECTED_FINAL_ROUND:
    print(f"WARNING: expected final round {EXPECTED_FINAL_ROUND}, got {loaded_round} -- this checkpoint "
          f"is NOT the final-round snapshot. The headline trend number (from accuracy_list at round "
          f"{EXPECTED_FINAL_ROUND}) and this diagnostic would then be reading DIFFERENT snapshots -- "
          f"do not treat them as mechanistically corresponding until this is resolved.")


def eval_model(model, loader):
    model.eval()
    class_correct = Counter()
    class_total = Counter()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(args.device).float(), labels.to(args.device).long()
            _, logits = model(images)
            pred = logits.max(1)[1]
            for c, p in zip(labels.tolist(), pred.tolist()):
                class_total[c] += 1
                if c == p:
                    class_correct[c] += 1
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    per_class_acc = {c: (class_correct[c] / class_total[c] if class_total[c] > 0 else float('nan')) for c in range(10)}
    per_class_counts = {c: (class_correct[c], class_total[c]) for c in range(10)}
    return correct / total, per_class_acc, per_class_counts


print(f"\n{'='*90}\nM={_M}, condition={_condition}\n{'='*90}")
for rank, ci in enumerate(dslr_client_indices):
    model = adcol_model(args.num_classes).to(args.device)
    model.load_state_dict(local_models_state[ci])

    train_idx = train_loader_list[ci].sampler.indices
    train_labels = np.array(train_loader_list[ci].dataset.labels)[train_idx]
    train_counts = Counter(train_labels.tolist())

    train_acc, train_per_class, train_counts_ct = eval_model(model, train_loader_list[ci])
    test_acc, test_per_class, test_counts_ct = eval_model(model, dslr_test_loader)

    print(f"\ndslr client {rank} (n_train={len(train_idx)}): "
          f"local train acc={train_acc*100:.2f}%, test acc={test_acc*100:.2f}%")
    print(f"  {'class':16s}{'n_train(this client)':22s}{'train acc':16s}{'test acc':16s}")
    for c in range(10):
        n_train_c = train_counts.get(c, 0)
        tr_a = train_per_class[c]
        te_a = test_per_class[c]
        tr_correct, tr_total = train_counts_ct[c]
        te_correct, te_total = test_counts_ct[c]
        flag = "  <-- n=0 in this client's training data" if n_train_c == 0 else ""
        tr_str = f"{tr_a*100:.1f}%({tr_correct}/{tr_total})" if not np.isnan(tr_a) else "n/a"
        te_str = f"{te_a*100:.1f}%({te_correct}/{te_total})" if not np.isnan(te_a) else "n/a"
        print(f"  {LABEL_NAMES[c]:16s}{n_train_c:<22d}{tr_str:16s}{te_str:16s}{flag}")

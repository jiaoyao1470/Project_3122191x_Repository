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


if len(sys.argv) < 3:
    raise SystemExit("Usage: eval_component_ablation_dslr.py <condition_label> <path/to/checkpoint.pt>")
_condition = sys.argv[1]
_checkpoint_path_arg = sys.argv[2]
_M = 2
sys.argv = ["eval_component_ablation_dslr"]
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


def eval_model_with_preds(model, loader):
    model.eval()
    class_correct = Counter()
    class_total = Counter()
    pred_counts = Counter()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(args.device).float(), labels.to(args.device).long()
            _, logits = model(images)
            pred = logits.max(1)[1]
            for c, p in zip(labels.tolist(), pred.tolist()):
                class_total[c] += 1
                pred_counts[p] += 1
                if c == p:
                    class_correct[c] += 1
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    per_class_acc = {c: (class_correct[c] / class_total[c] if class_total[c] > 0 else float('nan')) for c in range(10)}
    per_class_counts = {c: (class_correct[c], class_total[c]) for c in range(10)}
    return correct / total, per_class_acc, per_class_counts, pred_counts, total


print(f"\n{'='*90}\nM={_M}, condition={_condition}\n{'='*90}")

domain_pred_counts = Counter()
domain_total = 0
domain_correct = 0

for rank, ci in enumerate(dslr_client_indices):
    model = adcol_model(args.num_classes).to(args.device)
    model.load_state_dict(local_models_state[ci])

    train_idx = train_loader_list[ci].sampler.indices
    train_labels = np.array(train_loader_list[ci].dataset.labels)[train_idx]
    train_counts = Counter(train_labels.tolist())

    test_acc, test_per_class, test_counts_ct, pred_counts, total = eval_model_with_preds(model, dslr_test_loader)
    domain_pred_counts.update(pred_counts)
    domain_total += total
    domain_correct += sum(v[0] for v in test_counts_ct.values())

    max_pred_class, max_pred_n = pred_counts.most_common(1)[0]
    max_pred_share = max_pred_n / total

    print(f"\ndslr client {rank} (n_train={len(train_idx)}): test acc={test_acc*100:.2f}%  "
          f"max_pred_class_share={max_pred_share:.4f} (class={LABEL_NAMES[max_pred_class]}, {max_pred_n}/{total})"
          + ("  <-- COLLAPSE?" if max_pred_share > 0.5 else ""))
    print(f"  {'class':16s}{'n_train(this client)':22s}{'test acc':16s}{'predicted-as this class':24s}")
    for c in range(10):
        n_train_c = train_counts.get(c, 0)
        te_a = test_per_class[c]
        te_correct, te_total = test_counts_ct[c]
        flag = "  <-- n=0 in this client's training data" if n_train_c == 0 else ""
        te_str = f"{te_a*100:.1f}%({te_correct}/{te_total})" if not np.isnan(te_a) else "n/a"
        print(f"  {LABEL_NAMES[c]:16s}{n_train_c:<22d}{te_str:16s}{pred_counts.get(c, 0):<24d}{flag}")

domain_max_class, domain_max_n = domain_pred_counts.most_common(1)[0]
domain_max_share = domain_max_n / domain_total
print(f"\n--- domain-pooled (both clients' predictions on the shared dslr test set) ---")
print(f"pooled acc={domain_correct/domain_total*100:.2f}%  max_pred_class_share={domain_max_share:.4f} "
      f"(class={LABEL_NAMES[domain_max_class]})" + ("  <-- COLLAPSE?" if domain_max_share > 0.5 else ""))

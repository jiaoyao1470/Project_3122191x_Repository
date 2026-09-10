import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter

from option import args_parser
from federated_main import set_seed
from models import adcol_model
from util import prepare_data_office_multi_clients


if len(sys.argv) < 4:
    raise SystemExit("Usage: diag_head_vs_proto_dslr.py <condition_label> <checkpoint_path> <prototype_tensor_path>")
_condition = sys.argv[1]
_checkpoint_path_arg = sys.argv[2]
_proto_tensor_path = sys.argv[3]
_M = 2
sys.argv = ["diag_head_vs_proto_dslr"]
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
proto_data = torch.load(_proto_tensor_path, weights_only=False, map_location=args.device)
p_mk = proto_data["p_mk"]

print(f"\n{'='*80}\ncondition={_condition}  (checkpoint round={checkpoint['round']}, proto round={proto_data['round']})\n{'='*80}")

for ci in dslr_client_indices:
    model = adcol_model(args.num_classes).to(args.device)
    model.load_state_dict(checkpoint["local_models_state"][ci])
    model.eval()

    proto_classes = sorted(p_mk[ci].keys())
    proto_bank = torch.stack([F.normalize(p_mk[ci][c], dim=0) for c in proto_classes])

    correct_head, correct_proto, agree, total = 0, 0, 0, 0
    with torch.no_grad():
        for images, labels in dslr_test_loader:
            images, labels = images.to(args.device).float(), labels.to(args.device).long()
            rep, logits = model(images)
            pred_head = logits.max(1)[1]

            rep_norm = F.normalize(rep, dim=1)
            sims = rep_norm @ proto_bank.T
            pred_proto_idx = sims.max(1)[1]
            pred_proto = torch.tensor([proto_classes[i] for i in pred_proto_idx.tolist()], device=args.device)

            correct_head += pred_head.eq(labels).sum().item()
            correct_proto += pred_proto.eq(labels).sum().item()
            agree += pred_head.eq(pred_proto).sum().item()
            total += labels.size(0)

    print(f"dslr client {ci}: Acc_head={correct_head/total*100:6.2f}%   Acc_proto={correct_proto/total*100:6.2f}%   "
          f"head-proto agreement={agree/total*100:6.2f}%   (n_test={total})")

import sys
import os
sys.path.insert(0, os.getcwd())

import torch
from option import args_parser
from federated_main import set_seed
from models import adcol_model


if len(sys.argv) < 4:
    raise SystemExit("Usage: diag_head_on_prototype_dslr.py <condition_label> <checkpoint_path> <prototype_tensor_path>")
_condition = sys.argv[1]
_checkpoint_path_arg = sys.argv[2]
_proto_tensor_path = sys.argv[3]
sys.argv = ["diag_head_on_prototype_dslr"]
args = args_parser()
args.device = args.device if __import__("torch").cuda.is_available() else "cpu"
args.seed = 0
set_seed(args)

checkpoint = torch.load(_checkpoint_path_arg, weights_only=False, map_location=args.device)
proto_data = torch.load(_proto_tensor_path, weights_only=False, map_location=args.device)
p_mk = proto_data["p_mk"]

print(f"\n{'='*80}\ncondition={_condition}  (checkpoint round={checkpoint['round']}, proto round={proto_data['round']})\n{'='*80}")

for ci in sorted(p_mk.keys()):
    model = adcol_model(10).to(args.device)
    model.load_state_dict(checkpoint["local_models_state"][ci])
    model.eval()

    correct = 0
    margins = []
    rows = []
    with torch.no_grad():
        for c in sorted(p_mk[ci].keys()):
            p = p_mk[ci][c].to(args.device).unsqueeze(0)
            z = model.classifier(p).squeeze(0)
            pred = z.argmax().item()
            margin = (z[c] - torch.cat([z[:c], z[c+1:]]).max()).item()
            margins.append(margin)
            rows.append((c, pred, margin))
            if pred == c:
                correct += 1

    n = len(p_mk[ci])
    print(f"\ndslr client {ci}: head-on-prototype acc = {correct}/{n} = {correct/n*100:.2f}%   "
          f"mean margin = {sum(margins)/n:.4f}   median margin = {sorted(margins)[n//2]:.4f}")
    for c, pred, m in rows:
        flag = "" if pred == c else f"  <-- MISCLASSIFIED as class {pred}"
        print(f"    class {c}: margin={m:+.4f}{flag}")

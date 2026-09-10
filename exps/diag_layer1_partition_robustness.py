import sys, os
sys.path.insert(0, os.getcwd())
import copy
import numpy as np
import torch
import torch.nn.functional as F

from option import args_parser
from federated_main import set_seed
import data_utils
from torchvision import transforms
from models import adcol_model
from util import get_gpcl_domain_keyed, confidence_weight


sys.argv = ["diag_layer1_partition_robustness"]
args = args_parser()
args.num_classes = 10
args.size = 64
args.batch = 32
args.seed = 0
set_seed(args)
DEVICE = "cpu"

LABEL_DICT = {'back_pack': 0, 'bike': 1, 'calculator': 2, 'headphones': 3, 'keyboard': 4,
              'laptop_computer': 5, 'monitor': 6, 'mouse': 7, 'mug': 8, 'projector': 9}
DATA_BASE = "../data/office_caltech_10"
FIXED_MODEL_PATH = "D:/Dissertation/所有运行结果/l2_gpcl/weights_seed0/best_local_model_client0_caltech.pth"
M_VALUES = [2, 3, 4, 5]
DOMAINS = ["dslr", "caltech", "amazon"]

transform_eval = transforms.Compose([
    transforms.Resize([args.size, args.size]),
    transforms.ToTensor(),
])

print("Loading fixed feature extractor (one already-trained client model, held constant throughout)...")
fixed_model = adcol_model(num_classes=args.num_classes)
fixed_model.load_state_dict(torch.load(FIXED_MODEL_PATH, map_location="cpu", weights_only=False))
fixed_model.eval()


@torch.no_grad()
def extract_features_labels(dataset, indices=None):
    if indices is None:
        indices = np.arange(len(dataset.labels))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch,
        sampler=torch.utils.data.SubsetRandomSampler(indices) if indices is not None else None,
        shuffle=False, num_workers=0)
    feats, labels = [], []
    for images, lbls in loader:
        images = images.float()
        rep, _ = fixed_model(images)
        feats.append(rep)
        labels.append(lbls)
    return torch.cat(feats), torch.cat(labels)


def per_class_proto_and_count(feats, labels, num_classes):
    protos, counts = {}, {}
    for c in range(num_classes):
        mask = labels == c
        n = int(mask.sum().item())
        if n > 0:
            protos[c] = feats[mask].mean(dim=0)
            counts[c] = n
    return protos, counts


def combine_uniform(protos_list):
    stacked = torch.stack([F.normalize(p, dim=0) for p in protos_list], dim=0)
    return stacked.mean(dim=0)


def combine_count_weighted(protos_list, counts_list):
    total = sum(counts_list)
    acc = torch.zeros_like(F.normalize(protos_list[0], dim=0))
    for p, n in zip(protos_list, counts_list):
        acc += F.normalize(p, dim=0) * n
    return acc / total


def combine_distance_gpcl(protos_list):
    stacked = torch.stack([F.normalize(p, dim=0) for p in protos_list], dim=0)
    if len(protos_list) == 1:
        return stacked[0]
    mu = stacked.mean(dim=0)
    dists = ((stacked - mu) ** 2).sum(dim=1)
    d_total = dists.sum()
    if d_total.item() <= 1e-12:
        return mu
    weights = dists / d_total
    return (weights.unsqueeze(1) * stacked).sum(dim=0)


def combine_confidence(protos_list, counts_list, c=5.0):
    acc = torch.zeros_like(F.normalize(protos_list[0], dim=0))
    total_w = 0.0
    for p, n in zip(protos_list, counts_list):
        w = confidence_weight(n, c)
        acc += F.normalize(p, dim=0) * w
        total_w += w
    return acc / total_w if total_w > 1e-12 else acc


METHODS = {
    "uniform": lambda protos, counts: combine_uniform(protos),
    "count_weighted": lambda protos, counts: combine_count_weighted(protos, counts),
    "distance_gpcl": lambda protos, counts: combine_distance_gpcl(protos),
    "confidence": lambda protos, counts: combine_confidence(protos, counts),
}

results = {}

for domain in DOMAINS:
    print(f"\n{'='*20} domain = {domain} {'='*20}")
    train_set = data_utils.OfficeDataset(DATA_BASE, domain, transform=transform_eval)
    all_idx = np.arange(len(train_set.labels))
    print(f"  total samples: {len(all_idx)}")

    oracle_feats, oracle_labels = extract_features_labels(train_set, all_idx)
    oracle_protos, oracle_counts = per_class_proto_and_count(oracle_feats, oracle_labels, args.num_classes)
    oracle_protos_n = {c: F.normalize(p, dim=0) for c, p in oracle_protos.items()}
    print(f"  oracle per-class counts: {oracle_counts}")

    for M in M_VALUES:
        rng = np.random.RandomState(42 + M)
        shuffled = rng.permutation(all_idx)
        groups = np.array_split(shuffled, M)

        group_protos, group_counts = [], []
        for g_idx in groups:
            feats, labels = extract_features_labels(train_set, g_idx)
            protos, counts = per_class_proto_and_count(feats, labels, args.num_classes)
            group_protos.append(protos)
            group_counts.append(counts)

        for method_name, combine_fn in METHODS.items():
            combined = {}
            for c in range(args.num_classes):
                protos_c = [group_protos[g][c] for g in range(M) if c in group_protos[g]]
                counts_c = [group_counts[g][c] for g in range(M) if c in group_counts[g]]
                if len(protos_c) == 0:
                    continue
                combined[c] = combine_fn(protos_c, counts_c)
            results[(domain, method_name, M)] = combined

    print(f"\n  -- domain recovery fidelity (mean cosine sim to oracle, averaged over classes+M) --")
    for method_name in METHODS:
        sims_by_M = {}
        for M in M_VALUES:
            combined = results[(domain, method_name, M)]
            sims = []
            for c, p in combined.items():
                if c in oracle_protos_n:
                    sims.append(F.cosine_similarity(F.normalize(p, dim=0), oracle_protos_n[c], dim=0).item())
            sims_by_M[M] = np.mean(sims)
        overall = np.mean(list(sims_by_M.values()))
        print(f"    {method_name:16s} M=2:{sims_by_M[2]:.4f}  M=3:{sims_by_M[3]:.4f}  "
              f"M=4:{sims_by_M[4]:.4f}  M=5:{sims_by_M[5]:.4f}   overall_mean:{overall:.4f}")

    print(f"\n  -- partition stability (mean pairwise cosine sim across M=2/3/4/5, averaged over classes) --")
    for method_name in METHODS:
        pairwise_sims = []
        for i in range(len(M_VALUES)):
            for j in range(i + 1, len(M_VALUES)):
                M1, M2 = M_VALUES[i], M_VALUES[j]
                c1, c2 = results[(domain, method_name, M1)], results[(domain, method_name, M2)]
                common = set(c1.keys()) & set(c2.keys())
                for c in common:
                    pairwise_sims.append(F.cosine_similarity(
                        F.normalize(c1[c], dim=0), F.normalize(c2[c], dim=0), dim=0).item())
        print(f"    {method_name:16s} mean_pairwise_cosine_sim:{np.mean(pairwise_sims):.4f}  "
              f"std:{np.std(pairwise_sims):.4f}")

print("\n=== done ===")

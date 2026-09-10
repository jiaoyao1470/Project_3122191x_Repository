import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
from collections import Counter

from option import args_parser
from federated_main import set_seed
from util import prepare_data_office_multi_clients


sys.argv = ["check_dslr_fragmentation_sample_counts"]
args = args_parser()
args.dataset = "office"
args.num_classes = 10
args.size = 64
args.batch = 32
args.number_workers = 0
args.seed = 0
set_seed(args)

LABEL_NAMES = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard',
               'laptop_computer', 'monitor', 'mouse', 'mug', 'projector']

for M in [1, 2, 4]:
    print(f"\n{'='*70}\nM (dslr client count) = {M}\n{'='*70}")
    train_loader_list, client_domains, test_loader_list = prepare_data_office_multi_clients(args, dslr_client_override=M)

    domain_client_counts = Counter(client_domains)
    print(f"client counts per domain: {dict(domain_client_counts)}")
    assert domain_client_counts['caltech'] == 3, f"caltech client count changed to {domain_client_counts['caltech']}!"
    assert domain_client_counts['amazon'] == 2, f"amazon client count changed to {domain_client_counts['amazon']}!"
    assert domain_client_counts['webcam'] == 1, f"webcam client count changed to {domain_client_counts['webcam']}!"
    assert domain_client_counts['dslr'] == M, f"dslr client count is {domain_client_counts['dslr']}, expected {M}!"

    dslr_client_indices = [i for i, d in enumerate(client_domains) if d == 'dslr']
    dslr_total = 0
    for rank, ci in enumerate(dslr_client_indices):
        loader = train_loader_list[ci]
        idx = loader.sampler.indices
        labels = np.array(loader.dataset.labels)[idx]
        counts = Counter(labels.tolist())
        total = len(idx)
        dslr_total += total
        n_present = sum(1 for c in range(10) if counts.get(c, 0) > 0)
        per_class_str = ", ".join(f"{LABEL_NAMES[c]}={counts.get(c, 0)}" for c in range(10))
        print(f"  dslr client {rank} (total={total}, classes_present={n_present}/10):")
        print(f"    {per_class_str}")
    print(f"  dslr total across {M} client(s): {dslr_total}")

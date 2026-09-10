import math
import copy
import torch
from torchvision import datasets, transforms
import numpy as np
import data_utils
import seaborn as sns
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import pickle
from torch.optim import Optimizer
import torch.nn.functional as F
from collections import Counter
import functools

def proto_aggregation(local_protos_list, num_list):
    agg_protos_label = dict()
    total_num_list = functools.reduce(lambda x, y: x + y, num_list)
    for idx in range(len(local_protos_list)):
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]
    averaged_protos = {}
    for label, proto_list in agg_protos_label.items():
        for idx, proto in enumerate(proto_list):
            if label not in averaged_protos:
                averaged_protos[label] = proto * num_list[idx][label] / total_num_list[label]
            else:
                averaged_protos[label] += proto * num_list[idx][label] / total_num_list[label]
    return averaged_protos

def get_mean(args, proto_list, nums):
    weighted_protos = {}
    total_counts = {}
    for idx, proto in enumerate(proto_list):
        for cls, p in proto.items():
            if cls not in weighted_protos:
                weighted_protos[cls] = torch.zeros_like(p)
                total_counts[cls] = 0
            weighted_protos[cls] += F.normalize(p, dim=0) * nums[idx][cls]
            total_counts[cls] += nums[idx][cls]
    global_protos = {}
    for cls in weighted_protos.keys():
        if total_counts[cls] > 0:
            global_protos[cls] = weighted_protos[cls] / total_counts[cls]
    return global_protos


def get_mean_domain_keyed(args, proto_list, nums, client_domain_ids):
    weighted_protos = {}
    total_counts = {}
    for idx, proto in enumerate(proto_list):
        domain_id = client_domain_ids[idx]
        for cls, p in proto.items():
            key = (cls, domain_id)
            if key not in weighted_protos:
                weighted_protos[key] = torch.zeros_like(p)
                total_counts[key] = 0
            weighted_protos[key] += F.normalize(p, dim=0) * nums[idx][cls]
            total_counts[key] += nums[idx][cls]
    global_protos = {}
    for key in weighted_protos.keys():
        if total_counts[key] > 0:
            global_protos[key] = weighted_protos[key] / total_counts[key]
    return global_protos

def get_gpcl_domain_keyed(args, proto_list, nums, client_domain_ids):
    grouped = {}
    for idx, proto in enumerate(proto_list):
        domain_id = client_domain_ids[idx]
        for cls, p in proto.items():
            key = (cls, domain_id)
            grouped.setdefault(key, []).append(F.normalize(p, dim=0))

    global_protos = {}
    for key, protos in grouped.items():
        if len(protos) == 1:
            global_protos[key] = protos[0]
            continue
        stacked = torch.stack(protos, dim=0)
        mu = stacked.mean(dim=0)
        dists = ((stacked - mu) ** 2).sum(dim=1)
        d_total = dists.sum()
        if d_total.item() <= 1e-12:
            global_protos[key] = mu
            continue
        weights = dists / d_total
        global_protos[key] = (weights.unsqueeze(1) * stacked).sum(dim=0)
    return global_protos


def get_gpcl_flat(proto_list, nums):
    grouped = {}
    for idx, proto in enumerate(proto_list):
        for cls, p in proto.items():
            grouped.setdefault(cls, []).append(F.normalize(p, dim=0))

    global_protos = {}
    for cls, protos in grouped.items():
        if len(protos) == 1:
            global_protos[cls] = protos[0]
            continue
        stacked = torch.stack(protos, dim=0)
        mu = stacked.mean(dim=0)
        dists = ((stacked - mu) ** 2).sum(dim=1)
        d_total = dists.sum()
        if d_total.item() <= 1e-12:
            global_protos[cls] = mu
            continue
        weights = dists / d_total
        global_protos[cls] = (weights.unsqueeze(1) * stacked).sum(dim=0)
    return global_protos


def confidence_weight(n, c=5.0):
    return n / (n + c)


def get_confidence_weighted_domain_keyed(args, proto_list, nums, client_domain_ids, confidence_c=5.0):
    weighted_protos = {}
    total_weights = {}
    for idx, proto in enumerate(proto_list):
        domain_id = client_domain_ids[idx]
        for cls, p in proto.items():
            key = (cls, domain_id)
            n = nums[idx][cls]
            w = confidence_weight(n, confidence_c)
            p_norm = F.normalize(p, dim=0)
            if key not in weighted_protos:
                weighted_protos[key] = torch.zeros_like(p_norm)
                total_weights[key] = 0.0
            weighted_protos[key] += w * p_norm
            total_weights[key] += w
    global_protos = {}
    for key in weighted_protos.keys():
        if total_weights[key] > 1e-12:
            global_protos[key] = weighted_protos[key] / total_weights[key]
    return global_protos


def get_mean_dispersion_domain_keyed(proto_list, var_list, nums, client_domain_ids,
                                      shrinkage_tau=10.0, var_floor=1e-4):
    weighted_means = {}
    total_counts = {}
    for idx, proto in enumerate(proto_list):
        domain_id = client_domain_ids[idx]
        for cls, p in proto.items():
            key = (cls, domain_id)
            if key not in weighted_means:
                weighted_means[key] = torch.zeros_like(p)
                total_counts[key] = 0
            weighted_means[key] += p * nums[idx][cls]
            total_counts[key] += nums[idx][cls]
    domain_mean = {}
    for key in weighted_means:
        if total_counts[key] > 0:
            domain_mean[key] = weighted_means[key] / total_counts[key]

    weighted_within = {}
    weighted_between = {}
    for idx, proto in enumerate(proto_list):
        domain_id = client_domain_ids[idx]
        for cls, mu_i in proto.items():
            key = (cls, domain_id)
            if key not in domain_mean:
                continue
            n_i = nums[idx][cls]
            v_i = var_list[idx][cls]
            mu = domain_mean[key]
            if key not in weighted_within:
                weighted_within[key] = torch.zeros_like(v_i)
                weighted_between[key] = torch.zeros_like(v_i)
            weighted_within[key] += n_i * v_i
            weighted_between[key] += n_i * (mu_i - mu) ** 2
    domain_var_raw = {}
    for key in weighted_within:
        domain_var_raw[key] = (weighted_within[key] + weighted_between[key]) / total_counts[key]

    by_class = {}
    for (cls, dom), v in domain_var_raw.items():
        by_class.setdefault(cls, []).append(v)
    prior_var = {cls: torch.stack(vs).mean(dim=0) for cls, vs in by_class.items()}

    domain_dispersion = {}
    for key, v in domain_var_raw.items():
        cls, dom = key
        n = total_counts[key]
        alpha = shrinkage_tau / (n + shrinkage_tau)
        shrunk = (1 - alpha) * v + alpha * prior_var[cls]
        domain_dispersion[key] = torch.clamp(shrunk, min=var_floor)

    return domain_mean, domain_dispersion


def get_cross_domain_gpcl(domain_keyed_proto, prev_G=None, beta=0.99, eps=1e-8):
    by_class = {}
    for (cls, domain_id), vec in domain_keyed_proto.items():
        by_class.setdefault(cls, {})[domain_id] = vec

    G = {}
    weights_by_class = {}
    for cls, doms in by_class.items():
        domain_ids = list(doms.keys())
        normalized = {d: F.normalize(doms[d], dim=0) for d in domain_ids}
        mu_hat = sum(normalized.values()) / len(domain_ids)
        dists = {d: (normalized[d] - mu_hat).pow(2).sum() for d in domain_ids}
        S_k = sum(dists.values())
        if S_k.item() <= eps:
            weights = {d: 1.0 / len(domain_ids) for d in domain_ids}
        else:
            weights = {d: (dists[d] / S_k).item() for d in domain_ids}
        g_k = sum(weights[d] * normalized[d] for d in domain_ids)
        weights_by_class[cls] = weights

        if prev_G is not None and cls in prev_G:
            G[cls] = beta * prev_G[cls] + (1 - beta) * g_k
        else:
            G[cls] = g_k
    return G, weights_by_class


class PerAvgOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(PerAvgOptimizer, self).__init__(params, defaults)

    def step(self, beta=0):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if(beta != 0):
                    p.data.add_(other=d_p, alpha=-beta)
                else:
                    p.data.add_(other=d_p, alpha=-group['lr'])

def prepare_data_digit_feature_noniid(args):
    transform_mnist = transforms.Compose([
            transforms.Resize([args.size,args.size]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    transform_svhn = transforms.Compose([
            transforms.Resize([args.size,args.size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    transform_usps = transforms.Compose([
            transforms.Resize([args.size,args.size]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    transform_synth = transforms.Compose([
            transforms.Resize([args.size,args.size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    transform_mnistm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if args.num_users == 5:
        mnist_trainset =data_utils.DigitsDataset(data_path="../data/Digits/MNIST", channels=1, percent=args.percent, train=True,  transform=transform_mnist)
        mnist_testset = data_utils.DigitsDataset(data_path="../data/Digits/MNIST", channels=1, percent=args.percent, train=False, transform=transform_mnist)
        svhn_trainset = data_utils.DigitsDataset(data_path='../data/Digits/SVHN', channels=3, percent=args.percent,  train=True,  transform=transform_svhn)
        svhn_testset = data_utils.DigitsDataset(data_path='../data/Digits/SVHN', channels=3, percent=args.percent,  train=False, transform=transform_svhn)
        usps_trainset = data_utils.DigitsDataset(data_path='../data/Digits/USPS', channels=1, percent=args.percent,  train=True,  transform=transform_usps)
        usps_testset = data_utils.DigitsDataset(data_path='../data/Digits/USPS', channels=1, percent=args.percent,  train=False, transform=transform_usps)
        synth_trainset = data_utils.DigitsDataset(data_path='../data/Digits/SynthDigits/', channels=3, percent=args.percent,  train=True,  transform=transform_synth)
        synth_testset = data_utils.DigitsDataset(data_path='../data/Digits/SynthDigits/', channels=3, percent=args.percent,  train=False, transform=transform_synth)
        mnistm_trainset = data_utils.DigitsDataset(data_path='../data/Digits/MNIST_M/', channels=3, percent=args.percent,  train=True,  transform=transform_mnistm)
        mnistm_testset = data_utils.DigitsDataset(data_path='../data/Digits/MNIST_M/', channels=3, percent=args.percent,  train=False, transform=transform_mnistm)
        
        mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True, num_workers=args.number_workers, pin_memory=True)
        mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch,  shuffle=True, num_workers=args.number_workers, pin_memory=True)
        svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch,  shuffle=True, num_workers=args.number_workers, pin_memory=True)
        usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch,  shuffle=True, num_workers=args.number_workers, pin_memory=True)
        synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch,  shuffle=True, num_workers=args.number_workers, pin_memory=True)
        mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)

        train_loaders = [mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader]
        test_loaders  = [mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader]
    else:
        train_loaders, test_loaders = [], []
        for idx in range(args.num_users // 5):
            mnist_trainset = data_utils.DigitsDataset_mul_clients(idx, data_path="../data/Digits/MNIST", channels=1, percent=args.percent, train=True,  transform=transform_mnist, noise=args.noise)
            svhn_trainset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/SVHN', channels=3, percent=args.percent,  train=True,  transform=transform_svhn, noise=args.noise)
            usps_trainset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/USPS', channels=1, percent=args.percent,  train=True,  transform=transform_usps, noise=args.noise)
            synth_trainset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/SynthDigits/', channels=3, percent=args.percent,  train=True,  transform=transform_synth, noise=args.noise)
            mnistm_trainset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/MNIST_M/', channels=3, percent=args.percent,  train=True,  transform=transform_mnistm, noise=args.noise)
            mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True)
            svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch,  shuffle=True)
            usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch,  shuffle=True)
            synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch,  shuffle=True)
            mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch,  shuffle=True)
            train_loaders.extend([mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader])
        
        mnist_testset = data_utils.DigitsDataset_mul_clients(idx, data_path="../data/Digits/MNIST", channels=1, percent=args.percent, train=False, transform=transform_mnist, noise=args.noise)
        svhn_testset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/SVHN', channels=3, percent=args.percent,  train=False, transform=transform_svhn, noise=args.noise)
        usps_testset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/USPS', channels=1, percent=args.percent,  train=False, transform=transform_usps, noise=args.noise)
        synth_testset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/SynthDigits/', channels=3, percent=args.percent,  train=False, transform=transform_synth, noise=args.noise)
        mnistm_testset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/MNIST_M/', channels=3, percent=args.percent,  train=False, transform=transform_mnistm, noise=args.noise)
        mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        
        test_loaders.extend([mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader])
    return train_loaders, test_loaders

def prepare_data_office_feature_noniid(args):
    data_base_path = '../data/office_caltech_10'
    transform_office = transforms.Compose([
            transforms.Resize([args.size, args.size]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([args.size, args.size]),            
            transforms.ToTensor(),
    ])
    
    amazon_trainset = data_utils.OfficeDataset(data_base_path, 'amazon', transform=transform_office)
    amazon_testset = data_utils.OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False)
    caltech_trainset = data_utils.OfficeDataset(data_base_path, 'caltech', transform=transform_office)
    caltech_testset = data_utils.OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False)
    dslr_trainset = data_utils.OfficeDataset(data_base_path, 'dslr', transform=transform_office)
    dslr_testset = data_utils.OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False)
    webcam_trainset = data_utils.OfficeDataset(data_base_path, 'webcam', transform=transform_office)
    webcam_testset = data_utils.OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False)

    amazon_train_loader = torch.utils.data.DataLoader(amazon_trainset, batch_size=args.batch, shuffle=True, num_workers=args.number_workers, pin_memory=True)
    amazon_test_loader = torch.utils.data.DataLoader(amazon_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)

    caltech_train_loader = torch.utils.data.DataLoader(caltech_trainset, batch_size=args.batch, shuffle=True, num_workers=args.number_workers, pin_memory=True)
    caltech_test_loader = torch.utils.data.DataLoader(caltech_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)

    dslr_train_loader = torch.utils.data.DataLoader(dslr_trainset, batch_size=args.batch, shuffle=True, num_workers=args.number_workers, pin_memory=True)
    dslr_test_loader = torch.utils.data.DataLoader(dslr_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)

    webcam_train_loader = torch.utils.data.DataLoader(webcam_trainset, batch_size=args.batch, shuffle=True, num_workers=args.number_workers, pin_memory=True)
    webcam_test_loader = torch.utils.data.DataLoader(webcam_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
    
    train_loaders = [amazon_train_loader, caltech_train_loader, dslr_train_loader, webcam_train_loader]
    test_loaders = [amazon_test_loader, caltech_test_loader, dslr_test_loader, webcam_test_loader]
    return train_loaders, test_loaders

def prepare_data_PACS_feature_noniid(args):
    data_base_path = '../data/PACS'
    transform_PACS= transforms.Compose([
            transforms.Resize([args.size, args.size]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
            transforms.Resize([args.size, args.size]),            
            transforms.ToTensor(),
    ])
    dataset_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loaders, test_loaders = [], []
    for dataset in dataset_name:
        train_set = data_utils.PACSDataset(data_base_path, dataset, transform=transform_PACS)
        test_set = data_utils.PACSDataset(data_base_path, dataset, transform=transform_test, train=False)
        train_loaders.append(torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.number_workers, pin_memory=True))
        test_loaders.append(torch.utils.data.DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True))
    return train_loaders, test_loaders


def prepare_data_office_multi_clients(args, dslr_client_override=None, domain_client_overrides=None):
    transform_office = transforms.Compose([
                transforms.Resize([args.size, args.size]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-30,30)),
                transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
                transforms.Resize([args.size, args.size]),
                transforms.ToTensor(),
        ])
    data_base_path = '../data/office_caltech_10'
    train_loader_list = []
    test_loader_list = []
    client_domains = []
    selected_domain_dict = {'caltech': 3, 'amazon': 2, 'webcam': 1, 'dslr': 4}
    if dslr_client_override is not None:
        selected_domain_dict['dslr'] = dslr_client_override
    if domain_client_overrides is not None:
        for _dom, _n in domain_client_overrides.items():
            if _dom not in selected_domain_dict:
                raise ValueError(
                    f"domain_client_overrides has unknown domain {_dom!r}; "
                    f"expected one of {list(selected_domain_dict.keys())}"
                )
            selected_domain_dict[_dom] = _n
    for domain, client_num in selected_domain_dict.items():
        train_set = data_utils.OfficeDataset(data_base_path, domain, transform = transform_office)
        all_idx = np.arange(len(train_set.labels))
        shuffled_idx = np.random.permutation(all_idx)
        selected_idx = np.array_split(shuffled_idx, client_num)

        for idx in selected_idx:
            train_sampler = torch.utils.data.SubsetRandomSampler(idx)
            client_batch = args.batch
            if getattr(args, "adaptive_local_batch", False):
                client_batch = min(args.batch, int(math.ceil(len(idx) / 2)))
            drop_last_client = (len(idx) % client_batch == 1)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=client_batch, sampler=train_sampler, num_workers=args.number_workers, pin_memory=True, drop_last=drop_last_client)
            train_loader_list.append(train_loader)
            client_domains.append(domain)

    
    for domain in selected_domain_dict:
        test_set = data_utils.OfficeDataset(data_base_path, domain, transform=transform_test, train=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        test_loader_list.append(test_loader)

    return train_loader_list, client_domains, test_loader_list


def prepare_data_PACS_multi_clients(args, domain_client_overrides=None):
    data_base_path = '../data/PACS'
    transform_PACS= transforms.Compose([
                transforms.Resize([args.size, args.size]),            
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-30,30)),
                transforms.ToTensor(),
        ])
    transform_test = transforms.Compose([
                transforms.Resize([args.size, args.size]),            
                transforms.ToTensor(),
        ])
    client_domains = []
    train_loader_list = []
    test_loader_list = []
    selected_domain_dict = {'photo': 3, 'art_painting': 2, 'cartoon': 1, 'sketch': 4}
    if domain_client_overrides is not None:
        for _dom, _n in domain_client_overrides.items():
            if _dom not in selected_domain_dict:
                raise ValueError(
                    f"domain_client_overrides has unknown domain {_dom!r}; "
                    f"expected one of {list(selected_domain_dict.keys())}"
                )
            selected_domain_dict[_dom] = _n
    
    for domain, client_num in selected_domain_dict.items():
        train_set = data_utils.PACSDataset(data_base_path, domain, transform = transform_PACS)
        all_idx = np.arange(len(train_set.labels))
        shuffled_idx = np.random.permutation(all_idx)
        selected_idx = np.array_split(shuffled_idx, client_num)

        for idx in selected_idx:
            train_sampler = torch.utils.data.SubsetRandomSampler(idx)
            client_batch = args.batch
            if getattr(args, "adaptive_local_batch", False):
                client_batch = min(args.batch, int(math.ceil(len(idx) / 2)))
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=client_batch, sampler=train_sampler, num_workers=args.number_workers, pin_memory=True)
            train_loader_list.append(train_loader)
            client_domains.append(domain)

    for domain in selected_domain_dict:
        test_set = data_utils.PACSDataset(data_base_path, domain, transform=transform_test, train=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        test_loader_list.append(test_loader)


    return train_loader_list, client_domains, test_loader_list


def average_protos(protos):
    agg_protos = {}
    for [label, proto_list] in protos.items():
        proto = np.stack(proto_list)
        agg_protos[label] = np.mean(proto, axis=0)

    return agg_protos


def average_weights(w):
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        for i in range(1, len(w)):
            w_avg[0][key] += w[i][key]
        w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
        for i in range(1, len(w)):
            w_avg[i][key] = w_avg[0][key]
    return w_avg

def local_cluster_collect(local_cluster_protos):
    global_collected_protos = {}
    for [idx, cluster_protos_label] in local_cluster_protos.items():
        for [label, cluster_protos_list] in cluster_protos_label.items():
            for i in range(len(cluster_protos_list)):
                if label in global_collected_protos.keys():
                    global_collected_protos[label].append(cluster_protos_list[i])
                else:
                    global_collected_protos[label] = [cluster_protos_list[i]]
    return global_collected_protos

def proto_aggregation_cluster(global_protos_list):
    agg_protos_label = dict()
    for label in global_protos_list.keys():
        for i in range(len(global_protos_list[label])):
            if label in agg_protos_label:
                agg_protos_label[label].append(global_protos_list[label][i])
            else:
                agg_protos_label[label] = [global_protos_list[label][i]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0]
            for i in proto_list:
                proto += i
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0]

    return agg_protos_label
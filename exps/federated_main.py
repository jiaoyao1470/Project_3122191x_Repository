import data_utils
from option import args_parser
from util import *
import torch
import numpy as np
import pandas as pd
import random
from models import *
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torchvision.models as tmodels
from update import LocalUpdate, LocalTest
import copy
from torch import nn
from math import sqrt, isfinite
import os
import time
from transformers import ViTModel, ViTConfig
import collections
import time
from collections import Counter
from PIL import Image
import concurrent.futures
import pickle


def _get_loader_generator(loader):
    if loader is None:
        return None
    candidates = (
        getattr(loader, "generator", None),
        getattr(getattr(loader, "sampler", None), "generator", None),
        getattr(
            getattr(getattr(loader, "batch_sampler", None), "sampler", None),
            "generator",
            None,
        ),
    )
    for generator in candidates:
        if isinstance(generator, torch.Generator):
            return generator
    return None


def _capture_loader_generator_states(loaders):
    if loaders is None:
        return None
    states = []
    for loader in loaders:
        generator = _get_loader_generator(loader)
        states.append(None if generator is None else generator.get_state().clone())
    return states


def _restore_loader_generator_states(loaders, states, label="loaders"):
    if states is None:
        return
    if loaders is None:
        raise RuntimeError(f"Cannot restore {label}: loader list is None")
    if len(loaders) != len(states):
        raise RuntimeError(
            f"Cannot restore {label}: checkpoint has {len(states)} states, "
            f"but the current run has {len(loaders)} loaders"
        )
    for idx, (loader, state) in enumerate(zip(loaders, states)):
        if state is None:
            continue
        generator = _get_loader_generator(loader)
        if generator is None:
            raise RuntimeError(
                f"Cannot restore {label}[{idx}]: checkpoint contains an explicit "
                "generator state, but the current loader has no explicit generator"
            )
        generator.set_state(state.cpu())


def _validate_ours_configuration(args, feature_loader_list, use_mean_dispersion):
    if feature_loader_list is not None and len(feature_loader_list) != args.num_users:
        raise ValueError(
            "feature_loader_list must contain exactly one loader per client: "
            f"expected {args.num_users}, got {len(feature_loader_list)}"
        )
    if use_mean_dispersion and not getattr(args, "domain_keyed_proto", False):
        raise ValueError(
            "use_mean_dispersion=True requires args.domain_keyed_proto=True; "
            "otherwise the mean/dispersion branch is not consumed by the client loss"
        )

def generate_random_tensor(size, p):
    assert 0 <= p <= 1, "p should be between 0 and 1"
    num_ones = int(size * p)
    tensor = torch.cat((torch.ones(num_ones), torch.zeros(size - num_ones)))
    tensor = tensor[torch.randperm(tensor.size(0))]
    return tensor.view(size)

def calculate_q(args, L, diff_dict_flatten, round, client_weights):
    M, G = args.num_users, diff_dict_flatten.shape[1]
    t = round
    indicator = (diff_dict_flatten >= 0).float()
    L = (L * (t - 1) + indicator) / t

    C = torch.where(diff_dict_flatten >= 0, L, 1 - L)

    mask = (C >= args.tau).float()
    masked_diffs = mask * diff_dict_flatten
    d_m = (masked_diffs ** 2).sum(dim=1)

    if round == 1:
        args.p_t = torch.tensor(client_weights).to(args.device)
        args.delta_p_t = torch.zeros(M).to(args.device)

    d_sum = d_m.sum()
    delta_p_t = (1 - args.beta) * args.delta_p_t + args.beta * (d_m / d_sum)
    p_t = args.p_t + delta_p_t
    p_t = p_t / p_t.sum()

    args.p_t = p_t.detach()
    args.delta_p_t = delta_p_t.detach()

    p_t_expanded = p_t.view(M, 1)
    numerator = mask * p_t_expanded
    denominator = numerator.sum(dim=0, keepdim=True) 
    denominator_safe = denominator + (denominator == 0).float()
    q_t = numerator / denominator_safe

    return q_t, L

def compute_num_list(args, train_loader_list):
    num_list = [Counter() for idx in range(args.num_users)]
    for idx in range(args.num_users):
        train_set = iter(train_loader_list[idx])
        for batch_idx in range(len(train_set)):
            images, labels = next(train_set)
            images, labels = images.to(args.device).float(), labels.to(args.device).long()
            for label in labels:
                if label.item() not in num_list[idx]:
                    num_list[idx][label.item()] = 1
                else:
                    num_list[idx][label.item()]+=1
    return num_list

def model_fusion(list_dicts_local_params: list, list_nums_local_data: list):
    local_params = copy.deepcopy(list_dicts_local_params[0])
    for name_param in list_dicts_local_params[0]:
        list_values_param = []
        for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
            list_values_param.append(dict_local_params[name_param] * num_local_data)
        value_global_param = sum(list_values_param) / sum(list_nums_local_data)
        local_params[name_param] = value_global_param
    return local_params

def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(args.num_users):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(args.num_users):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.mode.lower() == 'fedrep':
            for key in server_model.features.state_dict().keys():
                temp = torch.zeros_like(server_model.features.state_dict()[key], dtype=torch.float32)
                for client_idx in range(args.num_users):
                    temp += client_weights[client_idx] * models[client_idx].features.state_dict()[key]
                server_model.features.state_dict()[key].data.copy_(temp)
                for client_idx in range(args.num_users):
                    models[client_idx].features.state_dict()[key].data.copy_(server_model.features.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return copy.deepcopy(server_model), copy.deepcopy(models)

def communication_heal(args, server_model, diff_dict, q):
    M, G = diff_dict.shape
    weighted_update_flat = (q * diff_dict).sum(dim=0)
    idx = 0
    with torch.no_grad():
        for param in server_model.parameters():
            if param.requires_grad:
                num_param = param.numel()
                update_chunk = weighted_update_flat[idx:idx + num_param].view_as(param).to(args.device)
                param.add_(update_chunk)
                idx += num_param

    return copy.deepcopy(server_model)

def get_cls_ratio(args, num_list):
    total_count = sum(num_list.values())
    proportion_list = {key:num / total_count for key, num in num_list.items()}
    return proportion_list

def cal_norm_mean(args, c_means, c_dis):
    glo_means = dict()
    c_dis_temp = torch.ones((args.num_users, args.num_classes))
    for idx in range(args.num_users):
        for key, value in c_dis[idx].items():
            c_dis_temp[idx][key] = value
    c_dis = c_dis_temp.to(args.device)
    total_num_per_cls = c_dis.sum(dim=0)
    for i in range(args.num_classes):
        for c_idx, c_mean in enumerate(c_means):
            if i not in c_mean.keys():
                continue
            temp = glo_means.get(i, 0)
            glo_means[i] = temp + \
                F.normalize(c_mean[i].view(1, -1),
                            dim=1).view(-1) * c_dis[c_idx][i]
        if glo_means.get(i) == None:
            continue
        t = glo_means[i]
        glo_means[i] = t / total_num_per_cls[i]
    return glo_means

def SingleSet(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]  
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2] = local_node.update_weights(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2])
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
    return train_loss, accuracy_list, datasets_name

def fedavg(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2] = local_node.update_weights(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
        global_model, local_models = communication(args, copy.deepcopy(global_model), copy.deepcopy(local_models), client_weights)
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference(args, global_model, train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference(args, global_model, test_loader_list[idx2])
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
    return train_loss, accuracy_list, datasets_name

def FedAS(args, train_loader_list, test_loader_list):
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    local_nodes = [None for i in range(args.num_users)]
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        client_weights = []
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                if round == 0:
                    local_nodes[idx2] = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2], client_weight = local_nodes[idx2].update_weights_FedAS(model=copy.deepcopy(global_model), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
                client_weights.append(client_weight)
        FIM_weight_list = [FIM_value/sum(client_weights) for FIM_value in client_weights]
        global_model, _ = communication(args, copy.deepcopy(global_model), copy.deepcopy(local_models), FIM_weight_list)
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference(args, copy.deepcopy(local_models[idx2]), train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference(args, copy.deepcopy(local_models[idx2]), test_loader_list[idx2])
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
    return train_loss, accuracy_list, datasets_name

def fedProx(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                if round == 0:
                    local_models[idx1 * len(datasets_name) + idx2] = local_node.update_weights(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
                else:
                    local_models[idx1 * len(datasets_name) + idx2] = local_node.update_weights_fedProx(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
        global_model, local_models = communication(args, copy.deepcopy(global_model), copy.deepcopy(local_models), client_weights)
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2])
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
    return train_loss, accuracy_list, datasets_name

def perfedavg(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2] = local_node.update_weights_perfedavg(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
        global_model, local_models = communication(args, copy.deepcopy(global_model), copy.deepcopy(local_models), client_weights)
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2])
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
        
    return train_loss, accuracy_list, datasets_name

def fedrep(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2] = local_node.update_weights_fedrep(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
        global_model, local_models = communication(args, copy.deepcopy(global_model), copy.deepcopy(local_models), client_weights)
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2])
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
        
    return train_loss, accuracy_list, datasets_name

def fedproto(args, train_loader_list, test_loader_list, num_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    global_proto = {}
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        protos = []
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2], proto = local_node.update_weights_proto(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2], global_proto=global_proto)
                protos.append(proto)
        global_proto = proto_aggregation(protos, num_list)
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference_proto(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2], global_proto)
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference_proto(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2], global_proto)
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
    return train_loss, accuracy_list, datasets_name

def fedBN(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2]= local_node.update_weights(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
        global_model, local_models = communication(args, copy.deepcopy(global_model), copy.deepcopy(local_models), client_weights)
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2])
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
    return train_loss, accuracy_list, datasets_name

def moon(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                if round == 0:
                    local_node = LocalUpdate(args=args)
                    local_node.old_model = copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2])
                old_model = copy.deepcopy(local_node.old_model)
                local_models[idx1 * len(datasets_name) + idx2] = local_node.update_weights_moon(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
        global_model, local_models = communication(args, global_model, local_models, client_weights)
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference_moon(args, old_model, local_models[idx1 * len(datasets_name) + idx2], global_model, train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference_moon(args, old_model, local_models[idx1 * len(datasets_name) + idx2], global_model,  test_loader_list[idx2])
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
    return train_loss, accuracy_list, datasets_name

def adcol(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9)
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        features_labels = []
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2], features = local_node.update_weights_adcol(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), discriminator=copy.deepcopy(discriminator), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
                ids = torch.ones(features.shape[0]) * (idx1 * len(datasets_name) + idx2)
                features_labels.append([features, ids])
        loss_temp = [0 for i in range(len(datasets_name))]
        loss_kl = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, kl_loss,  _ = local_test.test_inference_adcol(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2], copy.deepcopy(discriminator))
                    loss_temp[idx2] += loss
                    loss_kl[idx2] += kl_loss
                    _, _, acc = local_test.test_inference_adcol(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2], copy.deepcopy(discriminator))
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('\n{:<11s} | train loss: {:.4f} | kl loss : {:.4f}| Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), loss_kl[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
        features_dataset = data_utils.FeatureDataset(features_labels)
        features_loader = torch.utils.data.DataLoader(features_dataset, batch_size=args.batch, shuffle=True)  
        loss_func = nn.CrossEntropyLoss()
        discriminator.train()
        for _ in range(args.adcol_epoch):
            for x, y in features_loader:
                x, y = x.to(args.device).float(), y.to(args.device).long()
                y_pred = discriminator(x)
                loss = loss_func(y_pred, y).mean()
                discriminator_optimizer.zero_grad()
                loss.backward()
                discriminator_optimizer.step()
    return train_loss, accuracy_list, datasets_name

def RUCR(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        protos, features = [], []
        num = []
        num_list_ = Counter()
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                avg_proto, feature, num_list = local_node.compute_proto(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
                num_list_ += num_list
                num.append(num_list)
                features.append(feature)
                protos.append(avg_proto)
        ratio = get_cls_ratio(args, num_list_)
        global_proto = get_mean(args, protos, num)
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_models[idx1 * len(datasets_name) + idx2] = local_node.update_weights_rucr(model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2], global_proto=global_proto, ratio_list=ratio)
        global_model, local_models = communication(args, global_model, local_models, client_weights)
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2])
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
        norm_means = cal_norm_mean(args, protos, num)
        mixup_cls_params = []
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                mixup_cls_param = local_node.local_crt(copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), norm_means, features[idx1 * len(datasets_name) + idx2])
                mixup_cls_params.append(mixup_cls_param)
        mixup_classifier = model_fusion(mixup_cls_params, loader_size)
        global_model.classifier.load_state_dict(mixup_classifier)
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_models[idx1 * len(datasets_name) + idx2] = copy.deepcopy(global_model)
    return train_loss, accuracy_list, datasets_name

def FedHEAL(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    L = torch.stack([torch.zeros(sum(p.numel() for p in global_model.parameters() if p.requires_grad)) for i in range(args.num_users)], dim = 0).to(args.device)
    for round in tqdm(range(args.iters)):
        diff_dict = []
        print(f'\n | Global Training Round : {round} |\n')
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2] = local_node.update_weights(model=copy.deepcopy(global_model), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2])
                diff_dict_m = {}
                for (name_a, param_a), (name_b, param_b) in zip(local_models[idx1 * len(datasets_name) + idx2].named_parameters(), global_model.named_parameters()):
                    assert name_a == name_b
                    diff_dict_m[name_a] = param_a.data - param_b.data 
                diff_dict.append(diff_dict_m)
        diff_dict_flatten = torch.stack([torch.cat([p.view(-1) for p in diff_dict_m.values()], dim=0) for diff_dict_m in diff_dict], dim=0)
        if round > 0:
            q, L = calculate_q(args, L, diff_dict_flatten, round, client_weights)
        else:
            q = torch.tensor(client_weights).unsqueeze(dim=1).to(args.device) * torch.ones_like(diff_dict_flatten)
        if round != 0:
            global_model = communication_heal(args, copy.deepcopy(global_model), copy.deepcopy(diff_dict_flatten), q)
        else:
            global_model, _ = communication(args, copy.deepcopy(global_model), copy.deepcopy(local_models), client_weights)
        loss_temp = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, _ = local_test.test_inference(args, copy.deepcopy(local_models[idx2]), train_loader_list[idx1 * len(datasets_name) + idx2])
                    loss_temp[idx2] += loss
                    _, acc = local_test.test_inference(args, copy.deepcopy(local_models[idx2]), test_loader_list[idx2])
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('{:<11s} | train loss: {:.4f} | Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
    return train_loss, accuracy_list, datasets_name

def ours(args, train_loader_list, test_loader_list, client_domains=None, checkpoint_path=None, checkpoint_every=5, acc_log_path=None, weights_dir=None, g_log_path=None, lambda_G=0.0, kl_weight_multiplier=1.0, ce_weight_multiplier=1.0, lambda_mixup=0.0, domain_balanced_gc=False, feature_bank_domains=None, lambda_FB=0.0, bn_affine_sync_domains=None, feature_loader_list=None, preserve_feature_probe_state=False, use_mean_dispersion=False, mean_dispersion_shrinkage_tau=10.0, mean_dispersion_var_floor=1e-4, lambda_md=None, peer_update_mixing_domains=None, peer_update_mixing_alpha=0.0, input_mixup_domains=None, lambda_input_mixup=0.0, input_mixup_alpha=0.2):
    _validate_ours_configuration(args, feature_loader_list, use_mean_dispersion)
    global_mean_md = {}
    global_dispersion = {}
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    best_local_models = []
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]

    if client_domains is not None:
        datasets_name = []
        for d in client_domains:
            if d not in datasets_name:
                datasets_name.append(d)

    domain_to_id = {d: i for i, d in enumerate(datasets_name)}
    if args.domain_keyed_proto:
        print(f"domain_to_id = {domain_to_id}")
    client_domain_ids = [domain_to_id[d] for d in client_domains] if client_domains is not None else None

    num_domains = args.num_users if getattr(args, "no_discriminator_fix", False) else len(datasets_name)
    discriminator = Discriminator(global_model, num_domains).to(args.device)

    global_classifier = classifier_model(args, args.num_classes).to(args.device)
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9)
    classifier_optimizer = torch.optim.SGD(global_classifier.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9)
    global_proto = {}
    best_acc = 0
    prev_G_state = None
    feature_bank_prev = {}
    if feature_bank_domains and client_domains is None:
        raise ValueError(
            "feature_bank_domains requires client_domains for same-domain peer selection."
        )

    if bn_affine_sync_domains and client_domains is None:
        raise ValueError(
            "bn_affine_sync_domains requires client_domains to identify same-domain clients."
        )

    if peer_update_mixing_domains and client_domains is None:
        raise ValueError(
            "peer_update_mixing_domains requires client_domains to identify same-domain clients."
        )
    if peer_update_mixing_domains and not (0.0 <= peer_update_mixing_alpha <= 1.0):
        raise ValueError(
            f"peer_update_mixing_alpha={peer_update_mixing_alpha} is outside [0,1] -- alpha is a "
            "convex-mixing coefficient (theta^{t+1} = theta^{t,+} + alpha*(mean_delta - delta_i)), "
            "not a loss weight; values outside [0,1] are very likely a typo (e.g. 10 meant as 10%)."
        )
    _peer_mix_eligible_names = None
    if peer_update_mixing_domains:
        _bn_param_names = set()
        for _bn_scope_name, _m in local_models[0].features.named_modules():
            if isinstance(_m, nn.modules.batchnorm._BatchNorm) and _m.weight is not None:
                _bn_param_names.add(f"{_bn_scope_name}.weight")
                _bn_param_names.add(f"{_bn_scope_name}.bias")
        _peer_mix_eligible_names = [
            name for name, _ in local_models[0].features.named_parameters()
            if name not in _bn_param_names
        ]

    if input_mixup_domains and client_domains is None:
        raise ValueError(
            "input_mixup_domains requires client_domains to identify which clients to apply MixUp to."
        )
    if input_mixup_domains and not (0.0 <= lambda_input_mixup <= 1.0):
        raise ValueError(
            f"lambda_input_mixup={lambda_input_mixup} is outside [0,1] -- it is now a CONVEX-BLEND "
            "coefficient (CE_term = (1-lambda)*CE_clean + lambda*CE_mix), not an additive loss "
            "weight, so values outside [0,1] don't have a well-defined meaning."
        )
    if input_mixup_domains and lambda_input_mixup > 0 and input_mixup_alpha <= 0:
        raise ValueError(
            f"input_mixup_alpha={input_mixup_alpha} with lambda_input_mixup>0 -- alpha<=0 makes "
            "mix_rng.beta(alpha, alpha) draw lam_mix=1.0 every time (mixed_images degenerates to "
            "images, mixed_labels degenerates to labels), silently turning 'MixUp' into a redundant "
            "second forward pass of the same clean batch with no actual interpolation. This is very "
            "likely a misconfiguration -- fail loudly instead of running a mislabeled experiment."
        )
    if input_mixup_domains and client_domains is not None:
        _unknown_input_mixup_domains = set(input_mixup_domains) - set(client_domains)
        if _unknown_input_mixup_domains:
            raise ValueError(
                f"input_mixup_domains contains domain name(s) not present in client_domains: "
                f"{sorted(_unknown_input_mixup_domains)} (available: {sorted(set(client_domains))}). "
                "This is very likely a typo (e.g. wrong case) that would otherwise silently be a no-op."
            )

    start_round = 0
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        for client_idx in range(args.num_users):
            local_models[client_idx].load_state_dict(checkpoint["local_models_state"][client_idx])
        global_classifier.load_state_dict(checkpoint["global_classifier_state"])
        discriminator.load_state_dict(checkpoint["discriminator_state"])
        discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state"])
        classifier_optimizer.load_state_dict(checkpoint["classifier_optimizer_state"])
        train_loss = checkpoint["train_loss"]
        accuracy_list = checkpoint["accuracy_list"]
        best_acc = checkpoint["best_acc"]
        global_proto = checkpoint["global_proto"]
        global_mean_md = checkpoint.get("global_mean_md", {})
        global_dispersion = checkpoint.get("global_dispersion", {})
        prev_G_state = checkpoint.get("prev_G_state", None)
        feature_bank_prev = checkpoint.get("feature_bank_prev", {})
        if args.domain_keyed_proto and "domain_to_id" in checkpoint:
            assert checkpoint["domain_to_id"] == domain_to_id, (
                f"domain_to_id mismatch on resume: checkpoint has {checkpoint['domain_to_id']}, "
                f"this run computed {domain_to_id} -- resuming with a different domain ordering "
                f"would silently corrupt global_proto's (class,domain) keys"
            )
        try:
            random.setstate(checkpoint["random_state"])
            np.random.set_state(checkpoint["np_random_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu().byte())
            if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all([s.cpu().byte() for s in checkpoint["cuda_rng_state"]])
        except Exception as e:
            print(f"warning: failed to restore RNG state ({e}); continuing without it")
        _restore_loader_generator_states(
            train_loader_list,
            checkpoint.get("train_loader_generator_states"),
            "train_loader_list",
        )
        _restore_loader_generator_states(
            feature_loader_list,
            checkpoint.get("feature_loader_generator_states"),
            "feature_loader_list",
        )
        start_round = checkpoint["round"] + 1
        print(f"Resuming at round {start_round}")
    for round in tqdm(range(start_round, args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        features_idx = []
        features_labels = []
        client_encoder_deltas = {}
        for client_idx in range(args.num_users):
            local_node = LocalUpdate(args=args)
            if args.domain_keyed_proto:
                my_domain_id = client_domain_ids[client_idx]
                client_global_proto = {
                    cls: global_proto[(cls, my_domain_id)]
                    for cls in range(args.num_classes)
                    if (cls, my_domain_id) in global_proto
                }
                if use_mean_dispersion:
                    client_mean_md = {
                        cls: global_mean_md[(cls, my_domain_id)]
                        for cls in range(args.num_classes)
                        if (cls, my_domain_id) in global_mean_md
                    }
                    client_dispersion = {
                        cls: global_dispersion[(cls, my_domain_id)]
                        for cls in range(args.num_classes)
                        if (cls, my_domain_id) in global_dispersion
                    }
                else:
                    client_mean_md = None
                    client_dispersion = None
            else:
                client_global_proto = global_proto
                client_mean_md = None
                client_dispersion = None
            if isinstance(lambda_G, dict):
                if client_domains is None:
                    raise ValueError(
                        "dict-valued lambda_G requires client_domains to resolve per-domain values."
                    )
                domain_for_client = client_domains[client_idx]
                if domain_for_client not in lambda_G:
                    raise KeyError(
                        f"Missing lambda_G for domain '{domain_for_client}'. "
                        f"Available domains: {list(lambda_G.keys())}"
                    )
                lambda_G_client = float(lambda_G[domain_for_client])
            else:
                lambda_G_client = float(lambda_G)
            my_domain_for_fb = client_domains[client_idx] if client_domains is not None else None
            if feature_bank_domains and my_domain_for_fb in feature_bank_domains:
                peer_items = [
                    (feat, lab) for j, (feat, lab) in feature_bank_prev.get(my_domain_for_fb, {}).items()
                    if j != client_idx
                ]
                if peer_items:
                    client_feature_bank = (
                        torch.cat([feat for feat, _ in peer_items], dim=0),
                        torch.cat([lab for _, lab in peer_items], dim=0),
                    )
                else:
                    client_feature_bank = None
            else:
                client_feature_bank = None
            client_feature_loader = None
            if feature_loader_list is not None and feature_loader_list[client_idx] is not None:
                client_feature_loader = feature_loader_list[client_idx]
            my_domain_for_input_mixup = client_domains[client_idx] if client_domains is not None else None
            lambda_input_mixup_client = (
                lambda_input_mixup
                if (input_mixup_domains and my_domain_for_input_mixup in input_mixup_domains)
                else 0.0
            )
            input_mixup_seed_client = args.seed + 50000 + 1000 * client_idx + round
            my_domain_for_mix = client_domains[client_idx] if client_domains is not None else None
            _mix_pre_state = None
            if peer_update_mixing_domains and my_domain_for_mix in peer_update_mixing_domains:
                _pre_params = dict(local_models[client_idx].features.named_parameters())
                _mix_pre_state = {name: _pre_params[name].detach().clone() for name in _peer_mix_eligible_names}
            local_models[client_idx], features_label= local_node.update_weights_ours_debug(round, model=copy.deepcopy(local_models[client_idx]), discriminator=copy.deepcopy(discriminator), classifier_model=copy.deepcopy(global_classifier), train_loader=train_loader_list[client_idx],global_proto=client_global_proto, momentum = args.momentum, num_domains=num_domains, global_G=(prev_G_state if args.domain_keyed_proto else None), lambda_G=lambda_G_client, kl_weight_multiplier=kl_weight_multiplier, ce_weight_multiplier=ce_weight_multiplier, lambda_mixup=lambda_mixup, feature_bank=client_feature_bank, lambda_FB=lambda_FB, feature_loader=client_feature_loader, preserve_feature_probe_state=preserve_feature_probe_state, use_mean_dispersion=use_mean_dispersion, global_mean_md=client_mean_md, global_dispersion=client_dispersion, lambda_md=lambda_md, lambda_input_mixup=lambda_input_mixup_client, input_mixup_alpha=input_mixup_alpha, input_mixup_seed=input_mixup_seed_client)
            if _mix_pre_state is not None:
                _post_params = dict(local_models[client_idx].features.named_parameters())
                client_encoder_deltas[client_idx] = {
                    name: (_post_params[name].detach().clone() - _mix_pre_state[name])
                    for name in _peer_mix_eligible_names
                }
            if round == start_round:
                domain_name = client_domains[client_idx] if client_domains is not None else "N/A"
                print(f"    [lambda_G sanity check] client={client_idx} "
                      f"domain={domain_name} lambda_G={lambda_G_client:.5f}")
            features_labels.append([features_label[0], features_label[1]])
        local_protos = []
        num = []
        num_list_ = Counter()
        for client_idx in range(args.num_users):
            avg_proto, num_list = local_node.compute_proto_debug(train_loader=features_labels[client_idx])
            local_protos.append(avg_proto)
            num_list_ += num_list
            num.append(num_list)
        if args.domain_keyed_proto and getattr(args, "domain_keyed_proto_gpcl", False):
            global_proto = get_gpcl_domain_keyed(args, local_protos, num, client_domain_ids)
        elif args.domain_keyed_proto and getattr(args, "layer1_confidence_weighted", False):
            global_proto = get_confidence_weighted_domain_keyed(args, local_protos, num, client_domain_ids)
        elif args.domain_keyed_proto:
            global_proto = get_mean_domain_keyed(args, local_protos, num, client_domain_ids)
        elif getattr(args, "use_flat_gpcl", False):
            global_proto = get_gpcl_flat(local_protos, num)
        else:
            global_proto = get_mean(args, local_protos, num)

        if use_mean_dispersion:
            local_protos_raw = []
            local_vars_raw = []
            num_raw = []
            for client_idx in range(args.num_users):
                avg_proto_raw, var_proto_raw, num_list_raw = local_node.compute_proto_dispersion_debug(
                    train_loader=features_labels[client_idx])
                local_protos_raw.append(avg_proto_raw)
                local_vars_raw.append(var_proto_raw)
                num_raw.append(num_list_raw)
            global_mean_md, global_dispersion = get_mean_dispersion_domain_keyed(
                local_protos_raw, local_vars_raw, num_raw, client_domain_ids,
                shrinkage_tau=mean_dispersion_shrinkage_tau, var_floor=mean_dispersion_var_floor)

        if args.domain_keyed_proto:
            G_this_round, weights_this_round = get_cross_domain_gpcl(global_proto, prev_G=prev_G_state, beta=0.99)
            prev_G_state = G_this_round
        else:
            G_this_round, weights_this_round = None, None

        if feature_bank_domains:
            new_feature_bank = {}
            for client_idx in range(args.num_users):
                dom_for_bank = client_domains[client_idx] if client_domains is not None else datasets_name[client_idx]
                if dom_for_bank in feature_bank_domains:
                    new_feature_bank.setdefault(dom_for_bank, {})[client_idx] = (
                        features_labels[client_idx][0].detach().clone().cpu(),
                        features_labels[client_idx][1].detach().clone().cpu(),
                    )
            feature_bank_prev = new_feature_bank

        if args.domain_keyed_proto and g_log_path is not None:
            os.makedirs(os.path.dirname(g_log_path), exist_ok=True)
            write_header = not os.path.exists(g_log_path)
            with open(g_log_path, "a") as gf:
                if write_header:
                    gf.write("round,domain,avg_weight_across_classes,n_classes,G_avg_norm\n")
                domain_weight_sums = {}
                domain_weight_counts = {}
                for cls, w_by_domain in weights_this_round.items():
                    for dom_id, w in w_by_domain.items():
                        domain_weight_sums[dom_id] = domain_weight_sums.get(dom_id, 0.0) + w
                        domain_weight_counts[dom_id] = domain_weight_counts.get(dom_id, 0) + 1
                G_avg_norm = sum(v.norm().item() for v in G_this_round.values()) / len(G_this_round)
                if not isfinite(G_avg_norm):
                    print(f"  [G_LOG WARNING round {round}] G_avg_norm is not finite ({G_avg_norm}) <-- NaN/Inf?")
                for dom_id in sorted(domain_weight_sums.keys()):
                    dom_name = datasets_name[dom_id] if dom_id < len(datasets_name) else str(dom_id)
                    avg_w = domain_weight_sums[dom_id] / domain_weight_counts[dom_id]
                    if avg_w > 0.5:
                        print(f"  [G_LOG WARNING round {round}] {dom_name} avg_weight={avg_w:.4f} <-- DOMINANT WEIGHT?")
                    gf.write(f"{round},{dom_name},{avg_w},{domain_weight_counts[dom_id]},{G_avg_norm}\n")

        for client_idx in range(args.num_users):
            features = features_labels[client_idx][0]
            labels = features_labels[client_idx][1]
            size = features.shape[1]
            mask = generate_random_tensor(size, p=0.9).to(args.device)
            lam = np.round(args.uniform_left + args.uniform_right * np.random.random(), 2)
            features_noise = []
            if args.domain_keyed_proto:
                my_domain_id = client_domain_ids[client_idx]
                for idx, label in enumerate(labels):
                    key = (label.item(), my_domain_id)
                    assert key in global_proto, (
                        f"expected prototype for {key} to exist (client {client_idx} just "
                        f"contributed a sample of this class/domain this round)"
                    )
                    features_noise.append(lam * features[idx] + (1 - lam) * global_proto[key])
            else:
                for idx, label in enumerate(labels):
                    features_noise.append(lam * features[idx] + (1 - lam) * global_proto[label.item()])
            features_noise = torch.stack(features_noise, dim=0)
            features_masked = features_noise * mask
            features_labels[client_idx] = [features_masked, labels]
            if getattr(args, "no_discriminator_fix", False):
                ids = torch.ones(features.shape[0]) * (client_idx)
            else:
                domain_for_client = client_domains[client_idx] if client_domains is not None else datasets_name[client_idx]
                ids = torch.ones(features.shape[0]) * domain_to_id[domain_for_client]
            features_idx.append([features_masked, ids])

        loss_by_domain = {d: 0.0 for d in datasets_name}
        kl_by_domain = {d: 0.0 for d in datasets_name}
        acc_by_domain = {d: 0.0 for d in datasets_name}
        count_by_domain = {d: 0.0 for d in datasets_name}

        for client_idx in range(args.num_users):
            domain = client_domains[client_idx] if client_domains is not None else datasets_name[client_idx]
            domain_pos = datasets_name.index(domain)
            with torch.no_grad():
                local_test = LocalTest(args=args)
                loss, kl_loss, _ = local_test.test_inference_ours(args, local_models[client_idx], train_loader_list[client_idx], copy.deepcopy(discriminator), num_domains=num_domains)
                loss_by_domain[domain] += loss
                kl_by_domain[domain] += kl_loss
                _, _, acc = local_test.test_inference_ours(args, local_models[client_idx], test_loader_list[domain_pos], copy.deepcopy(discriminator), num_domains=num_domains)
                acc_by_domain[domain] += acc
                count_by_domain[domain] += 1


        for domain in datasets_name:
            avg_loss = loss_by_domain[domain] / count_by_domain[domain]
            avg_kl = kl_by_domain[domain] / count_by_domain[domain]
            avg_acc = acc_by_domain[domain] / count_by_domain[domain]
            print('\n{:<11s} | train loss: {:.4f} | kl loss : {:.4f}| Test Acc: {:.4f}'.format(domain, avg_loss, avg_kl, avg_acc))
            train_loss[domain].append(copy.deepcopy(avg_loss))
            accuracy_list[domain].append(copy.deepcopy(avg_acc))

        if acc_log_path is not None:
            os.makedirs(os.path.dirname(acc_log_path), exist_ok=True)
            write_header = not os.path.exists(acc_log_path)
            with open(acc_log_path, "a") as f:
                if write_header:
                    f.write("round,domain,train_loss,kl_loss,acc\n")
                for domain in datasets_name:
                    f.write(f"{round},{domain},{train_loss[domain][-1]},{kl_by_domain[domain]/count_by_domain[domain]},{accuracy_list[domain][-1]}\n")

        current_avg_acc = sum(acc_by_domain[d] / count_by_domain[d] for d in datasets_name) / len(datasets_name)
        if current_avg_acc > best_acc:
            model_paths = weights_dir if weights_dir is not None else f"weights/{args.seed}/{args.dataset}/"
            os.makedirs(model_paths, exist_ok=True)
            for client_idx in range(args.num_users):
                domain = client_domains[client_idx] if client_domains is not None else datasets_name[client_idx]
                if client_domains is not None:
                    model_save_path = f"best_local_model_client{client_idx}_{domain}.pth"
                else:
                    model_save_path = f"best_local_model_{domain}.pth"
                torch.save(local_models[client_idx].state_dict(), model_paths + model_save_path)
            best_acc = current_avg_acc
            
        if domain_balanced_gc:
            domain_sample_counts = {}
            for client_idx in range(args.num_users):
                d = client_domains[client_idx] if client_domains is not None else datasets_name[client_idx]
                domain_sample_counts[d] = domain_sample_counts.get(d, 0) + features_labels[client_idx][0].shape[0]
            n_total_pooled = sum(domain_sample_counts.values())
            n_domains_pooled = len(domain_sample_counts)
            weight_chunks = []
            for client_idx in range(args.num_users):
                d = client_domains[client_idx] if client_domains is not None else datasets_name[client_idx]
                n_client = features_labels[client_idx][0].shape[0]
                scaled_w = n_total_pooled / (n_domains_pooled * domain_sample_counts[d])
                weight_chunks.append(torch.full((n_client,), scaled_w))
            sample_weights = torch.cat(weight_chunks, dim=0)
            if round == start_round:
                per_domain_w = {
                    d: n_total_pooled / (n_domains_pooled * n) for d, n in domain_sample_counts.items()
                }
                print(f"    [domain-balanced GC] pooled samples per domain: {domain_sample_counts}")
                weight_str = ", ".join(f"{d}: {v:.4f}" for d, v in per_domain_w.items())
                print(f"    [domain-balanced GC] per-sample weights: {{{weight_str}}} "
                      f"(mean weight = 1.0 by construction)")

            features_label_dataset = data_utils.FeatureDatasetWeighted(features_labels, sample_weights)
            features_label_loader = torch.utils.data.DataLoader(features_label_dataset, batch_size=args.batch, shuffle=True)
            loss_func_per_sample = nn.CrossEntropyLoss(reduction="none")
            loss_func = nn.CrossEntropyLoss()
            global_classifier.train()
            for _ in range(args.adcol_epoch):
                for x, y, w in features_label_loader:
                    x, y = x.to(args.device).float(), y.to(args.device).long()
                    w = w.to(args.device).float()
                    y_pred = global_classifier(x)
                    per_sample = loss_func_per_sample(y_pred, y)
                    loss1 = (w * per_sample).mean()
                    classifier_optimizer.zero_grad()
                    loss1.backward()
                    classifier_optimizer.step()
        else:
            features_label_dataset = data_utils.FeatureDataset(features_labels)
            features_label_loader = torch.utils.data.DataLoader(features_label_dataset, batch_size=args.batch, shuffle=True)
            loss_func = nn.CrossEntropyLoss()
            global_classifier.train()
            for _ in range(args.adcol_epoch):
                for x, y in features_label_loader:
                    x, y= x.to(args.device).float(), y.to(args.device).long()
                    y_pred = global_classifier(x)
                    loss1 = loss_func(y_pred, y).mean()
                    classifier_optimizer.zero_grad()
                    loss1.backward()
                    classifier_optimizer.step()

        features_dataset = data_utils.FeatureDataset(features_idx)
        features_loader = torch.utils.data.DataLoader(features_dataset, batch_size=args.batch, shuffle=True)  
        discriminator.train()
        for _ in range(args.adcol_epoch):
            for x, y in features_loader:
                x, y = x.to(args.device).float(), y.to(args.device).long()
                y_pred = discriminator(x)
                loss2 = loss_func(y_pred, y).mean()
                discriminator_optimizer.zero_grad()
                loss2.backward()
                discriminator_optimizer.step()

        if bn_affine_sync_domains:
            for _sync_domain in bn_affine_sync_domains:
                _members = [i for i in range(args.num_users) if client_domains[i] == _sync_domain]
                if len(_members) < 2:
                    continue
                _counts = []
                for i in _members:
                    _idxs = getattr(train_loader_list[i].sampler, "indices", None)
                    _counts.append(len(_idxs) if _idxs is not None else len(train_loader_list[i].dataset))
                _total = float(sum(_counts))
                _bn_per_client = [
                    [m for m in local_models[i].features.modules()
                     if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.weight is not None]
                    for i in _members
                ]
                if len({len(b) for b in _bn_per_client}) != 1:
                    raise RuntimeError(
                        f"bn_affine_sync: clients in domain {_sync_domain} expose different BN layer "
                        f"counts {[len(b) for b in _bn_per_client]} -- models are not identically built."
                    )
                with torch.no_grad():
                    for _layer_pos in range(len(_bn_per_client[0])):
                        _g = sum(_counts[j] * _bn_per_client[j][_layer_pos].weight.data.double()
                                 for j in range(len(_members))) / _total
                        _b = sum(_counts[j] * _bn_per_client[j][_layer_pos].bias.data.double()
                                 for j in range(len(_members))) / _total
                        for j in range(len(_members)):
                            _bn = _bn_per_client[j][_layer_pos]
                            _bn.weight.data.copy_(_g.to(_bn.weight.dtype))
                            _bn.bias.data.copy_(_b.to(_bn.bias.dtype))
                if round == start_round:
                    print(f"    [bn_affine_sync] domain={_sync_domain} clients={_members} "
                          f"n_train={_counts} bn_layers={len(_bn_per_client[0])}")

        if peer_update_mixing_domains:
            for _mix_domain in peer_update_mixing_domains:
                _mix_members = [i for i in range(args.num_users) if client_domains[i] == _mix_domain]
                if len(_mix_members) < 2:
                    continue
                _missing_deltas = [i for i in _mix_members if i not in client_encoder_deltas]
                if _missing_deltas:
                    raise RuntimeError(
                        f"peer_update_mixing: domain {_mix_domain} clients {_missing_deltas} have no "
                        "recorded Delta_i^t for this round -- the pre-SGD snapshot loop above and this "
                        "mixing loop must be scoped identically. Current invariant guarantees this can't "
                        "happen (both branch on 'client_domains[idx] in peer_update_mixing_domains'); "
                        "this is a fail-fast guard against that invariant ever silently breaking."
                    )
                _mix_counts = []
                for i in _mix_members:
                    _idxs = getattr(train_loader_list[i].sampler, "indices", None)
                    _mix_counts.append(len(_idxs) if _idxs is not None else len(train_loader_list[i].dataset))
                _mix_total = float(sum(_mix_counts))
                _domain_mean_delta = {
                    name: sum(
                        _mix_counts[j] * client_encoder_deltas[_mix_members[j]][name].double()
                        for j in range(len(_mix_members))
                    ) / _mix_total
                    for name in _peer_mix_eligible_names
                }
                _delta_norm_by_client = {
                    i: sum(client_encoder_deltas[i][name].double().pow(2).sum() for name in _peer_mix_eligible_names).sqrt().item()
                    for i in _mix_members
                }
                _correction_norm_by_client = {
                    i: sum(
                        (peer_update_mixing_alpha * (_domain_mean_delta[name] - client_encoder_deltas[i][name].double())).pow(2).sum()
                        for name in _peer_mix_eligible_names
                    ).sqrt().item()
                    for i in _mix_members
                }
                with torch.no_grad():
                    for i in _mix_members:
                        _params_i = dict(local_models[i].features.named_parameters())
                        for name in _peer_mix_eligible_names:
                            _delta_i = client_encoder_deltas[i][name].double()
                            _mixed = _params_i[name].data.double() + peer_update_mixing_alpha * (
                                _domain_mean_delta[name] - _delta_i
                            )
                            _params_i[name].data.copy_(_mixed.to(_params_i[name].dtype))
                _delta_norm_str = ", ".join(f"{i}:{v:.6f}" for i, v in _delta_norm_by_client.items())
                _correction_norm_str = ", ".join(f"{i}:{v:.8f}" for i, v in _correction_norm_by_client.items())
                print(f"    [peer_update_mixing] round={round} domain={_mix_domain} clients={_mix_members} "
                      f"n_train={_mix_counts} eligible_params={len(_peer_mix_eligible_names)} "
                      f"alpha={peer_update_mixing_alpha}  "
                      f"||Delta_i||={{{_delta_norm_str}}}  "
                      f"||correction_i||={{{_correction_norm_str}}}")

        if checkpoint_path is not None and ((round + 1) % checkpoint_every == 0 or round == args.iters - 1):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            checkpoint = {
                "round": round,
                "local_models_state": [m.state_dict() for m in local_models],
                "global_classifier_state": global_classifier.state_dict(),
                "discriminator_state": discriminator.state_dict(),
                "discriminator_optimizer_state": discriminator_optimizer.state_dict(),
                "classifier_optimizer_state": classifier_optimizer.state_dict(),
                "domain_to_id": domain_to_id,
                "train_loss": train_loss,
                "accuracy_list": accuracy_list,
                "best_acc": best_acc,
                "global_proto": global_proto,
                "global_mean_md": global_mean_md,
                "global_dispersion": global_dispersion,
                "prev_G_state": prev_G_state,
                "feature_bank_prev": feature_bank_prev,
                "random_state": random.getstate(),
                "np_random_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "train_loader_generator_states": _capture_loader_generator_states(train_loader_list),
                "feature_loader_generator_states": _capture_loader_generator_states(feature_loader_list),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at round {round}")
    if args.exp == 4:
        file_path = f'pkl/{args.dataset}/{args.mode}_{args.seed}_features_labels.pkl'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(features_labels, f)
    return train_loss, accuracy_list, datasets_name

def ablation1(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    global_classifier = classifier_model(args, args.num_classes).to(args.device)
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9)
    classifier_optimizer = torch.optim.SGD(global_classifier.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9)
    global_proto = {}
    best_acc = 0
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        features_idx = []
        features_labels = [[] for i in range(len(datasets_name))]
        features_origin = []
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2], features_label= local_node.update_weights_ours_ablation1(round, model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), discriminator=copy.deepcopy(discriminator), classifier_model=copy.deepcopy(global_classifier), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2],global_proto=global_proto, momentum = args.momentum)
                features_origin.append([features_label[0], features_label[1]])
        local_protos = []
        num = []
        num_list_ = Counter()
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                avg_proto, num_list = local_node.compute_proto_debug(train_loader=features_origin[idx1 * len(datasets_name) + idx2])
                local_protos.append(avg_proto)
                num_list_ += num_list
                num.append(num_list)
        global_proto = get_mean(args, local_protos, num)
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                features = features_origin[idx1 * len(datasets_name) + idx2][0]
                labels = features_origin[idx1 * len(datasets_name) + idx2][1]
                size = features.shape[1]
                mask = generate_random_tensor(size, p=0.9).to(args.device)
                lam = np.round(args.uniform_left + args.uniform_right * np.random.random(), 2)
                features_noise = []
                for idx, label in enumerate(labels):
                    features_noise.append(lam * features[idx] + (1 - lam) * global_proto[label.item()])
                features_noise = torch.stack(features_noise, dim=0)
                features_masked = features_noise * mask
                features_labels[idx1 * len(datasets_name) + idx2] = [features_masked, labels]
                ids = torch.ones(features.shape[0]) * (idx1 * len(datasets_name) + idx2)
                features_idx.append([features_masked, ids])
        loss_temp = [0 for i in range(len(datasets_name))]
        loss_kl = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        loss_info = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, kl_loss, info_loss, _ = local_test.test_inference_ablation1(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2], global_proto=global_proto, dmodel = copy.deepcopy(discriminator))
                    loss_temp[idx2] += loss
                    loss_kl[idx2] += kl_loss
                    loss_info[idx2] += info_loss
                    _, _, _, acc = local_test.test_inference_ablation1(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2], global_proto=global_proto, dmodel = copy.deepcopy(discriminator))
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('\n{:<11s} | CE loss: {:.4f} | kl loss : {:.4f}| infoNCEloss : {:.4f}| Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), loss_kl[idx] / (args.num_users // len(datasets_name)), loss_info[idx] / (args.num_users // len(datasets_name)),  acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
        if np.average(acc_temp) > best_acc:
            file_path = f'pkl/{args.dataset}/{args.mode}_{args.seed}_features_labels_idx_loss{args.loss_component}.pkl'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            for i in range(args.num_users):
                features_origin[i][0] = features_origin[i][0].to("cpu")
                features_origin[i][1] = features_origin[i][1].to("cpu")
            with open(file_path, 'wb') as f:
                pickle.dump(features_origin, f)
            best_acc = np.average(acc_temp) 
        
        features_label_dataset = data_utils.FeatureDataset(features_labels)
        features_label_loader = torch.utils.data.DataLoader(features_label_dataset, batch_size=args.batch, shuffle=True)  
        loss_func = nn.CrossEntropyLoss()
        global_classifier.train()
        for _ in range(args.adcol_epoch):
            for x, y in features_label_loader:
                x, y= x.to(args.device).float(), y.to(args.device).long()
                y_pred = global_classifier(x)
                loss1 = loss_func(y_pred, y).mean()
                classifier_optimizer.zero_grad()
                loss1.backward()
                classifier_optimizer.step()
        if args.loss_component in [3, 4]:
            features_dataset = data_utils.FeatureDataset(features_idx)
            features_loader = torch.utils.data.DataLoader(features_dataset, batch_size=args.batch, shuffle=True)  
            discriminator.train()
            for _ in range(args.adcol_epoch):
                for x, y in features_loader:
                    x, y = x.to(args.device).float(), y.to(args.device).long()
                    y_pred = discriminator(x)
                    loss2 = loss_func(y_pred, y).mean()
                    discriminator_optimizer.zero_grad()
                    loss2.backward()
                    discriminator_optimizer.step()
    return train_loss, accuracy_list, datasets_name

def ablation2(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    global_classifier = classifier_model(args, args.num_classes).to(args.device)
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9)
    classifier_optimizer = torch.optim.SGD(global_classifier.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9)
    global_proto = {}
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        features_idx = []
        features_labels = []
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2], features_label= local_node.update_weights_ours_ablation2(round, model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), discriminator=copy.deepcopy(discriminator), classifier_model=copy.deepcopy(global_classifier), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2],global_proto=global_proto, momentum = args.momentum)
                features_labels.append([features_label[0], features_label[1]])
        local_protos = []
        num = []
        num_list_ = Counter()
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                avg_proto, num_list = local_node.compute_proto_debug(train_loader=features_labels[idx1 * len(datasets_name) + idx2])
                local_protos.append(avg_proto)
                num_list_ += num_list
                num.append(num_list)
        global_proto = get_mean(args, local_protos, num)
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                features = features_labels[idx1 * len(datasets_name) + idx2][0]
                labels = features_labels[idx1 * len(datasets_name) + idx2][1]
                size = features.shape[1]
                mask = generate_random_tensor(size, p=0.9).to(args.device)
                lam = np.round(args.uniform_left + args.uniform_right * np.random.random(), 2)
                features_noise = []
                for idx, label in enumerate(labels):
                    features_noise.append(lam * features[idx] + (1 - lam) * global_proto[label.item()])
                features_noise = torch.stack(features_noise, dim=0)
                features_masked = features_noise * mask
                features_labels[idx1 * len(datasets_name) + idx2] = [features_masked, labels]
                ids = torch.ones(features.shape[0]) * (idx1 * len(datasets_name) + idx2)
                features_idx.append([features_masked, ids])
        loss_temp = [0 for i in range(len(datasets_name))]
        loss_kl = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        loss_info = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, kl_loss, info_loss, _ = local_test.test_inference_ablation2(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2], global_proto=global_proto, dmodel = copy.deepcopy(discriminator))
                    loss_temp[idx2] += loss
                    loss_kl[idx2] += kl_loss
                    loss_info[idx2] += info_loss
                    _, _, _, acc = local_test.test_inference_ablation2(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2], global_proto=global_proto, dmodel = copy.deepcopy(discriminator))
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('\n{:<11s} | CE loss: {:.4f} | kl loss : {:.4f}| infoNCEloss : {:.4f}| Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), loss_kl[idx] / (args.num_users // len(datasets_name)), loss_info[idx] / (args.num_users // len(datasets_name)),  acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
        loss_func = nn.CrossEntropyLoss()
        if args.cls_component == 1:
            features_label_dataset = data_utils.FeatureDataset(features_labels)
            features_label_loader = torch.utils.data.DataLoader(features_label_dataset, batch_size=args.batch, shuffle=True)  
            global_classifier.train()
            for _ in range(args.adcol_epoch):
                for x, y in features_label_loader:
                    x, y= x.to(args.device).float(), y.to(args.device).long()
                    y_pred = global_classifier(x)
                    loss1 = loss_func(y_pred, y).mean()
                    classifier_optimizer.zero_grad()
                    loss1.backward()
                    classifier_optimizer.step()
        features_dataset = data_utils.FeatureDataset(features_idx)
        features_loader = torch.utils.data.DataLoader(features_dataset, batch_size=args.batch, shuffle=True)  
        discriminator.train()
        for _ in range(args.adcol_epoch):
            for x, y in features_loader:
                x, y = x.to(args.device).float(), y.to(args.device).long()
                y_pred = discriminator(x)
                loss2 = loss_func(y_pred, y).mean()
                discriminator_optimizer.zero_grad()
                loss2.backward()
                discriminator_optimizer.step()
    return train_loss, accuracy_list, datasets_name

def ablation3(args, train_loader_list, test_loader_list):
    loader_size = [len(train_loader.dataset) for train_loader in train_loader_list]
    client_weights = [item / sum(loader_size) for item in loader_size]
    if args.dataset == "digit":
        global_model = adcol_model().to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
    elif args.dataset == "office":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["amazon", "caltech", "dslr", "webcam"]
    elif args.dataset == "PACS":
        global_model = adcol_model(num_classes=args.num_classes).to(args.device)
        local_models = [copy.deepcopy(global_model).to(args.device) for idx in range(args.num_users)]
        discriminator = Discriminator(global_model, args.num_users).to(args.device)
        datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
    global_classifier = classifier_model(args, args.num_classes).to(args.device)
    train_loss = {item: [] for item in datasets_name}
    accuracy_list = {item: [] for item in datasets_name}
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9)
    classifier_optimizer = torch.optim.SGD(global_classifier.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9)
    global_proto = {}
    best_acc = 0
    for round in tqdm(range(args.iters)):
        print(f'\n | Global Training Round : {round} |\n')
        features_idx = []
        features_labels = []
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                local_node = LocalUpdate(args=args)
                local_models[idx1 * len(datasets_name) + idx2], features_label= local_node.update_weights_ours_debug(round, model=copy.deepcopy(local_models[idx1 * len(datasets_name) + idx2]), discriminator=copy.deepcopy(discriminator), classifier_model=copy.deepcopy(global_classifier), train_loader=train_loader_list[idx1 * len(datasets_name) + idx2],global_proto=global_proto, momentum = args.momentum)
                features_labels.append([features_label[0], features_label[1]])
        local_protos = []
        num = []
        num_list_ = Counter()
        for idx1 in range(args.num_users // len(datasets_name)):
            for idx2 in range(len(datasets_name)):
                avg_proto, num_list = local_node.compute_proto_debug(train_loader=features_labels[idx1 * len(datasets_name) + idx2])
                local_protos.append(avg_proto)
                num_list_ += num_list
                num.append(num_list)
        global_proto = get_mean(args, local_protos, num)

        if args.mix_mode == 1:
            for idx1 in range(args.num_users // len(datasets_name)):
                for idx2 in range(len(datasets_name)):
                    features = features_labels[idx1 * len(datasets_name) + idx2][0]
                    labels = features_labels[idx1 * len(datasets_name) + idx2][1]
                    features_labels[idx1 * len(datasets_name) + idx2] = [features, labels]
                    ids = torch.ones(features.shape[0]) * (idx1 * len(datasets_name) + idx2)
                    features_idx.append([features, ids])
        elif args.mix_mode == 2:
            for idx1 in range(args.num_users // len(datasets_name)):
                for idx2 in range(len(datasets_name)):
                    features = features_labels[idx1 * len(datasets_name) + idx2][0]
                    labels = features_labels[idx1 * len(datasets_name) + idx2][1]
                    noise = torch.randn_like(features)
                    features_noise = features + noise
                    features_labels[idx1 * len(datasets_name) + idx2] = [features_noise, labels]
                    ids = torch.ones(features.shape[0]) * (idx1 * len(datasets_name) + idx2)
                    features_idx.append([features_noise, ids])
        elif args.mix_mode == 3:
            for idx1 in range(args.num_users // len(datasets_name)):
                for idx2 in range(len(datasets_name)):
                    features = features_labels[idx1 * len(datasets_name) + idx2][0]
                    labels = features_labels[idx1 * len(datasets_name) + idx2][1]
                    size = features.shape[1]
                    mask = generate_random_tensor(size, p=0.9).to(args.device)
                    lam = np.round(args.uniform_left + args.uniform_right * np.random.random(), 2)
                    features_noise = []
                    for idx, label in enumerate(labels):
                        features_noise.append(lam * features[idx] + (1 - lam) * global_proto[label.item()])
                    features_noise = torch.stack(features_noise, dim=0)
                    features_masked = features_noise * mask
                    features_labels[idx1 * len(datasets_name) + idx2] = [features_masked, labels]
                    ids = torch.ones(features.shape[0]) * (idx1 * len(datasets_name) + idx2)
                    features_idx.append([features_masked, ids])
        loss_temp = [0 for i in range(len(datasets_name))]
        loss_kl = [0 for i in range(len(datasets_name))]
        acc_temp = [0 for i in range(len(datasets_name))]
        loss_info = [0 for i in range(len(datasets_name))]
        for idx1 in range(args.num_users // len(datasets_name)):
            with torch.no_grad():
                for idx2 in range(len(datasets_name)):
                    local_test = LocalTest(args=args)
                    loss, kl_loss, info_loss, _ = local_test.test_inference_ablation2(args, local_models[idx1 * len(datasets_name) + idx2], train_loader_list[idx1 * len(datasets_name) + idx2], global_proto=global_proto, dmodel = copy.deepcopy(discriminator))
                    loss_temp[idx2] += loss
                    loss_kl[idx2] += kl_loss
                    loss_info[idx2] += info_loss
                    _, _, _, acc = local_test.test_inference_ablation2(args, local_models[idx1 * len(datasets_name) + idx2], test_loader_list[idx2], global_proto=global_proto, dmodel = copy.deepcopy(discriminator))
                    acc_temp[idx2] += acc
        for idx in range(len(datasets_name)):
            print('\n{:<11s} | CE loss: {:.4f} | kl loss : {:.4f}| infoNCEloss : {:.4f}| Test Acc: {:.4f}'.format(datasets_name[idx], loss_temp[idx] / (args.num_users // len(datasets_name)), loss_kl[idx] / (args.num_users // len(datasets_name)), loss_info[idx] / (args.num_users // len(datasets_name)),  acc_temp[idx] / (args.num_users // len(datasets_name))))
            train_loss[datasets_name[idx]].append(copy.deepcopy(loss_temp[idx] / (args.num_users // len(datasets_name))))
            accuracy_list[datasets_name[idx]].append(copy.deepcopy(acc_temp[idx] / (args.num_users // len(datasets_name))))
        if np.average(acc_temp) > best_acc:
            file_path = f'pkl/debug/{args.dataset}/{args.mode}_{args.seed}_features_labels_idx_NoiseType{args.mix_mode}.pkl'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            for i in range(args.num_users):
                features_labels[i][0] = features_labels[i][0].to("cpu")
                features_labels[i][1] = features_labels[i][1].to("cpu")
            with open(file_path, 'wb') as f:
                pickle.dump(features_labels, f)
            for i in range(args.num_users):
                features_labels[i][0] = features_labels[i][0].to(args.device)
                features_labels[i][1] = features_labels[i][1].to(args.device)
            best_acc = np.average(acc_temp) 
        features_label_dataset = data_utils.FeatureDataset(features_labels)
        features_label_loader = torch.utils.data.DataLoader(features_label_dataset, batch_size=args.batch, shuffle=True)  
        loss_func = nn.CrossEntropyLoss()
        global_classifier.train()
        for _ in range(args.adcol_epoch):
            for x, y in features_label_loader:
                x, y= x.to(args.device).float(), y.to(args.device).long()
                y_pred = global_classifier(x)
                loss1 = loss_func(y_pred, y).mean()
                classifier_optimizer.zero_grad()
                loss1.backward()
                classifier_optimizer.step()
        features_dataset = data_utils.FeatureDataset(features_idx)
        features_loader = torch.utils.data.DataLoader(features_dataset, batch_size=args.batch, shuffle=True)  
        discriminator.train()
        for _ in range(args.adcol_epoch):
            for x, y in features_loader:
                x, y = x.to(args.device).float(), y.to(args.device).long()
                y_pred = discriminator(x)
                loss2 = loss_func(y_pred, y).mean()
                discriminator_optimizer.zero_grad()
                loss2.backward()
                discriminator_optimizer.step()
    
    return train_loss, accuracy_list, datasets_name

def set_seed(args):
    random.seed(args.seed)  
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  
    args.device = args.device if torch.cuda.is_available() else 'cpu'

def fed_main(args):
    torch.cuda.set_device(args.device)
    if args.exp == 1:
        datasets = ["digit", "office", "PACS"]
        seeds = [0,1,2]
        args.iters = 100
        args.lr = 0.01
        for dataset in datasets:
            args.mode = modes
            args.dataset = dataset
            args.save_path = f"../result/exp{args.exp}/{args.dataset}/resnet50/"
            os.makedirs(args.save_path, exist_ok=True)
            for seed in seeds:
                data = collections.defaultdict(list)
                args.seed = seed
                set_seed(args)
                if args.dataset == "office":
                    args.num_classes = 10
                    args.num_users = 4
                    args.size = 64
                    args.batch = 32
                    args.wk_iters = 10
                    args.adcol_epoch = 3
                    args.adcol_mu = 0.1
                    args.adcol_beta = 0.1
                    datasets_name = ["amazon", "caltech", "dslr", "webcam"]
                    train_loader_list, test_loader_list = prepare_data_office_feature_noniid(args=args)
                    num_list = compute_num_list(args, train_loader_list)
                elif args.dataset == "digit":
                    args.num_classes = 10
                    args.num_users = 5
                    args.size = 28
                    args.batch = 64
                    args.wk_iters = 5
                    args.adcol_mu = 0.7
                    args.adcol_beta = 0.3
                    args.adcol_epoch = 1
                    datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
                    train_loader_list, test_loader_list = prepare_data_digit_feature_noniid(args=args)
                    num_list = compute_num_list(args, train_loader_list)
                elif args.dataset == "PACS":
                    args.num_classes = 7
                    args.num_users = 4
                    args.size = 64
                    args.batch = 32
                    args.wk_iters = 10
                    args.adcol_mu = 0.1
                    args.adcol_beta = 0.1
                    args.adcol_epoch = 3
                    datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
                    train_loader_list, test_loader_list = prepare_data_PACS_feature_noniid(args=args)
                    num_list = compute_num_list(args, train_loader_list)
                print(args)
                save_path = args.save_path + f"{args.mode}"
                os.makedirs(save_path, exist_ok=True)
                excel_file = save_path + "/test_acc.xlsx"
                if not os.path.exists(excel_file):
                    pd.DataFrame({"dataset": datasets_name}).to_excel(excel_file, index=False)
                if args.mode == "fedBN":
                    train_loss, accuracy_list, datasets_name = fedBN(args, train_loader_list, test_loader_list)
                elif args.mode == "fedavg":
                    train_loss, accuracy_list, datasets_name = fedavg(args, train_loader_list, test_loader_list)
                elif args.mode == "SingleSet":
                    train_loss, accuracy_list, datasets_name = SingleSet(args, train_loader_list, test_loader_list)
                elif args.mode == "fedprox":
                    train_loss, accuracy_list, datasets_name = fedProx(args, train_loader_list, test_loader_list)
                elif args.mode == "fedproto":
                    train_loss, accuracy_list, datasets_name = fedproto(args, train_loader_list, test_loader_list, num_list)
                elif args.mode == "fedrep":
                    train_loss, accuracy_list, datasets_name = fedrep(args, train_loader_list, test_loader_list)
                elif args.mode == "perfedavg":
                    train_loss, accuracy_list, datasets_name = perfedavg(args, train_loader_list, test_loader_list)
                elif args.mode == "moon":
                    train_loss, accuracy_list, datasets_name = moon(args, train_loader_list, test_loader_list)
                elif args.mode == "adcol":
                    train_loss, accuracy_list, datasets_name = adcol(args, train_loader_list, test_loader_list)
                elif args.mode == "fedpcl":
                    train_loss, accuracy_list, datasets_name = fedpcl(args, train_loader_list, test_loader_list, num_list)
                elif args.mode == "fed_heal":
                    args.lr = 1e-3
                    if args.dataset == "digit":
                        args.beta = 0.4
                        args.tau = 0.3
                    elif args.dataset == "office":
                        args.beta = 0.4
                        args.tau = 0.4
                    elif args.dataset == "PACS":
                        args.beta = 0.4
                        args.tau = 0.4
                    excel_file = save_path + f"/test_acc_lr{args.lr}_tau{args.tau}_beta{args.beta}.xlsx"
                    if not os.path.exists(excel_file):
                        pd.DataFrame({"dataset": datasets_name}).to_excel(excel_file, index=False)
                    train_loss, accuracy_list, datasets_name = FedHEAL(args, train_loader_list, test_loader_list)
                elif args.mode == "ours":
                    train_loss, accuracy_list, datasets_name = ours(args, train_loader_list, test_loader_list)
                test_avg_acc = []
                df = pd.DataFrame(accuracy_list)
                df['average'] = df.mean(axis=1)
                for idx in range(len(accuracy_list[datasets_name[0]])):
                    test_avg_acc.append(sum([accuracy_list[dataset][idx] for dataset in datasets_name]) / len(datasets_name)) 
                max_value = np.max(test_avg_acc)
                indices = np.where(np.array(test_avg_acc) == max_value)[0]
                best_step = indices[-1]
                for dataset_name in datasets_name:
                    data[f"seed:{seed}"].append(accuracy_list[dataset_name][best_step])
                with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                    existing_data = pd.read_excel(excel_file)
                    new_data = pd.DataFrame(data)

                    combined_data = pd.concat([existing_data, new_data], axis=1)
                    combined_data.to_excel(writer, index=False)
    elif args.exp == 2:
        args.dataset = "office"
        args.mode = "ablation1"
        args.seed = 0
        args.iters = 100
        args.lr = 0.01
        combinations = [1,2,3,4]
        args.save_path = f"../result/exp{args.exp}/{args.dataset}/"
        os.makedirs(args.save_path, exist_ok=True)
        for combination in combinations:
            args.loss_component = combination
            data = collections.defaultdict(list)
            set_seed(args)
            if args.dataset == "office":
                args.num_classes = 10
                args.num_users = 4
                args.size = 64
                args.batch = 32
                args.wk_iters = 10
                args.adcol_epoch = 3
                args.adcol_mu = 0.1
                args.adcol_beta = 0.1
                datasets_name = ["amazon", "caltech", "dslr", "webcam"]
                train_loader_list, test_loader_list = prepare_data_office_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            elif args.dataset == "digit":
                args.num_classes = 10
                args.num_users = 5
                args.size = 28
                args.batch = 64
                args.wk_iters = 5
                args.adcol_mu = 0.7
                args.adcol_beta = 0.3
                args.adcol_epoch = 1
                datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
                train_loader_list, test_loader_list = prepare_data_digit_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            elif args.dataset == "PACS":
                args.num_classes = 7
                args.num_users = 4
                args.size = 64
                args.batch = 32
                args.wk_iters = 10
                args.adcol_mu = 0.1
                args.adcol_beta = 0.1
                args.adcol_epoch = 3
                datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
                train_loader_list, test_loader_list = prepare_data_PACS_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            print(args)
            save_path = args.save_path + f"loss_component_{args.loss_component}"
            os.makedirs(save_path, exist_ok=True)
            excel_file = save_path + "/test_acc.xlsx"
            if not os.path.exists(excel_file):
                pd.DataFrame({"dataset": datasets_name}).to_excel(excel_file, index=False)
            if args.mode == "ablation1":
                train_loss, accuracy_list, datasets_name = ablation1(args, train_loader_list, test_loader_list)
            test_avg_acc = []
            for idx in range(len(accuracy_list[datasets_name[0]])):
                test_avg_acc.append(sum([accuracy_list[dataset][idx] for dataset in datasets_name]) / 5) 
            max_value = np.max(test_avg_acc)
            indices = np.where(np.array(test_avg_acc) == max_value)[0]
            best_step = indices[-1]
            for dataset_name in datasets_name:
                data[f"seed:{args.seed}"].append(accuracy_list[dataset_name][best_step])
            with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                existing_data = pd.read_excel(excel_file)
                new_data = pd.DataFrame(data)
                combined_data = pd.concat([existing_data, new_data], axis=1)
                combined_data.to_excel(writer, index=False)
    elif args.exp == 3:
        args.dataset = "office"
        args.mode = "ablation2"
        args.seed = 0
        args.iters = 100
        args.lr = 0.01
        cls_modes = [1,2]
        args.save_path = f"../result/exp{args.exp}/{args.dataset}/"
        os.makedirs(args.save_path, exist_ok=True)
        for cls_mode in cls_modes:
            args.cls_component = cls_mode
            data = collections.defaultdict(list)
            set_seed(args)
            if args.dataset == "office":
                args.num_classes = 10
                args.num_users = 4
                args.size = 64
                args.batch = 32
                args.wk_iters = 10
                args.adcol_epoch = 3
                args.adcol_mu = 0.1
                args.adcol_beta = 0.1
                datasets_name = ["amazon", "caltech", "dslr", "webcam"]
                train_loader_list, test_loader_list = prepare_data_office_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            elif args.dataset == "digit":
                args.num_classes = 10
                args.num_users = 5
                args.size = 28
                args.batch = 64
                args.wk_iters = 5
                args.adcol_mu = 0.7
                args.adcol_beta = 0.3
                args.adcol_epoch = 1
                datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
                train_loader_list, test_loader_list = prepare_data_digit_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            elif args.dataset == "PACS":
                args.num_classes = 7
                args.num_users = 4
                args.size = 64
                args.batch = 32
                args.wk_iters = 10
                args.adcol_mu = 0.1
                args.adcol_beta = 0.1
                args.adcol_epoch = 3
                datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
                train_loader_list, test_loader_list = prepare_data_PACS_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            print(args)
            save_path = args.save_path + f"cls_strategy_{args.cls_component}"
            os.makedirs(save_path, exist_ok=True)
            excel_file = save_path + "/test_acc.xlsx"
            if not os.path.exists(excel_file):
                pd.DataFrame({"dataset": datasets_name}).to_excel(excel_file, index=False)
            if args.mode == "ablation2":
                train_loss, accuracy_list, datasets_name = ablation2(args, train_loader_list, test_loader_list)
            test_avg_acc = []
            for idx in range(len(accuracy_list[datasets_name[0]])):
                test_avg_acc.append(sum([accuracy_list[dataset][idx] for dataset in datasets_name]) / 5) 
            max_value = np.max(test_avg_acc)
            indices = np.where(np.array(test_avg_acc) == max_value)[0]
            best_step = indices[-1]
            for dataset_name in datasets_name:
                data[f"seed:{args.seed}"].append(accuracy_list[dataset_name][best_step])
            with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                existing_data = pd.read_excel(excel_file)
                new_data = pd.DataFrame(data)
                combined_data = pd.concat([existing_data, new_data], axis=1)
                combined_data.to_excel(writer, index=False)
    elif args.exp == 4:
        args.dataset = "office"
        args.mode = "ablation3"
        args.seed = 0
        args.iters = 100
        args.lr = 0.01
        mix_modes = [1, 2, 3]
        args.save_path = f"../result/exp{args.exp}/{args.dataset}/"
        os.makedirs(args.save_path, exist_ok=True)
        for mix_mode in mix_modes:
            args.mix_mode = mix_mode
            data = collections.defaultdict(list)
            set_seed(args)
            if args.dataset == "office":
                args.num_classes = 10
                args.num_users = 4
                args.size = 64
                args.batch = 32
                args.wk_iters = 10
                args.adcol_epoch = 3
                args.adcol_mu = 0.1
                args.adcol_beta = 0.1
                datasets_name = ["amazon", "caltech", "dslr", "webcam"]
                train_loader_list, test_loader_list = prepare_data_office_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            elif args.dataset == "digit":
                args.num_classes = 10
                args.num_users = 5
                args.size = 28
                args.batch = 64
                args.wk_iters = 5
                args.adcol_mu = 0.7
                args.adcol_beta = 0.3
                args.adcol_epoch = 1
                datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
                train_loader_list, test_loader_list = prepare_data_digit_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            elif args.dataset == "PACS":
                args.num_classes = 7
                args.num_users = 4
                args.size = 64
                args.batch = 32
                args.wk_iters = 10
                args.adcol_mu = 0.1
                args.adcol_beta = 0.1
                args.adcol_epoch = 3
                datasets_name = ["art_painting", "cartoon", "photo", "sketch"]
                train_loader_list, test_loader_list = prepare_data_PACS_feature_noniid(args=args)
                num_list = compute_num_list(args, train_loader_list)
            print(args)
            save_path = args.save_path + f"noisetype_{args.mix_mode}"
            os.makedirs(save_path, exist_ok=True)
            excel_file = save_path + "/test_acc.xlsx"
            if not os.path.exists(excel_file):
                pd.DataFrame({"dataset": datasets_name}).to_excel(excel_file, index=False)
            if args.mode == "ablation3":
                train_loss, accuracy_list, datasets_name = ablation3(args, train_loader_list, test_loader_list)
            test_avg_acc = []
            for idx in range(len(accuracy_list[datasets_name[0]])):
                test_avg_acc.append(sum([accuracy_list[dataset][idx] for dataset in datasets_name]) / 5) 
            max_value = np.max(test_avg_acc)
            indices = np.where(np.array(test_avg_acc) == max_value)[0]
            best_step = indices[-1]
            for dataset_name in datasets_name:
                data[f"seed:{args.seed}"].append(accuracy_list[dataset_name][best_step])
            with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                existing_data = pd.read_excel(excel_file)
                new_data = pd.DataFrame(data)
                combined_data = pd.concat([existing_data, new_data], axis=1)
                combined_data.to_excel(writer, index=False)
if __name__ == "__main__":
    args = args_parser()
    args.seed = 1
    fed_main(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

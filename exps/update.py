import torch
from torch import nn
import copy
from util import PerAvgOptimizer
import torch.nn.functional as F
import time
from collections import Counter
from torch.utils.data.dataset import Dataset
import numpy as np
import random
import pickle
from contextlib import contextmanager, nullcontext


@contextmanager
def _preserve_feature_probe_state(model):
    saved_buffers = {
        name: buffer.detach().clone() for name, buffer in model.named_buffers()
    }
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state().clone()
    cuda_rng_states = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    try:
        yield
    finally:
        current_buffers = dict(model.named_buffers())
        if current_buffers.keys() != saved_buffers.keys():
            raise RuntimeError("model buffers changed structurally during feature probe")
        with torch.no_grad():
            for name, saved in saved_buffers.items():
                current = current_buffers[name]
                if current.shape != saved.shape or current.dtype != saved.dtype:
                    raise RuntimeError(
                        f"model buffer {name!r} changed shape or dtype during feature probe"
                    )
                current.copy_(saved)

        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
from tqdm import tqdm
from torch.autograd import grad

class ConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, global_protos=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_count = contrast_count
            anchor_feature = torch.zeros_like(contrast_feature)
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        for i in range(batch_size*anchor_count):
            anchor_feature[i, :] = global_protos[labels[i%batch_size].item()]

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class MixupDataset_norm(Dataset):
    def __init__(self, mean, fs_all: list, args):
        self.data = []
        self.labels = []
        self.means = mean
        self.num_classes = args.num_classes
        self.device = args.device
        self.crt_feat_num = args.crt_feat_num
        self.fs_all = fs_all
        self.fs_len = len(fs_all)
        self.args = args

        self.__mixup_syn_feat_pure_rand_norm__()

    def __mixup_syn_feat_pure_rand_norm__(self):
        num = self.crt_feat_num
        l = self.args.uniform_left
        r_arg = self.args.uniform_right - l
        for cls in range(self.num_classes):
            fs_shuffle_idx = torch.randperm(self.fs_len)
            for i in range(num):
                lam = np.round(l + r_arg * np.random.random(), 2)
                neg_f = self.fs_all[fs_shuffle_idx[i]]
                mixup_f = lam * self.means[cls] + (1 - lam) * F.normalize(neg_f.view(1, -1), dim=1).view(-1)
                self.data.append(mixup_f)
            self.labels += [cls]*num
        self.data = torch.stack(self.data).to(self.device)
        self.labels = torch.tensor(self.labels).long().to(self.device)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]

def generate_random_tensor(size, p):
    assert 0 <= p <= 1, "p should be between 0 and 1"
    num_ones = int(size * p)
    tensor = torch.cat((torch.ones(num_ones), torch.zeros(size - num_ones)))
    tensor = tensor[torch.randperm(tensor.size(0))]
    return tensor.view(size)

class UVReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_classes = args.num_classes
        self.soft = nn.Softmax(dim=1)   
        tester = torch.eye(self.n_classes)
        self.batch_gamma = tester.std(dim=0).mean().item()

    def forward(self, X, Y, pro1):
        if len(X.shape) > 2:
            X = torch.reshape(X, (X.shape[0], np.prod(X.shape[1:])))
            Y = torch.reshape(Y, (Y.shape[0], np.prod(Y.shape[1:])))
        pdist_x = torch.pdist(pro1, p=2).pow(2)
        sigma_unif_x = torch.median(pdist_x[pdist_x != 0])
        unif_loss = pdist_x.mul(-1/sigma_unif_x).exp().mean()
        logsoft_out = self.soft(X)
        logsoft_out_std = logsoft_out.std(dim=0)
        std_loss = torch.mean(F.relu(self.batch_gamma - logsoft_out_std))
        loss = (
            self.args.std_coeff * std_loss
            + self.args.unif_coeff * unif_loss
        )
        return loss, np.array([0, np.round(std_loss.item(), 5),
                               0, 0,
                               np.round(unif_loss.item(), 10)])

class LocalUpdate(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.old_model = None
        random.seed(args.seed)  
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)  
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        args.device = args.device if torch.cuda.is_available() else 'cpu'
        self.criterion_CE = nn.CrossEntropyLoss().to(self.device)
        self.criterion_MSE = nn.MSELoss().to(self.device)
    
    def criterion_correction(self, features, labels, protos_dict):
        bs = features.shape[0]
        target = torch.ones(bs)
        data = torch.zeros(bs).to(self.device)
        for i in range(bs):
            label = labels[i].item()
            data[i] = torch.cosine_similarity(features[i], torch.tensor(protos_dict[label]).to(self.device), dim=0)
        target = torch.tensor(np.array(target)).to(self.device)
        loss = 2 * self.criterion_MSE(data, target)
        return loss

    def criterion_InfoNCE(self, features, labels, protos_dict):
        temperature = self.args.tau
        bs = features.shape[0]
        protos_list = []
        num_protos = 0
        protos_key_list = []
        for label in protos_dict.keys():
            if protos_dict[label].ndim == 1:
                protos_list_label = protos_dict[label].reshape(1,-1)
            else:
                protos_list_label = protos_dict[label]

            for i in range(len(protos_list_label)):
                protos_list.append(protos_list_label[i])
                protos_key_list.append(label)

        num_protos += len(protos_list)
        mask = np.zeros((bs, num_protos))

        protos_list = torch.tensor(np.array(protos_list)).to(self.device)
        protos_key_list = np.array(protos_key_list)
        logits = torch.zeros(bs, num_protos).to(self.device)

        for i in range(bs):
            label = labels[i].item()
            mask[i][np.where(protos_key_list == label)] = 1
            logits[i] = torch.cosine_similarity(features[i].unsqueeze(0), protos_list, dim=1)
        mask = torch.tensor(mask).to(self.device)

        logits = logits / temperature

        exp_logits = torch.exp(logits)
        sum_exp_logits = exp_logits.sum(1, keepdim=True)
        pos_logits = exp_logits * mask
        sum_pos_logits = pos_logits.sum(1, keepdim=True)

        loss = - torch.log(sum_pos_logits/sum_exp_logits)
        return loss.mean()

    def criterion_InfoNCE_alpha(self, features, labels, protos_dict):
        temperature = self.args.tau
        alpha = self.args.alpha
        bs = features.shape[0]
        protos_list = []
        num_protos = 0
        protos_key_list = []
        for label in protos_dict.keys():
            protos_dict[label] = np.array(protos_dict[label])
            if protos_dict[label].ndim == 1:
                protos_list_label = protos_dict[label].reshape(1, -1)
            else:
                protos_list_label = protos_dict[label]
            for i in range(len(protos_list_label)):
                protos_list.append(protos_list_label[i])
                protos_key_list.append(label)
        num_protos += len(protos_list)
        mask = np.zeros((bs, num_protos))
        protos_list = torch.tensor(np.array(protos_list)).to(self.device)
        protos_key_list = np.array(protos_key_list)
        logits = torch.zeros(bs, num_protos).to(self.device)
        for i in range(bs):
            label = labels[i].item()
            mask[i][np.where(protos_key_list == label)] = 1
            logits[i] = torch.cosine_similarity(features[i].unsqueeze(0), protos_list, dim=1)
        mask = torch.tensor(mask).to(self.device)
        logits = logits.pow(alpha)
        logits = logits / temperature
        exp_logits = torch.exp(logits)
        sum_exp_logits = exp_logits.sum(1, keepdim=True)
        pos_logits = exp_logits * mask
        sum_pos_logits = pos_logits.sum(1, keepdim=True)
        loss = - torch.log(sum_pos_logits / sum_exp_logits)
        return loss.mean()

    def update_weights(self, model, train_loader):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        for step in range(self.args.wk_iters):
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                _, logits = model(images)
                loss = self.criterion_CE(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return copy.deepcopy(model)
    
    def update_weights_FedAS(self, model, train_loader):
        if not self.old_model:
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
            for step in range(self.args.wk_iters):
                train_iter = iter(train_loader)
                for batch_idx in range(len(train_iter)):
                    images, labels = next(train_iter)
                    images, labels = images.to(self.device).float(), labels.to(self.device).long()
                    _, logits = model(images)
                    loss = self.criterion_CE(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            self.old_model = copy.deepcopy(model)
            return copy.deepcopy(model), len(train_loader.dataset)
        else:
            old_prototypes = [[] for _ in range(self.args.num_classes)]
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                proto_batch, _ = self.old_model(images)
                for proto, y in zip(proto_batch, labels):
                    old_prototypes[y.item()].append(proto.detach().clone())
            old_mean_prototypes = []
            for class_prototypes in old_prototypes:
                if not class_prototypes == []:
                    stacked_protos = torch.stack(class_prototypes)
                    mean_proto = torch.mean(stacked_protos, dim=0)
                    old_mean_prototypes.append(mean_proto)
                else:
                    old_mean_prototypes.append(None)
            
            model.train()
            alignment_optimizer = torch.optim.SGD(model.features.parameters(), lr=0.01)
            alignment_loss_fn = torch.nn.MSELoss()
            train_iter = iter(train_loader)
            for _ in range(1):
                for batch_idx in range(len(train_iter)):
                    images, labels = next(train_iter)
                    images, labels = images.to(self.device).float(), labels.to(self.device).long()
                    global_proto_batch = model.features(images)
                    loss = 0
                    loss_list = []
                    for label in labels.unique():
                        if old_mean_prototypes[label.item()] is not None:
                            feats = global_proto_batch[labels == label]
                            proto = old_mean_prototypes[label.item()].unsqueeze(0).repeat(feats.size(0), 1)
                            loss_list.append(alignment_loss_fn(feats, proto))
                    loss = torch.stack(loss_list).mean()
                    alignment_optimizer.zero_grad()
                    loss.backward()
                    alignment_optimizer.step()
            for new_param, old_param in zip(model.features.parameters(), self.old_model.features.parameters()):
                old_param.data = new_param.data.clone()
            self.old_model.train()
            optimizer = torch.optim.SGD(self.old_model.parameters(), lr=self.args.lr)
            for step in range(self.args.wk_iters):
                train_iter = iter(train_loader)
                for batch_idx in range(len(train_iter)):
                    images, labels = next(train_iter)
                    images, labels = images.to(self.device).float(), labels.to(self.device).long()
                    _, logits = self.old_model(images)
                    loss = self.criterion_CE(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            self.old_model.eval()
            train_iter = iter(train_loader)
            fim_trace_sum = 0
            for batch_idx in range(len(train_iter)):
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                _, outputs = self.old_model(images)
                nll = -torch.nn.functional.log_softmax(outputs, dim=1)[range(len(labels)), labels].mean()

                grads = grad(nll, self.old_model.parameters())

                for g in grads:
                    fim_trace_sum += torch.sum(g ** 2).detach()

            return copy.deepcopy(self.old_model), fim_trace_sum
    
    def update_weights_fedProx(self, model, train_loader):
        server_model = copy.deepcopy(model)
        model.train()
        server_model.eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        for step in range(self.args.wk_iters):
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                optimizer.zero_grad()
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                rep, output = model(images)
                loss = self.criterion_CE(output, labels)
                if batch_idx>0:
                    w_diff = torch.tensor(0., device=self.args.device)
                    for w, w_t in zip(server_model.parameters(), model.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                    loss += self.args.mu / 2. * w_diff
                loss.backward()
                optimizer.step()
        return copy.deepcopy(model)

    def update_weights_perfedavg(self, model, train_loader):
        model.train()
        optimizer = PerAvgOptimizer(model.parameters(), lr=self.args.lr)
        for step in range(self.args.wk_iters):
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                if batch_idx % 2 == 0 and batch_idx != len(train_iter) - 1:
                    temp_model = copy.deepcopy(list(model.parameters()))
                    optimizer.zero_grad()
                    images, labels = next(train_iter)
                    images, labels = images.to(self.device).float(), labels.to(self.device).long()
                    rep, logits = model(images)
                    loss = self.criterion_CE(logits, labels)
                    loss.backward()
                    optimizer.step()
                elif batch_idx % 2 == 0 and batch_idx == len(train_iter) - 1:
                    continue
                else:
                    optimizer.zero_grad()
                    images, labels = next(train_iter)
                    images, labels = images.to(self.device).float(), labels.to(self.device).long()
                    rep, logits = model(images)
                    loss = self.criterion_CE(logits, labels)
                    loss.backward()
                    for old_param, new_param in zip(model.parameters(), temp_model):
                        old_param.data = new_param.data.clone()
                    optimizer.step(beta=self.args.beta)
        return copy.deepcopy(model)

    def update_weights_fedrep(self, model, train_loader):
        model.train()
        optimizer_feat = torch.optim.SGD(model.features.parameters(), lr=self.args.lr)
        optimizer_class = torch.optim.SGD(model.classifier.parameters(), lr=self.args.lr)
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True  
        for step in range(self.args.wk_iters):
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                optimizer_class.zero_grad()
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                rep, output = model(images)
                loss = self.criterion_CE(output, labels)
                loss.backward()
                optimizer_class.step()
        for param in model.features.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = False
        train_iter = iter(train_loader)
        for batch_idx in range(len(train_iter)):
            optimizer_feat.zero_grad()
            images, labels = next(train_iter)
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            rep, output = model(images)
            loss = self.criterion_CE(output, labels)
            loss.backward()
            optimizer_feat.step()
        return copy.deepcopy(model)

    def update_weights_proto(self, model, train_loader, global_proto):
        server_model = copy.deepcopy(model)
        model.train()        
        server_model.eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        loss_mse = nn.MSELoss()
        for step in range(self.args.wk_iters):
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                optimizer.zero_grad()
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                proto, output = model(images)
                loss1 = self.criterion_CE(output, labels)
                if len(global_proto) == 0:
                    loss2 = loss1 * 0
                else:
                    proto_new = copy.deepcopy(proto.detach())
                    i = 0
                    for label in labels:
                        if label.item() in global_proto.keys():
                            proto_new[i, :] = global_proto[label.item()].data
                        i += 1
                    loss2 = loss_mse(proto_new, proto)
                loss = loss1 + loss2 * self.args.ld
                loss.backward()
                optimizer.step()
        protos = {}
        model.eval()
        train_iter = iter(train_loader)
        for batch_idx in range(len(train_iter)):
            images, labels = next(train_iter)
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            proto, output = model(images)
            for i in range(len(labels)):
                if labels[i].item() in protos:
                    protos[labels[i].item()].append(proto[i,:].detach().clone())
                else:
                    protos[labels[i].item()] = [proto[i,:].detach().clone()]
        averaged_protos = {}
        for label, proto_list in protos.items():
            averaged_protos[label] = torch.mean(torch.stack(proto_list), dim=0)
        return copy.deepcopy(model), averaged_protos

    def update_weights_moon(self, model, train_loader):
        server_model = copy.deepcopy(model)
        model.train()
        server_model.eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        for step in range(self.args.wk_iters):
            batch_loss = []
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                optimizer.zero_grad()
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                rep, output = model(images)
                loss1 = self.criterion_CE(output, labels)
                rep_global = server_model(images)[0].detach()
                rep_old = self.old_model(images)[0].detach()
                loss2 = - torch.log(torch.exp(F.cosine_similarity(rep, rep_global) / self.args.tau) / (torch.exp(F.cosine_similarity(rep, rep_global) / self.args.tau) + torch.exp(F.cosine_similarity(rep, rep_old) / self.args.tau)))
                loss = loss1 + torch.mean(loss2) * self.args.ld
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
        self.old_model = copy.deepcopy(model)
        return copy.deepcopy(model)

    def update_weights_adcol(self, model, discriminator,  train_loader):
        model.train()
        features = []
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        for step in range(self.args.wk_iters):
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                rep, logits = model(images)
                loss1 = self.criterion_CE(logits, labels)
                client_index = discriminator(rep)
                client_index_softmax = F.log_softmax(client_index, dim=-1)
                target_index = torch.full(client_index.shape, 1 / self.args.num_users).to(
                    self.device
                )
                target_index_softmax = F.softmax(target_index, dim=-1)
                kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(self.device)
                kl_loss = kl_loss_func(client_index_softmax, target_index_softmax)           
                loss = loss1 + self.args.adcol_beta * kl_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        train_iter = iter(train_loader)
        for batch_idx in range(len(train_iter)):
            images, labels = next(train_iter)
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            rep, logits = model(images)
            features.append(rep.detach().clone().cpu())
        features = torch.cat(features, dim=0)
        return copy.deepcopy(model), features
  
    def update_weights_ours_debug(self, round, model, discriminator, classifier_model, train_loader, global_proto, num_domains=None, momentum = 1, global_G=None, lambda_G=0.0, kl_weight_multiplier=1.0, ce_weight_multiplier=1.0, lambda_mixup=0.0, feature_bank=None, lambda_FB=0.0, feature_loader=None, preserve_feature_probe_state=False, use_mean_dispersion=False, global_mean_md=None, global_dispersion=None, mean_dispersion_eps=1e-6, lambda_md=None, lambda_input_mixup=0.0, input_mixup_alpha=0.2, input_mixup_seed=0):
        if feature_loader is None:
            feature_loader = train_loader
        if num_domains is None:
            num_domains = self.args.num_users
        model_state_dict = model.classifier.state_dict()
        classifier_model_state_dict = classifier_model.classifier.state_dict()
        with torch.no_grad():
            for key in model_state_dict:
                if "bn" not in key:
                    model_param = model_state_dict[key]
                    classifier_model_param = classifier_model_state_dict[key]
                    model_state_dict[key].copy_((1 - momentum) * model_param + momentum * classifier_model_param)
        model.classifier.load_state_dict(model_state_dict)
        model.train()
        features = []

        support_threshold = getattr(self.args, "mixup_support_threshold", 0)
        client_sample_indices = getattr(train_loader.sampler, "indices", None)
        if client_sample_indices is not None and hasattr(train_loader.dataset, "labels"):
            client_labels_this_round = np.asarray(train_loader.dataset.labels)[client_sample_indices]
            class_counts = np.bincount(client_labels_this_round, minlength=self.args.num_classes)
            missing_classes = {c for c in range(self.args.num_classes) if class_counts[c] <= support_threshold}
        else:
            missing_classes = set()

        if feature_bank is not None and feature_bank[0] is not None and feature_bank[0].shape[0] > 0:
            fb_features, fb_labels = feature_bank
            fb_features = fb_features.detach().to(self.device)
            fb_labels = fb_labels.detach().to(self.device)
            fb_features_n = F.normalize(fb_features, dim=1)
        else:
            fb_features_n = None
            fb_labels = None

        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        mix_rng = np.random.RandomState(input_mixup_seed) if lambda_input_mixup > 0 else None
        for step in range(self.args.wk_iters):
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                rep, logits = model(images)
                CE_loss = self.criterion_CE(logits, labels)
                if use_mean_dispersion:
                    if not global_mean_md or not global_dispersion:
                        loss1 = 0
                    else:
                        proto_dim = next(iter(global_mean_md.values())).shape[0]
                        num_classes = self.args.num_classes
                        mean_matrix = torch.zeros((num_classes, proto_dim), device=self.device)
                        disp_matrix = torch.ones((num_classes, proto_dim), device=self.device)
                        present_mask = torch.zeros(num_classes, dtype=torch.bool, device=self.device)
                        for label in global_mean_md:
                            mean_matrix[label] = global_mean_md[label]
                            disp_matrix[label] = global_dispersion[label]
                            present_mask[label] = True
                        sample_valid = present_mask[labels]
                        if sample_valid.any():
                            diff_sq = (rep.unsqueeze(1) - mean_matrix.unsqueeze(0)) ** 2
                            dist = (diff_sq / (disp_matrix.unsqueeze(0) + mean_dispersion_eps)).mean(dim=2)
                            logits_md = -dist / self.args.T
                            logits_md = logits_md.masked_fill(~present_mask.unsqueeze(0), float('-inf'))
                            log_probs_md = F.log_softmax(logits_md, dim=1)
                            loss1 = -log_probs_md[torch.arange(rep.shape[0], device=self.device), labels][sample_valid].mean()
                        else:
                            loss1 = 0
                elif len(global_proto) == 0:
                    loss1 = 0
                elif self.args.domain_keyed_proto:
                    proto_dim = next(iter(global_proto.values())).shape[0]
                    num_classes = self.args.num_classes
                    global_proto_matrix = torch.zeros((num_classes, proto_dim), device=self.device)
                    present_mask = torch.zeros(num_classes, dtype=torch.bool, device=self.device)
                    for label, proto in global_proto.items():
                        global_proto_matrix[label] = proto
                        present_mask[label] = True
                    sample_valid = present_mask[labels]
                    if sample_valid.any():
                        rep_normalized = F.normalize(rep, dim=1)
                        global_proto_matrix_normalized = F.normalize(global_proto_matrix, dim=1)
                        C_y = global_proto_matrix_normalized[labels]
                        numerator = torch.exp((rep_normalized * C_y).sum(dim=1) / self.args.T)
                        sim = (rep_normalized @ global_proto_matrix.T) / self.args.T
                        sim = sim.masked_fill(~present_mask.unsqueeze(0), float('-inf'))
                        denominator = torch.exp(sim).sum(dim=1)
                        loss1 = (-torch.log(numerator / denominator))[sample_valid].mean()
                    else:
                        loss1 = 0
                else:
                    num_classes = len(global_proto)
                    proto_dim = global_proto[0].shape[0]
                    global_proto_matrix = torch.zeros((num_classes, proto_dim), device=self.device)
                    for label, proto in global_proto.items():
                        global_proto_matrix[label] = proto
                    rep_normalized = F.normalize(rep, dim=1)
                    global_proto_matrix_normalized = F.normalize(global_proto_matrix, dim=1)
                    C_y = global_proto_matrix_normalized[labels]
                    numerator = torch.exp((rep_normalized * C_y).sum(dim=1) / self.args.T)
                    denominator = torch.exp((rep_normalized @ global_proto_matrix.T) / self.args.T).sum(dim=1)
                    loss1 = -torch.log(numerator / denominator).mean()

                if global_G is None or len(global_G) == 0:
                    loss_cross = 0
                else:
                    proto_dim_G = next(iter(global_G.values())).shape[0]
                    num_classes_G = self.args.num_classes
                    G_matrix = torch.zeros((num_classes_G, proto_dim_G), device=self.device)
                    G_present_mask = torch.zeros(num_classes_G, dtype=torch.bool, device=self.device)
                    for label, proto in global_G.items():
                        G_matrix[label] = proto
                        G_present_mask[label] = True
                    G_sample_valid = G_present_mask[labels]
                    if G_sample_valid.any():
                        rep_normalized_G = F.normalize(rep, dim=1)
                        G_matrix_normalized = F.normalize(G_matrix, dim=1)
                        C_y_G = G_matrix_normalized[labels]
                        numerator_G = torch.exp((rep_normalized_G * C_y_G).sum(dim=1) / self.args.T)
                        sim_G = (rep_normalized_G @ G_matrix.T) / self.args.T
                        sim_G = sim_G.masked_fill(~G_present_mask.unsqueeze(0), float('-inf'))
                        denominator_G = torch.exp(sim_G).sum(dim=1)
                        loss_cross = (-torch.log(numerator_G / denominator_G))[G_sample_valid].mean()
                    else:
                        loss_cross = 0

                if fb_features_n is None:
                    loss_fb = rep.new_zeros(())
                else:
                    rep_n_fb = F.normalize(rep, dim=1)
                    sim_fb = (rep_n_fb @ fb_features_n.T) / self.args.T
                    pos_mask_fb = labels.unsqueeze(1).eq(fb_labels.unsqueeze(0))
                    num_pos_fb = pos_mask_fb.sum(dim=1)
                    has_pos_fb = num_pos_fb > 0
                    if has_pos_fb.any():
                        log_prob_fb = sim_fb - torch.logsumexp(sim_fb, dim=1, keepdim=True)
                        pos_log_prob_fb = (
                            (log_prob_fb * pos_mask_fb.float()).sum(dim=1)
                            / num_pos_fb.clamp_min(1).float()
                        )
                        loss_fb = -pos_log_prob_fb[has_pos_fb].mean()
                    else:
                        loss_fb = rep.new_zeros(())

                missing_valid_classes = [c for c in missing_classes if c in global_proto] if lambda_mixup > 0 else []
                if len(missing_valid_classes) >= 2:
                    partner_idx = torch.randint(0, rep.shape[0], (len(missing_valid_classes),), device=self.device)
                    partner_rep = rep[partner_idx].detach()
                    proto_stack = torch.stack([global_proto[c] for c in missing_valid_classes], dim=0).detach()
                    lam = torch.empty(len(missing_valid_classes), 1, device=self.device).uniform_(
                        self.args.mixup_lambda_min, self.args.mixup_lambda_max)
                    synth_rep = lam * proto_stack + (1 - lam) * partner_rep
                    synth_logits = model.classifier(synth_rep)
                    synth_labels = torch.tensor(missing_valid_classes, device=self.device, dtype=torch.long)
                    mixup_loss = self.criterion_CE(synth_logits, synth_labels)
                else:
                    mixup_loss = 0

                if lambda_input_mixup > 0 and images.shape[0] >= 2:
                    perm = torch.as_tensor(mix_rng.permutation(images.shape[0]), device=self.device, dtype=torch.long)
                    lam_mix = float(mix_rng.beta(input_mixup_alpha, input_mixup_alpha)) if input_mixup_alpha > 0 else 1.0
                    mixed_images = lam_mix * images + (1 - lam_mix) * images[perm]
                    _bn_modules_mix = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
                    _bn_orig_track_flags = [m.track_running_stats for m in _bn_modules_mix]
                    for _m in _bn_modules_mix:
                        _m.track_running_stats = False
                    try:
                        _, logits_mix = model(mixed_images)
                    finally:
                        for _m, _orig_flag in zip(_bn_modules_mix, _bn_orig_track_flags):
                            _m.track_running_stats = _orig_flag
                    labels_b = labels[perm]
                    input_mixup_ce = (
                        lam_mix * self.criterion_CE(logits_mix, labels)
                        + (1 - lam_mix) * self.criterion_CE(logits_mix, labels_b)
                    )
                    CE_term = (1 - lambda_input_mixup) * CE_loss + lambda_input_mixup * input_mixup_ce
                else:
                    CE_term = CE_loss

                client_index = discriminator(rep)
                client_index_softmax = F.log_softmax(client_index, dim=-1)
                target_index = torch.full(client_index.shape, 1 / num_domains).to(
                    self.device
                )
                target_index_softmax = F.softmax(target_index, dim=-1)
                kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(self.device)
                kl_loss = kl_loss_func(client_index_softmax, target_index_softmax)
                loss1_weight = lambda_md if (use_mean_dispersion and lambda_md is not None) else self.args.adcol_mu
                loss = ce_weight_multiplier * CE_term + self.args.adcol_beta * kl_weight_multiplier * kl_loss + loss1_weight * loss1 + lambda_G * loss_cross + lambda_mixup * mixup_loss + lambda_FB * loss_fb
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        labels_ = []
        with (_preserve_feature_probe_state(model) if preserve_feature_probe_state else nullcontext()):
            train_iter = iter(feature_loader)
            for batch_idx in range(len(train_iter)):
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                rep, logits = model(images)
                features.append(rep.detach().clone().cpu())
                labels_.append(labels.detach().clone().cpu())
        features = torch.cat(features, dim=0).to(self.args.device)

        labels_ = torch.cat(labels_, dim=0)
        return copy.deepcopy(model), [features, labels_]

    def update_weights_ours_ablation1(self, round, model, discriminator, classifier_model, train_loader, global_proto,momentum = 1):
        model_state_dict = model.classifier.state_dict()
        classifier_model_state_dict = classifier_model.classifier.state_dict()
        with torch.no_grad():
            for key in model_state_dict:
                if "bn" not in key:
                    model_param = model_state_dict[key]
                    classifier_model_param = classifier_model_state_dict[key]
                    model_state_dict[key].copy_((1 - momentum) * model_param + momentum * classifier_model_param)
        model.classifier.load_state_dict(model_state_dict)
        model.train()
        features = []
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        for step in range(self.args.wk_iters):
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                rep, logits = model(images)
                CE_loss = self.criterion_CE(logits, labels)
                if len(global_proto) != 0 and self.args.loss_component in [2,4]:
                    num_classes = len(global_proto)
                    proto_dim = global_proto[0].shape[0]
                    global_proto_matrix = torch.zeros((num_classes, proto_dim), device=self.device)
                    for label, proto in global_proto.items():
                        global_proto_matrix[label] = proto
                    rep_normalized = F.normalize(rep, dim=1)
                    global_proto_matrix_normalized = F.normalize(global_proto_matrix, dim=1)
                    C_y = global_proto_matrix_normalized[labels]
                    numerator = torch.exp((rep_normalized * C_y).sum(dim=1) / self.args.T)
                    denominator = torch.exp((rep_normalized @ global_proto_matrix.T) / self.args.T).sum(dim=1)
                    loss1 = -torch.log(numerator / denominator).mean()
                else:
                    loss1 = 0
                if self.args.loss_component in [3,4]:
                    client_index = discriminator(rep)
                    client_index_softmax = F.log_softmax(client_index, dim=-1)
                    target_index = torch.full(client_index.shape, 1 / self.args.num_users).to(
                        self.device
                    )
                    target_index_softmax = F.softmax(target_index, dim=-1)
                    kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(self.device)
                    kl_loss = kl_loss_func(client_index_softmax, target_index_softmax)
                else:
                    kl_loss = 0
                loss = CE_loss + self.args.adcol_beta * kl_loss + self.args.adcol_mu * loss1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        train_iter = iter(train_loader)
        labels_ = []
        for batch_idx in range(len(train_iter)):
            images, labels = next(train_iter)
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            rep, logits = model(images)
            features.append(rep.detach().clone().cpu())
            labels_.append(labels.detach().clone().cpu())
        features = torch.cat(features, dim=0).to(self.args.device)
        labels_ = torch.cat(labels_, dim=0)
        return copy.deepcopy(model), [features, labels_]

    def update_weights_ours_ablation2(self, round, model, discriminator, classifier_model, train_loader, global_proto,momentum = 1):
        if self.args.cls_component == 1:
            model_state_dict = model.classifier.state_dict()
            classifier_model_state_dict = classifier_model.classifier.state_dict()
            with torch.no_grad():
                for key in model_state_dict:
                    if "bn" not in key:
                        model_param = model_state_dict[key]
                        classifier_model_param = classifier_model_state_dict[key]
                        model_state_dict[key].copy_((1 - momentum) * model_param + momentum * classifier_model_param)
            model.classifier.load_state_dict(model_state_dict)
        model.train()
        features = []
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        for step in range(self.args.wk_iters):
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                rep, logits = model(images)
                CE_loss = self.criterion_CE(logits, labels)
                if len(global_proto) == 0:
                    loss1 = 0
                else:
                    num_classes = len(global_proto)
                    proto_dim = global_proto[0].shape[0]
                    global_proto_matrix = torch.zeros((num_classes, proto_dim), device=self.device)
                    for label, proto in global_proto.items():
                        global_proto_matrix[label] = proto
                    rep_normalized = F.normalize(rep, dim=1)
                    global_proto_matrix_normalized = F.normalize(global_proto_matrix, dim=1)
                    C_y = global_proto_matrix_normalized[labels]
                    numerator = torch.exp((rep_normalized * C_y).sum(dim=1) / self.args.T)
                    denominator = torch.exp((rep_normalized @ global_proto_matrix.T) / self.args.T).sum(dim=1)
                    loss1 = -torch.log(numerator / denominator).mean()
                client_index = discriminator(rep)
                client_index_softmax = F.log_softmax(client_index, dim=-1)
                target_index = torch.full(client_index.shape, 1 / self.args.num_users).to(
                    self.device
                )
                target_index_softmax = F.softmax(target_index, dim=-1)
                kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(self.device)
                kl_loss = kl_loss_func(client_index_softmax, target_index_softmax)           
                loss = CE_loss + self.args.adcol_beta * kl_loss + self.args.adcol_mu * loss1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        train_iter = iter(train_loader)
        labels_ = []
        for batch_idx in range(len(train_iter)):
            images, labels = next(train_iter)
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            rep, logits = model(images)
            features.append(rep.detach().clone().cpu())
            labels_.append(labels.detach().clone().cpu())
        features = torch.cat(features, dim=0).to(self.args.device)
        labels_ = torch.cat(labels_, dim=0)
        return copy.deepcopy(model), [features, labels_]
    
    def compute_proto(self, model, train_loader):
        protos = {}
        outputall = []
        num_list = Counter()
        model.eval()
        train_iter = iter(train_loader)
        for batch_idx in range(len(train_iter)):
            images, labels = next(train_iter)
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            proto, _ = model(images)
            for i in range(len(labels)):
                num_list[labels[i].item()] += 1
                if labels[i].item() in protos:
                    protos[labels[i].item()].append(proto[i,:].detach().clone())
                else:
                    protos[labels[i].item()] = [proto[i,:].detach().clone()]
                outputall.append(proto[i, :].detach().clone())
        averaged_protos = {}
        for label, proto_list in protos.items():
            averaged_protos[label] = torch.mean(torch.stack(proto_list), dim=0)
        return averaged_protos, torch.stack(outputall, dim=0), num_list

    def compute_proto_ours(self, model, train_loader, classifier_model, momentum):
        model_state_dict = model.classifier.state_dict()
        classifier_model_state_dict = classifier_model.classifier.state_dict()
        with torch.no_grad():
            for key in model_state_dict:
                if "bn" not in key:
                    model_param = model_state_dict[key]
                    classifier_model_param = classifier_model_state_dict[key]
                    if momentum == 0:
                        model_state_dict[key].copy_(classifier_model_param)
                    else:
                        model_state_dict[key].copy_(model_param + momentum * classifier_model_param)
        model.classifier.load_state_dict(model_state_dict)
        protos = {}
        num_list = Counter()
        model.eval()
        train_iter = iter(train_loader)
        for batch_idx in range(len(train_iter)):
            images, labels = next(train_iter)
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            proto, output = model(images)
            for i in range(len(labels)):
                num_list[labels[i].item()] += 1
                if labels[i].item() in protos:
                    protos[labels[i].item()].append(proto[i,:].detach().clone())
                else:
                    protos[labels[i].item()] = [proto[i,:].detach().clone()]
        averaged_protos = {}
        for label, proto_list in protos.items():
            averaged_protos[label] = torch.mean(torch.stack(proto_list), dim=0)
        return averaged_protos,  num_list

    def compute_proto_debug(self, train_loader):
        protos = {}
        outputall = []
        num_list = Counter()
        features, labels = train_loader[0], train_loader[1]
        for batch_idx in range(len(features)):
            feature = features[batch_idx]
            label = labels[batch_idx]
            num_list[label.item()] += 1
            if label.item() in protos:
                protos[label.item()].append(feature.detach().clone())
            else:
                protos[label.item()] = [feature.detach().clone()]
        averaged_protos = {}
        for label, proto_list in protos.items():
            averaged_protos[label] = torch.mean(torch.stack(proto_list), dim=0)
        return averaged_protos, num_list

    def compute_proto_dispersion_debug(self, train_loader):
        protos = {}
        num_list = Counter()
        features, labels = train_loader[0], train_loader[1]
        for batch_idx in range(len(features)):
            feature = features[batch_idx]
            label = labels[batch_idx]
            num_list[label.item()] += 1
            if label.item() in protos:
                protos[label.item()].append(feature.detach().clone())
            else:
                protos[label.item()] = [feature.detach().clone()]
        averaged_protos = {}
        var_protos = {}
        for label, proto_list in protos.items():
            stacked = torch.stack(proto_list)
            averaged_protos[label] = torch.mean(stacked, dim=0)
            var_protos[label] = torch.var(stacked, dim=0, correction=0)
        return averaged_protos, var_protos, num_list

    def update_weights_rucr(self,  model, train_loader, global_proto, ratio_list):
        global_proto = torch.stack([global_proto[idx] for idx in range(len(global_proto))], dim=0)
        ratio_list = torch.tensor([ratio_list[idx] for idx in range(len(ratio_list))]).to(self.args.device)
        global_F = F.normalize(global_proto, dim=1)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        for step in range(self.args.wk_iters):
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                optimizer.zero_grad()
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                proto, output = model(images)
                loss1 = self.criterion_CE(output, labels)
                feat_loss = self.bal_simclr_imp(proto, labels, global_F, ratio_list)
                loss = loss1 + feat_loss * self.args.feat_loss_arg
                loss.backward()
                optimizer.step()
        return copy.deepcopy(model)

    def bal_simclr_imp(self, f, labels, global_proto, ratio_list):
        f_norm = F.normalize(f, dim=1)
        sim_logit = f_norm.mm(global_proto.T)
        sim_logit_tau = sim_logit.div(self.args.tau)
        src_ratio = ratio_list[labels].log() * self.args.times
        add_src = torch.scatter(torch.zeros_like(sim_logit), 1, labels.unsqueeze(1), src_ratio.view(-1, 1))
        f_out = sim_logit_tau + add_src
        loss = self.criterion_CE(f_out, labels)
        return loss
    
    def local_crt(self, model, glo_means, fs_all):
        for param_name, param in model.named_parameters():
            if 'classifier' not in param_name:
                param.requires_grad = False
        
        crt_dataset = MixupDataset_norm(glo_means, fs_all, self.args)
        model.eval()
        temp_optimizer = torch.optim.SGD(model.classifier.parameters(), lr=self.args.lr)
        for i in range(self.args.wk_iters):
            crt_loader = torch.utils.data.DataLoader(dataset=crt_dataset,
                                    batch_size=self.args.batch,
                                    shuffle=True)
            for feat, cls in crt_loader:
                feat, cls = feat.to(self.device), cls.to(self.device)
                outputs = model.classifier(feat)
                loss = self.criterion_CE(outputs, cls)
                temp_optimizer.zero_grad()
                loss.backward()
                temp_optimizer.step()
        return copy.deepcopy(model.classifier.state_dict())

class LocalTest(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()
        
    def test_inference(self, args, model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(args.device).float(), target.to(args.device).long()
            rep, output = model(data)
            test_loss += self.criterion(output, target).item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()
        acc = correct / len(test_loader.dataset)        
        return test_loss / len(test_loader), acc
    
    def test_inference_heal(self, args, model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(args.device).float(), target.to(args.device).long()
            rep, output = model(data)
            test_loss += self.criterion(output, target).item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()
        acc = correct / len(test_loader.dataset)        
        return test_loss / len(test_loader), acc

    def test_inference_proto(self, args, model, test_loader, global_proto):
        model.eval()
        test_loss = 0
        correct = 0
        loss_mse = nn.MSELoss()
        for data, target in test_loader:
            data, target = data.to(args.device).float(), target.to(args.device).long()
            proto, output = model(data)
            loss1 = self.criterion(output, target).item()
            proto_new = copy.deepcopy(proto.detach())
            i = 0
            for label in target:
                if label.item() in global_proto.keys():
                    proto_new[i, :] = global_proto[label.item()].data
                i += 1
            loss2 = loss_mse(proto_new, proto)
            test_loss += loss1 + loss2 * self.args.ld
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()
        acc = correct / len(test_loader.dataset)        
        return test_loss / len(test_loader), acc

    def test_inference_moon(self, args, old_model, model, global_model, test_loader):
        model.eval()
        old_model.eval()
        global_model.eval()
        test_loss = 0
        correct = 0
        targets = []
        for data, target in test_loader:
            data, target = data.to(self.device).float(), target.to(self.device).long()
            rep, output = model(data)
            loss1 = self.criterion(output, target)
            rep_global = global_model(data)[0].detach()
            rep_old = old_model(data)[0].detach()
            loss2 = - torch.log(torch.exp(F.cosine_similarity(rep, rep_global) / self.args.tau) / (torch.exp(F.cosine_similarity(rep, rep_global) / self.args.tau) + torch.exp(F.cosine_similarity(rep, rep_old) / self.args.tau)))
            test_loss += loss1 + torch.mean(loss2) * self.args.ld
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()
        acc = correct / len(test_loader.dataset)        
        return test_loss / len(test_loader), acc

    def test_inference_adcol(self, args, model, test_loader, dmodel):
        model.eval()
        dmodel.eval()
        test_loss = 0
        correct = 0
        loss_kl = 0
        for images, labels in test_loader:
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            rep, logits = model(images)
            loss1 = self.criterion(logits, labels)
            client_index = dmodel(rep)
            client_index_softmax = F.log_softmax(client_index, dim=-1)
            target_index = torch.full(client_index.shape, 1 / self.args.num_users).to(
                self.device
            )
            target_index_softmax = F.softmax(target_index, dim=-1)
            kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(self.device)
            kl_loss = kl_loss_func(client_index_softmax, target_index_softmax)  
            loss_kl += kl_loss
            test_loss += loss1
            pred = logits.data.max(1)[1]
            correct += pred.eq(labels.view(-1)).sum().item()
        acc = correct / len(test_loader.dataset)        
        return test_loss / len(test_loader), loss_kl / len(test_loader), acc
    
    def test_inference_ours(self, args, model, test_loader, dmodel, num_domains=None):
        if num_domains is None:
            num_domains = args.num_users
        model.eval()
        dmodel.eval()
        test_loss = 0
        correct = 0
        loss_kl = 0
        for images, labels in test_loader:
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            rep, logits = model(images)
            loss1 = self.criterion(logits, labels)
            client_index = dmodel(rep)
            client_index_softmax = F.log_softmax(client_index, dim=-1)
            target_index = torch.full(client_index.shape, 1 / num_domains).to(
                self.device
            )
            target_index_softmax = F.softmax(target_index, dim=-1)
            kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(self.device)
            kl_loss = kl_loss_func(client_index_softmax, target_index_softmax)  
            loss_kl += kl_loss
            test_loss += loss1
            pred = logits.data.max(1)[1]
            correct += pred.eq(labels.view(-1)).sum().item()
        acc = correct / len(test_loader.dataset)        
        return test_loss / len(test_loader), loss_kl / len(test_loader), acc
    
    def test_inference_ablation1(self, args, model, test_loader, global_proto,  dmodel):
        correct = 0
        loss_kl = 0
        model.eval()
        dmodel.eval()
        loss_CE = 0
        loss_info = 0
        train_iter = iter(test_loader)
        for batch_idx in range(len(train_iter)):
            images, labels = next(train_iter)
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            rep, logits = model(images)
            CE_loss = self.criterion(logits, labels)
            if len(global_proto) != 0 and self.args.loss_component in [2,4]:
                num_classes = len(global_proto)
                proto_dim = global_proto[0].shape[0]
                global_proto_matrix = torch.zeros((num_classes, proto_dim), device=self.device)
                for label, proto in global_proto.items():
                    global_proto_matrix[label] = proto
                rep_normalized = F.normalize(rep, dim=1)
                global_proto_matrix_normalized = F.normalize(global_proto_matrix, dim=1)
                C_y = global_proto_matrix_normalized[labels]
                numerator = torch.exp((rep_normalized * C_y).sum(dim=1) / self.args.T)
                denominator = torch.exp((rep_normalized @ global_proto_matrix.T) / self.args.T).sum(dim=1)
                loss1 = -torch.log(numerator / denominator).mean()
            else:
                loss1 = 0
            if self.args.loss_component in [3,4]:
                client_index = dmodel(rep)
                client_index_softmax = F.log_softmax(client_index, dim=-1)
                target_index = torch.full(client_index.shape, 1 / self.args.num_users).to(
                    self.device
                )
                target_index_softmax = F.softmax(target_index, dim=-1)
                kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(self.device)
                kl_loss = kl_loss_func(client_index_softmax, target_index_softmax)
            else:
                kl_loss = 0
            loss_CE += CE_loss
            loss_kl += kl_loss
            loss_info += loss1
            loss = CE_loss + self.args.adcol_beta * kl_loss + self.args.adcol_mu * loss1
            pred = logits.data.max(1)[1]
            correct += pred.eq(labels.view(-1)).sum().item()
        acc = correct / len(test_loader.dataset)  
        return loss_CE / len(test_loader), loss_kl / len(test_loader), loss_info / len(test_loader) , acc
    
    def test_inference_ablation2(self, args, model, test_loader, global_proto,  dmodel):
        correct = 0
        loss_kl = 0
        model.eval()
        dmodel.eval()
        loss_CE = 0
        loss_info = 0
        train_iter = iter(test_loader)
        for batch_idx in range(len(train_iter)):
            images, labels = next(train_iter)
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            rep, logits = model(images)
            CE_loss = self.criterion(logits, labels)
            if len(global_proto) != 0:
                num_classes = len(global_proto)
                proto_dim = global_proto[0].shape[0]
                global_proto_matrix = torch.zeros((num_classes, proto_dim), device=self.device)
                for label, proto in global_proto.items():
                    global_proto_matrix[label] = proto
                rep_normalized = F.normalize(rep, dim=1)
                global_proto_matrix_normalized = F.normalize(global_proto_matrix, dim=1)
                C_y = global_proto_matrix_normalized[labels]
                numerator = torch.exp((rep_normalized * C_y).sum(dim=1) / self.args.T)
                denominator = torch.exp((rep_normalized @ global_proto_matrix.T) / self.args.T).sum(dim=1)
                loss1 = -torch.log(numerator / denominator).mean()
            else:
                loss1 = 0
            client_index = dmodel(rep)
            client_index_softmax = F.log_softmax(client_index, dim=-1)
            target_index = torch.full(client_index.shape, 1 / self.args.num_users).to(
                self.device
            )
            target_index_softmax = F.softmax(target_index, dim=-1)
            kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(self.device)
            kl_loss = kl_loss_func(client_index_softmax, target_index_softmax)
            loss_CE += CE_loss
            loss_kl += kl_loss
            loss_info += loss1
            loss = CE_loss + self.args.adcol_beta * kl_loss + self.args.adcol_mu * loss1
            pred = logits.data.max(1)[1]
            correct += pred.eq(labels.view(-1)).sum().item()
        acc = correct / len(test_loader.dataset)  
        return loss_CE / len(test_loader), loss_kl / len(test_loader), loss_info / len(test_loader) , acc

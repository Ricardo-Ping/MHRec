
import datetime
import os

import torch
import numpy as np
import random
import logging

from metrics import precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k, map_at_k


# logging.basicConfig(level=logging.INFO)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # os.environ['PYTHONHASHSEED'] = str(seed)


def gpu():
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # logging.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        # logging.info("CUDA is not available, using CPU.")
    return device


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y-%H-%M-%S')
    return cur


class EarlyStopping:
    def __init__(self, patience=50, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metrics = None

    def __call__(self, score, metrics):
        if self.best_score is None:
            self.best_score = score
            self.best_metrics = metrics
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_metrics = metrics
            self.counter = 0


def distance_correlation(X1, X2, device):
    def _create_centered_distance(X):
        r = torch.sum(torch.square(X), 1, keepdim=True)
        D = torch.sqrt(torch.maximum(r - 2 * torch.matmul(X, X.transpose(1, 0)) + r.transpose(1, 0),
                                     torch.tensor([0.0]).to(device)) + 1e-8)
        D = D - torch.mean(D, dim=0, keepdim=True) - torch.mean(D, dim=1, keepdim=True) + torch.mean(D)
        return D

    def _create_distance_covariance(D1, D2):
        n_samples = torch.tensor(D1.shape[0], dtype=torch.float32).to(device)
        dcov = torch.sqrt(
            torch.maximum(torch.sum(D1 * D2) / (n_samples * n_samples), torch.tensor([0.0]).to(device)) + 1e-8)
        return dcov

    X1 = X1.to(device)
    X2 = X2.to(device)
    D1 = _create_centered_distance(X1)
    D2 = _create_centered_distance(X2)

    dcov_12 = _create_distance_covariance(D1, D2)
    dcov_11 = _create_distance_covariance(D1, D1)
    dcov_22 = _create_distance_covariance(D2, D2)

    dcor = dcov_12 / (torch.sqrt(torch.max(dcov_11 * dcov_22, torch.tensor([0.0]).to(device))) + 1e-10)
    dcor = dcor.squeeze()
    return dcor


def gene_metrics(val_data, rank_list, k_list):
    metrics = {k: {'precision': 0, 'recall': 0, 'ndcg': 0, 'hit_rate': 0, 'map': 0} for k in k_list}

    for data in val_data:
        user = data[0]
        pos_items = data[1:]
        ranked_items = rank_list[user].tolist()

        for k in k_list:
            metrics[k]['precision'] += precision_at_k(ranked_items, pos_items, k)
            metrics[k]['recall'] += recall_at_k(ranked_items, pos_items, k)
            metrics[k]['ndcg'] += ndcg_at_k(ranked_items, pos_items, k)
            metrics[k]['hit_rate'] += hit_rate_at_k(ranked_items, pos_items, k)
            metrics[k]['map'] += map_at_k(ranked_items, pos_items, k)

    num_users = len(val_data)

    for k in k_list:
        metrics[k]['precision'] /= num_users
        metrics[k]['recall'] /= num_users
        metrics[k]['ndcg'] /= num_users
        metrics[k]['hit_rate'] /= num_users
        metrics[k]['map'] /= num_users

    return metrics


def convert_to_dict(data):
    user_item_dict = {}
    for entry in data:
        user = entry[0]
        items = entry[1:]
        if user not in user_item_dict:
            user_item_dict[user] = []
        user_item_dict[user].extend(items)
    return user_item_dict


def topk_sample(k, user_graph_dict, num_user):
    user_graph_index = []
    count_num = 0
    tasike = [0] * k

    for i in range(num_user):
        if len(user_graph_dict[i][0]) < k:
            count_num += 1
            if len(user_graph_dict[i][0]) == 0:
                user_graph_index.append(tasike)
                continue
            user_graph_sample = user_graph_dict[i][0][:k]
            while len(user_graph_sample) < k:
                rand_index = np.random.randint(0, len(user_graph_sample))
                user_graph_sample.append(user_graph_sample[rand_index])
            user_graph_index.append(user_graph_sample)
            continue

        user_graph_sample = user_graph_dict[i][0][:k]
        user_graph_index.append(user_graph_sample)

    return user_graph_index

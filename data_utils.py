import os
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from scipy import sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from torch_sparse import SparseTensor
from baselines import *
from torch.distributions import Normal, kl, kl_divergence
import math
import copy
from contrastive_loss import CLUB
from param_utils import params, ratio
import seaborn as sns

def _module_path(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)

def train_step(optim, loss):
    optim.zero_grad()
    loss.backward()
    optim.step()
    return optim

def decouple_data(dataset_ind, dataset_ood_te):
    struct_data_ind = copy.deepcopy(dataset_ind)
    struct_data_ood_te = copy.deepcopy(dataset_ood_te)
    feat_data_ind = copy.deepcopy(dataset_ind)
    feat_data_ood_te = copy.deepcopy(dataset_ood_te)
    neutral_fill = params("neutral_fill")

    # Set all structure shifts node feature to constant
    struct_data_ind.x = torch.ones_like(struct_data_ind.x) - neutral_fill

    if isinstance(struct_data_ood_te, list):
        for d in struct_data_ood_te:
            d.x = torch.ones_like(d.x) - neutral_fill
    else:
        struct_data_ood_te.x = torch.ones_like(struct_data_ood_te.x) - neutral_fill
    
    # Set all feature shifts edge index to self-loop
    feat_data_ind.edge_index = torch.stack([torch.arange(feat_data_ind.num_nodes), torch.arange(feat_data_ind.num_nodes)], dim=0)
    
    if isinstance(feat_data_ood_te, list):
        for d in feat_data_ood_te:
            d.edge_index = torch.stack([torch.arange(d.num_nodes), torch.arange(d.num_nodes)], dim=0)
    else:
        feat_data_ood_te.edge_index = torch.stack([torch.arange(feat_data_ood_te.num_nodes), torch.arange(feat_data_ood_te.num_nodes)], dim=0)
    return struct_data_ind, struct_data_ood_te, feat_data_ind, feat_data_ood_te
    
def rand_splits(node_idx, train_prop=ratio("dropout"), valid_prop=ratio("planetoid_valid_prop")):
    """ randomly splits label into train/valid/test splits """
    splits = {}
    n = node_idx.size(0)

    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    splits['train'] = node_idx[train_indices]
    splits['valid'] = node_idx[val_indices]
    splits['test'] = node_idx[test_indices]

    return splits

def load_fixed_splits(data_dir, dataset, name, protocol):
    splits_lst = []
    if name in ['cora', 'citeseer', 'pubmed'] and protocol == 'semi':
        splits = {}
        splits['train'] = torch.as_tensor(dataset.train_mask.nonzero().squeeze(1))
        splits['valid'] = torch.as_tensor(dataset.val_mask.nonzero().squeeze(1))
        splits['test'] = torch.as_tensor(dataset.test_mask.nonzero().squeeze(1))
        splits_lst.append(splits)
    elif name in ['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'film', 'cornell', 'texas', 'wisconsin']:
        for i in range(params("mask_splits")):
            splits_file_path = '{}/geom-gcn/splits/{}'.format(data_dir, name) + '_split_0.6_0.2_'+str(i)+'.npz'
            splits = {}
            with np.load(splits_file_path) as splits_file:
                splits['train'] = torch.BoolTensor(splits_file['train_mask'])
                splits['valid'] = torch.BoolTensor(splits_file['val_mask'])
                splits['test'] = torch.BoolTensor(splits_file['test_mask'])
            splits_lst.append(splits)
    else:
        raise NotImplementedError

    return splits_lst

def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.quantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


def to_planetoid(dataset):
    """
        Takes in a NCDataset and returns the dataset in H2GCN Planetoid form, as follows:
        x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ty => the one-hot labels of the test instances as numpy.ndarray object;
        ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        split_idx => The ogb dictionary that contains the train, valid, test splits
    """
    split_idx = dataset.get_idx_split('random', 0.25)
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    graph, label = dataset[0]

    label = torch.squeeze(label)

    print("generate x")
    x = graph['node_feat'][train_idx].numpy()
    x = sp.csr_matrix(x)

    tx = graph['node_feat'][test_idx].numpy()
    tx = sp.csr_matrix(tx)

    allx = graph['node_feat'].numpy()
    allx = sp.csr_matrix(allx)

    y = F.one_hot(label[train_idx]).numpy()
    ty = F.one_hot(label[test_idx]).numpy()
    ally = F.one_hot(label).numpy()

    edge_index = graph['edge_index'].T

    graph = defaultdict(list)

    for i in range(0, label.shape[0]):
        graph[i].append(i)

    for start_edge, end_edge in edge_index:
        graph[start_edge.item()].append(end_edge.item())

    return x, tx, allx, y, ty, ally, graph, split_idx


def to_sparse_tensor(edge_index, edge_feat, num_nodes):
    """ converts the edge_index into SparseTensor
    """
    num_edges = edge_index.size(1)

    (row, col), N, E = edge_index, num_nodes, num_edges
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]

    value = edge_feat[perm]
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(N, N), is_sorted=True)

    # Pre-process some important attributes.
    adj_t.storage.rowptr()
    adj_t.storage.csr2csc()

    return adj_t


def normalize(edge_index):
    """ normalizes the edge_index
    """
    adj_t = edge_index.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    return adj_t


def gen_normalized_adjs(dataset):
    """ returns the normalized adjacency matrix
    """
    row, col = dataset.graph['edge_index']
    N = dataset.graph['num_nodes']
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    D_isqrt = deg.pow(-0.5)
    D_isqrt[D_isqrt == float('inf')] = 0

    DAD = D_isqrt.view(-1,1) * adj * D_isqrt.view(1,-1)
    DA = D_isqrt.view(-1,1) * D_isqrt.view(-1,1) * adj
    AD = adj * D_isqrt.view(1,-1) * D_isqrt.view(1,-1)
    return DAD, DA, AD


def stable_cumsum(arr, rtol=ratio("stable_rtol"), atol=ratio("stable_atol")):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=ratio("recall_level"), pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))
    if np.array_equal(classes, [1]):
        return thresholds[cutoff]  # return threshold

    return fps[cutoff] / (np.sum(np.logical_not(y_true))), thresholds[cutoff]

def get_measures(_pos, _neg, recall_level=ratio("recall_level")):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)
    fpr, threshould = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr, threshould


def eval_f1(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().detach().numpy()
    if y_pred.shape == y_true.shape:
        y_pred = y_pred.detach().cpu().detach().numpy()
    else:
        y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().detach().numpy()

    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true, y_pred, average='micro')
        acc_list.append(f1)

    return sum(acc_list)/len(acc_list)

def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().detach().numpy()
    if y_pred.shape == y_true.shape:
        y_pred = y_pred.detach().cpu().detach().numpy()
    else:
        y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().detach().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)


def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().detach().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:,1].unsqueeze(1).cpu().detach().detach().numpy()
    else:
        y_pred = y_pred.detach().cpu().detach().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                                
            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)

def mutual_independence_loss(model_a, model_b, dataset_a, dataset_b, club_fn, device, weight=ratio("pairwise_weight"), \
                             verbose = False, name = ""):
    with torch.no_grad():
        x_a, edge_index_a = dataset_a.x.to(device), dataset_a.edge_index.to(device)
        x_b, edge_index_b = dataset_b.x.to(device), dataset_b.edge_index.to(device)
        embed_a = model_a.encoder(x_a, edge_index_a)[1]
        embed_b = model_b.encoder(x_b, edge_index_b)[1]

    loss = club_fn.learning_loss(embed_a, embed_b)
    # print("{} Mutual Independence Loss: {:.4f}".format(name, loss.item()))
    return loss * weight

@torch.no_grad()
def evaluate_classify(model, dataset, eval_func, criterion, args, device):
    model.eval()

    train_idx, valid_idx, test_idx = dataset.splits['train'], dataset.splits['valid'], dataset.splits['test']
    y = dataset.y
    out, _, _, _ = model(dataset, device).cpu().detach()

    train_score = eval_func(y[train_idx], out[train_idx])
    valid_score = eval_func(y[valid_idx], out[valid_idx])
    test_score = eval_func(y[test_idx], out[test_idx])

    if args.method != 'GPN':
        if args.dataset in ('proteins', 'ppi'):
            valid_loss = criterion(out[valid_idx], y[valid_idx].to(torch.float))
        else:
            valid_out = F.log_softmax(out[valid_idx], dim=1)
            valid_loss = criterion(valid_out, y[valid_idx].squeeze(1))

        return train_score, valid_score, test_score, valid_loss
    else:
        return train_score, valid_score, test_score

#@torch.no_grad()
def evaluate_detect(model, structure_model, feature_model, dataset_ind, dataset_ood, struct_data_ind, struct_data_ood, \
                    feature_data_ind, feature_data_ood, criterion, eval_func, args, \
                    club_zq, club_zv, club_qv, device, best_val = None, return_score=False, target_model = "Model"):
    if target_model == "Structure":
        # print("Structure Model")
        target_model = structure_model
        data_ind = struct_data_ind
        data_ood = struct_data_ood

    elif target_model == "Feature":
        # print("Feature Model")
        target_model = feature_model
        data_ind = feature_data_ind
        data_ood = feature_data_ood

    elif target_model == "Model":
        # print("Model")
        target_model = model
        data_ind = dataset_ind
        data_ood = dataset_ood

    target_model.eval()
    model.eval()
    structure_model.eval()
    feature_model.eval()
    club_zq.eval()
    club_zv.eval()
    club_qv.eval()

    with torch.no_grad():
        test_ind_score = target_model.detect(data_ind, data_ind.splits['test'], device, args).cpu().detach()
    if isinstance(data_ood, list):
        result = []
        threshould = []
        test_ood_score = []
        for d in data_ood:
            with torch.no_grad():
                ood_score = target_model.detect(d, d.node_idx, device, args).cpu().detach()
            auroc, aupr, fpr, thresh = get_measures(test_ind_score, ood_score)
            result += [auroc] + [aupr] + [fpr]
            threshould.append(thresh)
            test_ood_score.append(ood_score)
    else:
        with torch.no_grad():
            test_ood_score = target_model.detect(data_ood, data_ood.node_idx, device, args).cpu().detach()
        auroc, aupr, fpr, threshould = get_measures(test_ind_score, test_ood_score)
        result = [auroc] + [aupr] + [fpr]

    kl_loss = torch.tensor(0)
    recon_loss = torch.tensor(0)
    qv_indep_loss_id = torch.tensor(0)
    valid_idx = data_ind.splits['valid']
    test_idx = data_ind.splits['test']

    if args.backbone == 'gcn':
        out, Z_in = target_model(data_ind, device)
    else:
        with torch.no_grad():
            out, Z_in, mu, std, x_recon_in, mu_q, log_var_q = target_model(data_ind, device)
            x_in = data_ind.x.to(device).detach()
            if args.beta != 0:
                kl_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
            if (target_model == model) and (args.gamma != 0):
                recon_loss = F.l1_loss(x_recon_in[valid_idx], x_in[valid_idx]).detach()
                log_q = -0.5 * torch.mean(log_var_q + (x_in - mu_q) ** 2 / torch.exp(log_var_q))
                recon_loss = (recon_loss - 0.01 * log_q)
            if args.use_pairwise:
                clubzq_loss = mutual_independence_loss(feature_model, model, feature_data_ind, dataset_ind, club_zq, device, weight = args.pmi_w)
                clubzv_loss = mutual_independence_loss(structure_model, model, struct_data_ind, dataset_ind, club_zv, device, weight = args.pmi_w)
                qv_indep_loss_id = clubzv_loss + clubzq_loss

    out = out.cpu().detach()
    test_score = eval_func(data_ind.y[test_idx], out[test_idx])

    valid_out = F.log_softmax(out[valid_idx], dim=1)
    valid_sup_loss = criterion(valid_out, data_ind.y[valid_idx].squeeze(1))
    valid_loss = valid_sup_loss + args.beta * kl_loss + args.gamma * recon_loss + qv_indep_loss_id
    result += [test_score] + [valid_loss]

    if args.save_model:
        if (best_val is not None) and (valid_loss.item() < best_val):
            print("Saving Model", result)
            save_dir = _module_path('saved_models')
            os.makedirs(save_dir, exist_ok=True)
            if args.dataset not in ['twitch', 'arxiv']:
                torch.save(target_model.state_dict(), os.path.join(save_dir, f'{args.backbone}_{args.dataset}_{args.ood_type}_{args.use_pairwise}.pth'))
                torch.save(feature_model.state_dict(), os.path.join(save_dir, f'{args.backbone}_{args.dataset}_{args.ood_type}_{args.use_pairwise}_Feat.pth'))
                torch.save(structure_model.state_dict(), os.path.join(save_dir, f'{args.backbone}_{args.dataset}_{args.ood_type}_{args.use_pairwise}_Struct.pth'))
            else:
                torch.save(target_model.state_dict(), os.path.join(save_dir, f'{args.backbone}_{args.dataset}_{args.use_pairwise}_Best.pth'))
            best_val = valid_loss.item()
            print("IZY", valid_sup_loss, "IZX", kl_loss)

    if return_score:
        return result, test_ind_score, test_ood_score, threshould
    elif best_val:
        return result, best_val, Z_in
    else:
        return result, Z_in

def convert_to_adj(edge_index,n_node):
    '''convert from pyg format edge_index to n by n adj matrix'''
    adj=torch.zeros((n_node,n_node))
    row,col=edge_index
    adj[row,col]=1
    return adj

import subprocess
def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    return gpu_memory

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

dataset_drive_url = {
    'snap-patents' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia', 
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y', 
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ', 
}

splits_drive_url = {
    'snap-patents' : '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N', 
    'pokec' : '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_',
}

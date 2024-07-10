import numpy as np
import pickle as pkl
import torch.nn as nn
import torch.nn.functional as F


import scipy.sparse as sp
import re
import torch
import json

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def true_metric_loss(true,mask,no_of_classes, scale=1):
    # true: 真实的标签向量，其中每个元素代表一个样本的类别标签。
    # no_of_classes: 数据集中总的类别数量。
    # scale: 一个可选的缩放因子，默认值为1，用于调节类别之间差异的影响。
    batch_size = true.size(0) # 批次中样本的数量。

    true = true.view(batch_size,1) # 将真实标签向量转换为(batch_size, 1)的形状。
    # 将真实标签向量转换为LongTensor类型，并在列方向上重复no_of_classes次，形成一个矩阵，然后转换为浮点数。这个矩阵的每一行都是相同的真实标签。
    true_labels = torch.cuda.LongTensor(true).repeat(1, no_of_classes).float()
    # class_labels = torch.arange(no_of_classes).float().cuda()：生成一个从0到no_of_classes-1的连续整数向量，然后转换为浮点数并移动到CUDA设备上。
    class_labels = torch.arange(no_of_classes).float().cuda()
    # 计算class_labels向量和true_labels矩阵之间的绝对差值，然后乘以缩放因子scale
    phi = (scale * torch.abs(class_labels - true_labels)).cuda()
    # 对phi矩阵的每一行进行softmax操作，得到一个概率分布。
    y = nn.Softmax(dim=1)(-phi) # 用-phi是为了让距离较小（即类别接近真实标签）的类别有较高的概率值。
    results = torch.zeros(batch_size, no_of_classes).cuda() # 初始化一个(batch_size, no_of_classes)的矩阵。
    mask = mask.bool()
    results[mask] = y[mask] # 将y矩阵中的每一行（即每个样本）的最大值所在的索引位置置为1，其余位置置为0。
    return results


def loss_function(output, labels,mask, expt_type=5, scale=1.8):
    targets = true_metric_loss(labels,mask, expt_type, scale)
    return torch.sum(- targets * F.log_softmax(output, -1), -1).mean()


def gr_metrics(op, t,train_mask):

    if isinstance(train_mask, torch.Tensor):
        train_mask = train_mask.cpu().numpy()

    if isinstance(op, torch.Tensor):
        op = op.cpu().numpy()

    if isinstance(t, torch.Tensor):
        t = t.cpu().numpy()

    op_t_mask = (op == t)
    top_mask = (t > op)
    opt_mask = (op > t)

    train_mask_bool = (train_mask == 1)

    TP = np.sum(op_t_mask[train_mask_bool])
    FN = np.sum(top_mask[train_mask_bool])
    FP = np.sum(opt_mask[train_mask_bool])

    GP = TP/(TP + FP)
    GR = TP/(TP + FN)
    FS = 2 * GP * GR / (GP + GR)
    return GP, GR, FS


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)

def load_corpus_torch(dataset, device):
    """
    Loads input corpus from gcn/data directory, torch tensor version

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    adjs = []

    graph_list = ['seq','sem','syn']
    # graph_list = ['syn','seq','sem']
    for one in graph_list:
    # for one in ['seq','sem']:
        adjs.append(pkl.load(open('./data/{}.{}_adj'.format(dataset,one),'rb')))
    
    adj_len = len(adjs)

    if adj_len == 3:
        print('3')
        adj, adj1, adj2 = adjs[0], adjs[1], adjs[2]
    elif adj_len == 2:
        print('2')
        adj, adj1 = adjs[0], adjs[1]
    else:
        print('1')
        adj = adjs[0]

        

    # adj, adj1, adj2 = adjs[0], adjs[1], adjs[2]
    # adj, adj1 = adjs[0], adjs[1]




    
    data = json.load(open('./data/{}_data.json'.format(dataset),'r'))
    train_ids, val_ids,test_ids, corpus, labels, vocab, word_id_map, id_word_map, label_list = data

    num_labels = len(label_list)

    train_size = len(train_ids)
    val_size = len(val_ids)
    test_size = len(test_ids)

    labels = np.asarray(labels[:train_size+val_size]+[0]*len(vocab)+labels[train_size+val_size:])
    # print(len(labels))
    # idx_train = range(train_size-val_size)
    idx_train = range(train_size)
    idx_val = range(train_size, train_size+val_size)
    idx_test = range(train_size+val_size+len(vocab), train_size+val_size+test_size+len(vocab))
    
    train_mask = sample_mask(idx_train, labels.shape[0])
    print(train_mask)
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_test[test_mask] = labels[test_mask]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj1 = adj1 + adj1.T.multiply(adj1.T > adj1) - adj1.multiply(adj1.T > adj1)
    adj2 = adj2 + adj2.T.multiply(adj2.T > adj2) - adj2.multiply(adj2.T > adj2)


    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    train_mask = torch.tensor(train_mask, dtype=torch.float).to(device)
    val_mask = torch.tensor(val_mask, dtype=torch.float).to(device)
    test_mask = torch.tensor(test_mask, dtype=torch.float).to(device)

    return adj, adj1, adj2, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, val_size,test_size, num_labels,vocab
    # return adj, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, val_size,test_size, num_labels,vocab
    # return adj, adj1, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, val_size,test_size, num_labels
    
def get_edge_tensor_list(adj_list, device):
    indice_list, data_list = [], []
    for adj in adj_list:
        row = torch.tensor(adj.row, dtype=torch.long).to(device)
        col = torch.tensor(adj.col, dtype=torch.long).to(device)
        data = torch.tensor(adj.data, dtype=torch.float).to(device)
        indice = torch.stack((row,col),dim=0)
        indice_list.append(indice)
        data_list.append(data)
    return indice_list, data_list

def get_edge_tensor(adj):
    row = torch.tensor(adj.row, dtype=torch.long)
    col = torch.tensor(adj.col, dtype=torch.long)
    data = torch.tensor(adj.data, dtype=torch.float)
    indice = torch.stack((row,col),dim=0)
    return indice, data

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

def preprocess_features_origin(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def preprocess_adj_mix(adj):
    adj_normalized = adj + sp.eye(adj.shape[0])
    return sparse_to_tuple(adj)

def preprocess_adj_tensor(adj, device):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return torch.sparse_coo_tensor(np.stack([adj_normalized.row, adj_normalized.col], axis=0), adj_normalized.data, adj_normalized.shape, dtype=torch.float).to(device)

def preprocess_adj_mix_tensor(adj, device):
    adj_normalized = adj + sp.eye(adj.shape[0])
    # return torch.sparse_csr_tensor(crow_indices=adj.indptr, col_indices=adj.indices, values=adj.data, dtype=torch.float).to_sparse_coo().to(device)
    return torch.tensor(adj.todense(), dtype=torch.float).to(device)
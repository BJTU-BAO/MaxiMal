import pickle as pkl
import time
import copy
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, auc, roc_curve
import os
import random


def add_node(adj,features):
    n_nodes, feat_dim = features.shape
    csr_matrix1=np.zeros((n_nodes,1))
    csr_matrix2=np.zeros((1,n_nodes+1))
    csr_matrix1=sp.csr_matrix(csr_matrix1)
    csr_matrix2=sp.csr_matrix(csr_matrix2)
    adj=sp.hstack((adj,csr_matrix1))
    adj=sp.vstack((adj,csr_matrix2))
    adj=adj.A
    adj=sp.csr_matrix(adj)

    features_array=features.numpy()
    index=np.random.randint(0,n_nodes)
    features_new_node=features_array[index]
    features_new_node=sp.csr_matrix(features_new_node)
    features_array=sp.csr_matrix(features_array)

    features=sp.vstack((features_array,features_new_node))
    features=features.A
    features=torch.from_numpy(features)

    return adj,features


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
        # objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def find_IMgard_max(grad,features,dataset,addNum):
    grad_array=grad.cpu().numpy()
    node,feat_dim=grad.shape

    step=0.3
    if dataset=="cora" or dataset=="citeseer":

        features_copy = features.clone()
        for j in range(addNum):
            item = j + 1
            random.seed(43 + j)
            for i in random.sample(range(feat_dim), feat_dim // 5):
                features_copy[node - item][i] = 1 if grad_array[node - item][i] > 0 else 0 #<
        features = features_copy

    return features.cuda()
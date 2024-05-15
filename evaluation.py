from deeprobust.graph.utils import accuracy
from deeprobust.graph.defense import GCN
import torch
import numpy as np
from deeprobust.graph.data import Dataset
import os
import random
import pickle as pkl
import os.path as osp
from pGRACE.dataset import get_dataset
import argparse

SEED=2023
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0') #cuda:0
    parser.add_argument('--dataset', type=str, default='CiteSeer')
    parser.add_argument('--addNodesPercent', type=int, default=0.01)
    args = parser.parse_args()
    device = torch.device(args.device)

    path = osp.expanduser('dataset')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]

    nodelist = [i for i in range(data.num_nodes)]
    nodelist = torch.tensor(nodelist, dtype=torch.long)
    idx_train = nodelist[data.train_mask]
    idx_val = nodelist[data.val_mask]
    idx_test = nodelist[data.test_mask]

    edge_index = data.edge_index
    edge_sp_adj = torch.sparse.FloatTensor(edge_index,
                                           torch.ones(edge_index.shape[1]),
                                           [data.num_nodes, data.num_nodes])
    edge_adj = edge_sp_adj.to_dense()
    features = data.x
    adj = edge_adj
    labels = data.y
    perturbed_adj = pkl.load(open('poisoned_adj/%s_MaxiMal_%f_adj.pkl' % (args.dataset, args.addNodesPercent), 'rb'))
    perturbed_adj_fea = pkl.load(open('poisoned_adj/%s_MaxiMal_%f_fea.pkl' % (args.dataset, args.addNodesPercent), 'rb'))


    list1 =[]
    list2 =[]
    #
    def asr2(features, adj, modified_features, modified_adj, labels, idx_test, idx_train, idx_val, retrain=True):

        if retrain:
            gcn = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                      nhid=512, dropout=0.5, with_relu=False, with_bias=True, device='cuda')
            gcn.fit(modified_features, modified_adj, labels, idx_train, idx_val, patience=50, device='cuda')
        modified_output = gcn.predict(modified_features, modified_adj)
        # modified_output = gcn.predict(features, adj)
        acc2 = float(accuracy(modified_output[idx_test], labels[idx_test]))

        print('The accuracy after the attacks:', acc2)
        print('The nodes before the attacks:', adj.shape[0])
        print('The nodes after the attacks:', modified_adj.shape[0])
        return acc2
    # #

    for i in range(5):

        acc2= asr2(features, adj, perturbed_adj_fea, perturbed_adj, labels, idx_test, idx_train, idx_val, retrain=True)
        gcn = None
        list2.append(acc2)

    print(list2)
    average = sum(list2) / len(list2)

    print(average)


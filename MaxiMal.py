import networkx as nx
import scipy.sparse
import torch_geometric
from scipy.sparse import coo_matrix
from torch.nn.modules.module import Module
import argparse
import os.path as osp
import time
import pickle as pkl
import torch
from torch_geometric.utils import dropout_adj, degree, to_undirected, dense_to_sparse, is_undirected, to_networkx, \
    contains_isolated_nodes
import numpy as np
from INjection_way import load_data, add_node, find_IMgard_max
from simple_param.sp import SimpleParam
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted, feature_drop_weights_dense
from pGRACE.utils import get_activation, compute_pr, eigenvector_centrality
from pGRACE.dataset import get_dataset
from differentiable_models.gcn import GCN
from differentiable_models.model import GRACE
import scipy.sparse as sp
from scipy.sparse import csr_matrix


class Metacl(Module):
    def __init__(self, args, dataset, param, device, data, addNodeNum, addEdgeNum):
        super(Metacl, self).__init__()
        self.model = None
        self.optimizer = None
        self.param = param
        self.args = args
        self.device = device
        self.dataset = dataset
        self.data = data.cuda()
        self.drop_weights = None
        self.feature_weights = None
        self.addNum = addNodeNum
        self.addEdgeNum = addEdgeNum


    def train_gcn(self):
        self.model.train()
        self.optimizer.zero_grad()
        edge_index_1 = self.data.edge_index_ori
        edge_index_2 = self.data.edge_index
        x_1 = self.data.x_ori
        x_2 = self.data.x
        edge_sp_adj_1 = torch.sparse.FloatTensor(edge_index_1.cuda(),
                                                 torch.ones(edge_index_1.shape[1]).to(self.device),
                                                 [self.data.num_nodes, self.data.num_nodes])
        edge_sp_adj_2 = torch.sparse.FloatTensor(edge_index_2.cuda(),
                                                 torch.ones(edge_index_2.shape[1]).to(self.device),
                                                 [self.data.num_nodes, self.data.num_nodes])

        z1 = self.model(x_1, edge_sp_adj_1, sparse=True)
        z2 = self.model(x_2, edge_sp_adj_2, sparse=True)
        loss = self.model.loss(z1, z2, batch_size=None)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item()

    def compute_drop_weights(self):
        if self.param['drop_scheme'] == 'degree':
            self.drop_weights = degree_drop_weights(self.data.edge_index).to(self.device)
        elif self.param['drop_scheme'] == 'pr':
            self.drop_weights = pr_drop_weights(self.data.edge_index, aggr='sink', k=200).to(self.device)
        elif self.param['drop_scheme'] == 'evc':
            self.drop_weights = evc_drop_weights(self.data).to(self.device)
        else:
            self.drop_weights = None

        if self.param['drop_scheme'] == 'degree':
            edge_index_ = to_undirected(self.data.edge_index)
            node_deg = degree(edge_index_[1])
            if self.args.dataset == 'WikiCS':
                self.feature_weights = feature_drop_weights_dense(self.data.x, node_c=node_deg).to(self.device)
            else:
                self.feature_weights = feature_drop_weights(self.data.x, node_c=node_deg).to(self.device)
        elif self.param['drop_scheme'] == 'pr':
            node_pr = compute_pr(self.data.edge_index)
            if self.args.dataset == 'WikiCS':
                self.feature_weights = feature_drop_weights_dense(self.data.x, node_c=node_pr).to(self.device)
            else:
                self.feature_weights = feature_drop_weights(self.data.x, node_c=node_pr).to(self.device)
        elif self.param['drop_scheme'] == 'evc':
            node_evc = eigenvector_centrality(self.data)
            if self.args.dataset == 'WikiCS':
                self.feature_weights = feature_drop_weights_dense(self.data.x, node_c=node_evc).to(self.device)
            else:
                self.feature_weights = feature_drop_weights(self.data.x, node_c=node_evc).to(self.device)
        else:
            self.feature_weights = torch.ones((self.data.x.size(1),)).to(self.device)

    def inner_train(self):

        encoder = GCN(self.dataset.num_features, self.param['num_hidden'], get_activation(self.param['activation']))
        self.model = GRACE(encoder, self.param['num_hidden'], self.param['num_proj_hidden'], self.param['tau']).to(
            self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.param['learning_rate'],
            weight_decay=self.param['weight_decay']
        )

        # self.compute_drop_weights()

        lossmax = 999.999
        for epoch in range(1, self.param['num_epochs'] + 1):
            loss = self.train_gcn()
            # ----------------------------
            if loss < lossmax:
                lossmax = loss
                patience_count = 0
            else:
                patience_count = patience_count + 1

            if patience_count >= args.patience:
                print('Early Stopping.  epoch=' + str(epoch))
                break
            # ----------------------------------

        return loss

    def compute_gradient(self):
        self.model.eval()
        edge_index_1 = self.data.edge_index_ori
        edge_index_2 = self.data.edge_index
        x_1 = self.data.x_ori
        x_2 = self.data.x.clone().detach()
        edge_sp_adj_1 = torch.sparse.FloatTensor(edge_index_1,
                                                 torch.ones(edge_index_1.shape[1]).to(self.device),
                                                 [self.data.num_nodes, self.data.num_nodes])
        edge_sp_adj_2 = torch.sparse.FloatTensor(edge_index_2,
                                                 torch.ones(edge_index_2.shape[1]).to(self.device),
                                                 [self.data.num_nodes, self.data.num_nodes])

        edge_adj_1 = edge_sp_adj_1.to_dense()
        edge_adj_2 = edge_sp_adj_2.to_dense()
        edge_adj_1.requires_grad = True
        edge_adj_2.requires_grad = True

        x_1.requires_grad = True
        x_2.requires_grad = True

        z1 = self.model(x_1, edge_adj_1, sparse=False)
        z2 = self.model(x_2, edge_adj_2, sparse=False)
        loss = self.model.loss(z1, z2, batch_size=None)
        loss.backward()
        return edge_adj_1.grad, edge_adj_2.grad, x_1.grad, x_2.grad

    def attack(self):
        perturbed_edges = []
        num_total_edges = self.data.num_edges

        adj, features = load_data(args.dataset_str)
        G1 = nx.from_numpy_array(adj)
        degree1_max = len(nx.degree_histogram(G1)) - 1
        print("degree1_max:" + str(degree1_max))

        adj = coo_matrix(
            (np.ones(self.data.num_edges),
             (self.data.edge_index[0].cpu().numpy(), self.data.edge_index[1].cpu().numpy())),
            shape=(self.data.num_nodes, self.data.num_nodes))

        adj = adj.tocsr()
        G2 = nx.from_numpy_array(adj)
        degree2_max = len(nx.degree_histogram(G2)) - 1
        adj = adj.todense()
        adj = torch.from_numpy(adj).cuda()


        degrees = [deg * count for deg, count in enumerate(nx.degree_histogram(G1))]
        average_degree = round(sum(degrees) / G1.number_of_nodes())
        allowAddEdgeNum = int(average_degree * addNodeNum * 2)
        print("average_degree" + str(average_degree))
        print("allowAddEdgeNum" + str(allowAddEdgeNum))


        print('Begin perturbing.....')

        while len(perturbed_edges) < int(allowAddEdgeNum) * 2:

            start = time.time()
            self.inner_train()

            adj_1_grad, adj_2_grad, x1_grad, x2_grad = self.compute_gradient()
            grad_f = x2_grad
            features = find_IMgard_max(grad_f, self.data.x, args.dataset_str, self.addNum)

            # grad_sum = adj_1_grad + adj_2_grad
            grad_sum = adj_2_grad

            col_indices = np.loadtxt(args.dataset_str + '_tar/data_IM_RIS_sus.csv', dtype=int, unpack=False)
            row_indices = list(range(grad_sum.shape[0] - self.addNum, grad_sum.shape[0]))

            new_adj_matrix = adj[row_indices][:, col_indices]

            sorted_indices = np.argsort(np.abs(grad_sum[row_indices][:, col_indices].cpu()), axis=None)
            sorted_indices = torch.flip(sorted_indices, dims=[0])
            sorted_indices = sorted_indices.numpy()
            count = 0
            for idx in sorted_indices:

                row_idx, col_idx = np.unravel_index(idx, (new_adj_matrix.shape[0], new_adj_matrix.shape[1]))
                if new_adj_matrix[row_idx, col_idx] == 0 and grad_sum[
                    row_indices[row_idx], col_indices[col_idx]] > 0:  # 1 && neg
                    adj[row_indices[row_idx], col_indices[col_idx]] = 1
                    adj[col_indices[col_idx], row_indices[row_idx]] = 1
                    perturbed_edges.append([row_indices[row_idx], col_indices[col_idx]])
                    perturbed_edges.append([col_indices[col_idx], row_indices[row_idx]])
                    count += 1
                    if len(perturbed_edges) > int(allowAddEdgeNum) * 2:
                        break
                    if count == 1:  # Set the number of edges for a single iteration perturbation
                        break


            self.data.edge_index = dense_to_sparse(adj)[0]
            self.data.x = features

            end = time.time()
            G2 = nx.from_numpy_array(adj.cpu().numpy())
            degree2_max = len(nx.degree_histogram(G2)) - 1

            print("degree2_max" + str(degree2_max))
            print('Perturbing edges: %d/%d. Finished in %.2fs' % (
                len(perturbed_edges) / 2, int(addEdgeNum / 2), end - start))

        print('Number of perturbed edges: %d' % (len(perturbed_edges) / 2))
        output_adj = adj.to(device)
        output_adj_fea = features.to(device)

        return output_adj, output_adj_fea


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='CiteSeer')   # Cora CiteSeer PubMed
    parser.add_argument('--dataset_str', type=str, default='citeseer')  # cora citeseer pubmed
    parser.add_argument('--param', type=str, default='local:general.json')
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--addNodesPercent', type=int, default=0.02)
    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 1000,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()
    addPercent = args.addNodesPercent

    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')

    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    use_nni = args.param == 'nni'
    if use_nni and args.device != 'cpu':
        args.device = 'cuda'

    device = torch.device(args.device)

    path = osp.expanduser('dataset')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)

    data = dataset[0]
    ori_edgeNum = data.edge_index.shape[1]
    addNodeNum = int(addPercent * data.num_nodes)
    adj, features = load_data(args.dataset_str)
    for i in range(addNodeNum):
        adj, features = add_node(adj, features)

    #
    coo_matrix_adj = adj.tocoo()
    row_indices_adj = coo_matrix_adj.row
    col_indices_adj = coo_matrix_adj.col
    edge_index_ori = np.stack((row_indices_adj, col_indices_adj), axis=0)
    edge_index_ori = torch.from_numpy(edge_index_ori).long()
    data['edge_index_ori'] = edge_index_ori

    #
    x_ori = features
    data['x_ori'] = x_ori

    data.edge_index = edge_index_ori
    data.x = features

    after_edgeNum = data.edge_index.shape[1]
    addEdgeNum = after_edgeNum - ori_edgeNum

    model = Metacl(args, dataset, param, device, data, addNodeNum, addEdgeNum).to(device)
    poisoned_adj, poisoned_adj_fea = model.attack()
    pkl.dump(poisoned_adj.to(torch.device('cpu')),
             open('poisoned_adj/%s_MaxiMal_%f_adj.pkl' % (args.dataset, args.addNodesPercent), 'wb'))
    pkl.dump(poisoned_adj_fea.to(torch.device('cpu')),
             open('poisoned_adj/%s_MaxiMal_%f_fea.pkl' % (args.dataset, args.addNodesPercent), 'wb'))
    print(('---%f perturbed adjacency matrix saved---'% args.addNodesPercent))

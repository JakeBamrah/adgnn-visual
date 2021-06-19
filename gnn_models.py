import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import gnn_layers as gnn_l


class SoftmaxModule():
    def __init__(self):
        self.softmax_metric = 'log_softmax'

    def forward(self, outputs):
        if self.softmax_metric == 'log_softmax':
            return F.log_softmax(outputs, dim=1)
        else:
            raise (NotImplementedError)


class AMGNN(nn.Module):
    def __init__(self, args):
        super(AMGNN, self).__init__()

        self.metric_network = args.metric_network
        self.emb_size = args.feature_num
        self.args = args

        if self.metric_network == 'gnn':
            assert (self.args.train_N_way == self.args.test_N_way)
            num_inputs = self.emb_size + self.args.train_N_way
            print('Features:',num_inputs)
            # input('0_0')
            self.gnn_obj = gnn_l.GNN_nl(args, num_inputs, nf=96, J=1)

        else:
            raise NotImplementedError

    def gnn_iclr_forward(self, z_c, z, zi_c, zi_s, labels_yi,adj):
        # Creating WW matrix
        zero_pad = Variable(torch.zeros(labels_yi[0].size()))  # batch_size
        if self.args.cuda:
            zero_pad = zero_pad.cuda()
        labels_yi = [zero_pad] + labels_yi

        #Generate node features
        zi_s = [z] + zi_s
        zi_c = [z_c] + zi_c
        zi_s = [torch.squeeze(zi_un) for zi_un in zi_s]
        zi_s_ = [torch.cat([zic, zi], 1) for zi, zic in zip(zi_s, zi_c)]

        nodes = [torch.cat([label_yi, zi], 1) for zi, label_yi in zip(zi_s_, labels_yi)]
        nodes = [a.unsqueeze(1) for a in nodes]
        nodes = torch.cat(nodes, 1)


        logits = self.gnn_obj(nodes,adj).squeeze(-1)
        outputs = torch.sigmoid(logits)

        return outputs, logits

    def forward(self, inputs):
        # original forward
        [z_c, z, zi_c, zi_s, labels_yi, _, adj] = inputs
        return self.gnn_iclr_forward(z_c, z, zi_c, zi_s, labels_yi, adj)

    # def forward(self, *inputs):
    #     # tensorboard compatible forward method
    #     [z_c, z, zi_c, zi_s, labels_yi, _, adj] = inputs
    #     return self.gnn_iclr_forward(z_c, z, zi_c, zi_s, labels_yi, adj)

    def compute_adj(self, batch_x, batches_xi):
        """Compute adjacency matrix of Graph Neural Network"""
        x = torch.squeeze(batch_x)
        xi_s = [torch.squeeze(batch_xi) for batch_xi in batches_xi]

        nodes = [x] + xi_s
        nodes = [node.unsqueeze(1) for node in nodes]
        nodes = torch.cat(nodes, 1)
        age = nodes.narrow(2, 0, 1)
        age = age.cpu().numpy()
        gender = nodes.narrow(2, 1, 1)
        gendre = gender.cpu().numpy()
        apoe = nodes.narrow(2, 2, 1)
        apoe = apoe.cpu().numpy()
        edu = nodes.narrow(2, 3, 1)
        edu = edu.cpu().numpy()
        adj = np.ones(
            (self.args.batch_size, self.args.train_N_way * self.args.train_N_shots + 1, self.args.train_N_way * self.args.train_N_shots + 1, 1),
            dtype='float32')+4

        for batch_num in range(self.args.batch_size):
            for i in range(self.args.train_N_way * self.args.train_N_shots + 1):
                for j in range(i + 1, self.args.train_N_way * self.args.train_N_shots + 1):
                    if np.abs(age[batch_num, i, 0] - age[batch_num, j, 0]) <= 0.06:
                        adj[batch_num, i, j, 0] -= 1
                        adj[batch_num, j, i, 0] -= 1
                    if np.abs(edu[batch_num, i, 0] - edu[batch_num, j, 0]) <= 0.14:
                        adj[batch_num, i, j, 0] -= 1
                        adj[batch_num, j, i, 0] -= 1
                    if gendre[batch_num, i, 0] == gendre[batch_num, j, 0]:
                        adj[batch_num, i, j, 0] -= 1
                        adj[batch_num, j, i, 0] -= 1
                    if apoe[batch_num, i, 0] == apoe[batch_num, j, 0]:
                        adj[batch_num, i, j, 0] -= 1
                        adj[batch_num, j, i, 0] -= 1
        adj = 1/adj
        adj = torch.from_numpy(adj)
        return adj.cuda()


def create_models(args, cnn_dim1 = 4):
    return AMGNN(args)

import torch
import math
from tools import utils
from copy import deepcopy
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid


class GraphConvolution(nn.Module):
    """
    Graph Convolutional Layer (Message Passing)
    """
    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        Graph Convolutional Layer Forward
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    """
    2 Layer Graph Convolutional Network.
    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid: int
        number of hidden units
    nclass: int
        size of output dimension
    dropout: float
        dropout rate for GCN
    """
    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 dropout=0.5,
                 lr=0.01,
                 weight_decay=5e-5,
                 device='cpu'
                 ):

        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        """
        Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self,
            x,
            adj,
            label,
            train_mask,
            val_mask,
            epochs=200,
            verbose=True
            ):

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_acc_val = 0

        for i in range(epochs):
            self.train()
            optimizer.zero_grad()
            output = self.forward(x, adj)
            loss_train = F.nll_loss(output[train_mask], label[train_mask])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(x, adj)
            acc_val = utils.accuracy(output[val_mask], label[val_mask])

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                weights = deepcopy(self.state_dict())

            if i % 10 == 0:
                if verbose:
                    print('Epoch {}, Training Loss: {:.4f}, Val Acc: {:.4f}'.format(i, loss_train.item(), acc_val))
        self.load_state_dict(weights)

    def train_model(self,
                    x,
                    adj,
                    label,
                    train_mask,
                    val_mask,
                    test_mask,
                    epochs=200,
                    verbose=True
                    ):

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_acc_val = 0

        for i in range(epochs):
            self.train()
            optimizer.zero_grad()
            output = self.forward(x, adj)
            loss_train = F.nll_loss(output[train_mask], label[train_mask])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(x, adj)
            acc_val = utils.accuracy(output[val_mask], label[val_mask])

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                weights = deepcopy(self.state_dict())

            if i % 10 == 0:
                if verbose:
                    print('Epoch {}, Training Loss: {:.4f}, Val Acc: {:.4f}'.format(i, loss_train.item(), acc_val))
        self.load_state_dict(weights)

        self.eval()
        acc_test = utils.accuracy(self(x, adj)[test_mask], label[test_mask])
        if verbose:
            print('Accuracy on test set: {:.4f}'.format(acc_test))


if __name__ == "__main__":
    from sklearn.decomposition import PCA

    mode = 'train'
    dataset = Planetoid(r'E:\Machine_Learning\Dataset\Processing', 'Cora', transform=T.NormalizeFeatures())
    data = dataset[0].to('cuda')

    adj = utils.edge_index_2_adj(data.edge_index.max().item() + 1, data.edge_index)
    adj_norm = utils.normalize_adj_tensor(adj)

    x_np = data.x.to('cpu').numpy()
    pca = PCA(n_components=256)
    x_np = pca.fit_transform(x_np)
    x = torch.tensor(x_np).to('cuda')

    gcn = GCN(nfeat=x.shape[1],
              nhid=32,
              nclass=data.y.max().item() + 1,
              dropout=0.5,
              lr=0.01,
              weight_decay=5e-5).to('cuda')
    gcn = gcn.to('cuda')

    if mode == 'fit':
        gcn.fit(x=x,
                adj=adj_norm,
                label=data.y,
                train_mask=data.train_mask,
                val_mask=data.val_mask,
                epochs=200,
                verbose=True)
    if mode == 'train':
        gcn.train_model(x=x,
                        adj=adj_norm,
                        label=data.y,
                        train_mask=data.train_mask,
                        val_mask=data.test_mask,
                        test_mask=data.test_mask,
                        epochs=200,
                        verbose=True)

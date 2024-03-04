import torch
import math
from tools import utils
from torch.nn import Module
from typing import Optional
from torch import Tensor
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn as nn
from copy import deepcopy
from torch_geometric.utils import negative_sampling
from module.gae import GAE

MAX_LOGSTD = 10


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


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GraphConvolution(in_channels, 2 * out_channels)
        self.conv_mu = GraphConvolution(2 * out_channels, out_channels)
        self.conv_logstd = GraphConvolution(2 * out_channels, out_channels)

    def forward(self, x, adj):
        x = self.conv1(x, adj).relu()
        return self.conv_mu(x, adj), self.conv_logstd(x, adj)


class VGAE(GAE):
    def __init__(self, encoder: Module, decoder: Optional[Module] = None, lr=0.01, device='cpu'):
        super().__init__(encoder, decoder)

        self.lr = lr
        self.device = device

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs) -> Tensor:
        """"""
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu: Optional[Tensor] = None,
                logstd: Optional[Tensor] = None) -> Tensor:
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (torch.Tensor, optional): The latent space for :math:`\mu`. If
                set to :obj:`None`, uses the last computation of :math:`\mu`.
                (default: :obj:`None`)
            logstd (torch.Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`. (default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))

    def fit(self,
            x,
            train_adj,
            train_pos_edge_label_index,
            epochs=200,
            verbose=True
            ):

        best_loss_val = 100
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(1, epochs + 1):

            self.train()
            optimizer.zero_grad()
            z = self.encode(x, train_adj)
            loss = self.recon_loss(z, train_pos_edge_label_index)
            loss = loss + (1 / train_adj.shape[0]) * self.kl_loss()
            loss.backward()
            optimizer.step()

            if best_loss_val > loss:
                best_loss_val = loss
                model_param = deepcopy(self.state_dict())
            if epoch % 10 == 0:
                if verbose:
                    print('Epoch: {:03d}, Loss: {:.4f}.'.format(epoch, loss))
        self.load_state_dict(model_param)

    def train_model(self,
                    x,
                    train_adj,
                    train_pos_edge_label_index,
                    test_adj,
                    test_pos_edge_label_index,
                    test_neg_edge_label_index,
                    epochs=200,
                    verbose=True
                    ):

        best_loss_val = 100
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(1, epochs + 1):

            self.train()
            optimizer.zero_grad()
            z = self.encode(x, train_adj)
            loss = self.recon_loss(z, train_pos_edge_label_index)
            loss.backward()
            optimizer.step()

            self.eval()
            z = self.encode(x, test_adj)
            auc, ap = self.test(z, test_pos_edge_label_index, test_neg_edge_label_index)

            if best_loss_val > loss:
                best_loss_val = loss
                model_param = deepcopy(self.state_dict())
            if epoch % 10 == 0:
                if verbose:
                    print('Epoch: {:03d}, Loss: {:.4f}, AUC: {:.4f}, AP: {:.4f}.'.format(epoch, loss, auc, ap))
        self.load_state_dict(model_param)


if __name__ == "__main__":
    from sklearn.decomposition import PCA

    mode = 'train'
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False)
    ])
    dataset = Planetoid(r'E:\Machine_Learning\Dataset\Processing', 'cora', transform=transform)
    data = dataset[0]
    test_neg_edge_label_index = negative_sampling(edge_index=data[2].edge_index,
                                                  num_nodes=data[2].edge_index.max().item() + 1,
                                                  num_neg_samples=data[2].edge_index.size(1))

    x_np = data[0].x.to('cpu').numpy()
    pca = PCA(n_components=256)
    x_np = pca.fit_transform(x_np)
    x = torch.tensor(x_np).to('cuda')

    train_adj = utils.edge_index_2_adj(data[0].edge_index.max().item() + 1, data[0].edge_index)
    train_adj_norm = utils.normalize_adj_tensor(train_adj)
    test_adj = utils.edge_index_2_adj(data[2].edge_index.max().item() + 1, data[2].edge_index)
    test_adj_norm = utils.normalize_adj_tensor(test_adj)

    vgae = VGAE(VariationalGCNEncoder(x.shape[1], 16), lr=0.01, device='cuda')
    vgae = vgae.to('cuda')

    if mode == 'fit':
        vgae.fit(x=x,
                 train_adj=train_adj_norm,
                 train_pos_edge_label_index=data[0].pos_edge_label_index,
                 epochs=400,
                 verbose=True)
    if mode == 'train':
        vgae.train_model(x=x,
                         train_adj=train_adj_norm,
                         train_pos_edge_label_index=data[2].pos_edge_label_index,
                         test_adj=test_adj_norm,
                         test_pos_edge_label_index=data[2].pos_edge_label_index,
                         test_neg_edge_label_index=data[2].neg_edge_label_index,
                         epochs=400,
                         verbose=True)

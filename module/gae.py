import torch
import math
from tools import utils
from torch.nn import Module
from typing import Optional, Tuple
from torch import Tensor
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.nn as nn
from copy import deepcopy
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.nn.inits import reset

EPS = 1e-15


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


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()

        self.conv1 = GraphConvolution(in_channels, 2 * out_channels)
        self.conv2 = GraphConvolution(2 * out_channels, out_channels)

    def forward(self, x, adj):
        x = self.conv1(x, adj).relu()
        return self.conv2(x, adj)


class GAE(torch.nn.Module):
    def __init__(self, encoder: Module, decoder: Optional[Module] = None, lr=0.01, device='cpu'):
        super().__init__()

        self.lr = lr
        self.device = device

        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        GAE.reset_parameters(self)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.decoder)

    def forward(self, *args, **kwargs) -> Tensor:  # pragma: no cover
        r"""Alias for :meth:`encode`."""
        return self.encoder(*args, **kwargs)

    def encode(self, *args, **kwargs) -> Tensor:
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs) -> Tensor:
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, z: Tensor, pos_edge_index: Tensor,
             neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)

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

    gae = GAE(GCNEncoder(x.shape[1], 16), lr=0.01, device='cuda')
    gae = gae.to('cuda')

    if mode == 'fit':
        gae.fit(x=x,
                train_adj=train_adj_norm,
                train_pos_edge_label_index=data[0].pos_edge_label_index,
                epochs=400,
                verbose=True)
    if mode == 'train':
        gae.train_model(x=x,
                        train_adj=train_adj_norm,
                        train_pos_edge_label_index=data[2].pos_edge_label_index,
                        test_adj=test_adj_norm,
                        test_pos_edge_label_index=data[2].pos_edge_label_index,
                        test_neg_edge_label_index=data[2].neg_edge_label_index,
                        epochs=400,
                        verbose=True)

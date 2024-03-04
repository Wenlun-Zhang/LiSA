import torch
from tools import utils
from copy import deepcopy
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid


class SGConv(nn.Module):
    def __init__(self, in_channels, out_channels, K=2, bias=True):
        super(SGConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # Apply graph convolution
        for _ in range(self.K):
            x = torch.spmm(adj, x)

        x = torch.matmul(x, self.weight)

        if self.bias is not None:
            x = x + self.bias

        return x

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class SGC(nn.Module):
    """
    2 Hops SGC Network.
    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nclass: int
        size of output dimension
    """
    def __init__(self,
                 nfeat,
                 nclass,
                 K=2,
                 lr=0.01,
                 weight_decay=5e-5,
                 device='cpu'
                 ):

        super(SGC, self).__init__()
        self.sgc = SGConv(nfeat, nclass, K)
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device

    def forward(self, x, adj):
        x = self.sgc(x, adj)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        """
        Initialize parameters of SGC.
        """
        self.sgc.reset_parameters()

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

    sgc = SGC(nfeat=x.shape[1],
              nclass=data.y.max().item() + 1,
              K=2,
              lr=0.01,
              weight_decay=5e-5).to('cuda')
    sgc = sgc.to('cuda')

    if mode == 'fit':
        sgc.fit(x=x,
                adj=adj_norm,
                label=data.y,
                train_mask=data.train_mask,
                val_mask=data.val_mask,
                epochs=200,
                verbose=True)
    if mode == 'train':
        sgc.train_model(x=x,
                        adj=adj_norm,
                        label=data.y,
                        train_mask=data.train_mask,
                        val_mask=data.test_mask,
                        test_mask=data.test_mask,
                        epochs=200,
                        verbose=True)

import torch
from tools import utils
from copy import deepcopy
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    """
    2 Layer GraphSAGE Network.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid: int
        number of hidden units
    nclass: int
        size of output dimension

    """
    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 lr=0.01,
                 weight_decay=5e-5,
                 device='cpu'
                 ):

        super(GraphSAGE, self).__init__()
        self.sageconv1 = SAGEConv(nfeat, nhid, aggr='mean')
        self.sageconv2 = SAGEConv(nhid, nclass, aggr='mean')

        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device

    def forward(self, x, edge_index):
        x = F.relu(self.sageconv1(x, edge_index))
        x = self.sageconv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        """
        Initialize parameters of GraphSAGE.
        """
        self.sageconv1.reset_parameters()
        self.sageconv2.reset_parameters()

    def fit(self,
            x,
            edge_index,
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
            output = self.forward(x, edge_index)
            loss_train = F.nll_loss(output[train_mask], label[train_mask])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(x, edge_index)
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
                    edge_index,
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
            output = self.forward(x, edge_index)
            loss_train = F.nll_loss(output[train_mask], label[train_mask])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(x, edge_index)
            acc_val = utils.accuracy(output[val_mask], label[val_mask])

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                weights = deepcopy(self.state_dict())

            if i % 10 == 0:
                if verbose:
                    print('Epoch {}, Training Loss: {:.4f}, Val Acc: {:.4f}'.format(i, loss_train.item(), acc_val))
        self.load_state_dict(weights)

        self.eval()
        acc_test = utils.accuracy(self(x, edge_index)[test_mask], label[test_mask])
        if verbose:
            print('Accuracy on test set: {:.4f}'.format(acc_test))


if __name__ == "__main__":
    from sklearn.decomposition import PCA

    mode = 'train'
    dataset = Planetoid(r'E:\Machine_Learning\Dataset\Processing', 'Cora', transform=T.NormalizeFeatures())
    data = dataset[0].to('cuda')

    x_np = data.x.to('cpu').numpy()
    pca = PCA(n_components=256)
    x_np = pca.fit_transform(x_np)
    x = torch.tensor(x_np).to('cuda')

    edge_index = data.edge_index

    graphsage = GraphSAGE(nfeat=x.shape[1],
                          nhid=32,
                          nclass=data.y.max().item() + 1,
                          lr=0.01,
                          weight_decay=5e-5).to('cuda')
    graphsage = graphsage.to('cuda')

    if mode == 'fit':
        graphsage.fit(x=x,
                      edge_index=edge_index,
                      label=data.y,
                      train_mask=data.train_mask,
                      val_mask=data.val_mask,
                      epochs=200,
                      verbose=True)
    if mode == 'train':
        graphsage.train_model(x=x,
                              edge_index=edge_index,
                              label=data.y,
                              train_mask=data.train_mask,
                              val_mask=data.test_mask,
                              test_mask=data.test_mask,
                              epochs=200,
                              verbose=True)

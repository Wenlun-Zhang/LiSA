import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tools import utils


class nia(nn.Module):
    def __init__(self,
                 surrogate,
                 nnodes,
                 ori_edge_index,
                 ori_feat,
                 num_victim=1000,
                 link_rate=0.5,
                 device='cpu'
                 ):
        """
        Node Injection Attack with Given Link Generation Rate.
        :param surrogate: Node classification surrogate model.
        :param nnodes: Node number of original graph.
        :param ori_edge_index: Edge index of original graph.
        :param ori_feat: Feature of original graph.
        :param num_victim: Number of targeted victim nodes.
        :param link_rate: Given link generation rate.
        :param device: Training device: cuda or cpu or mps.
        """
        super(nia, self).__init__()

        self.surrogate = surrogate
        self.nnodes = nnodes
        self.ori_edge_index = ori_edge_index
        self.ori_feat = ori_feat
        self.num_victim = num_victim
        self.link_rate = link_rate
        self.device = device

        self.ori_adj = utils.edge_index_2_adj(self.nnodes, self.ori_edge_index)
        self.victim_nodes = self._target_victim_nodes()
        self.adj, self.feat, self.adj_for_train = self._initialize_graph()
        self.adj_norm = utils.normalize_adj_tensor(self.adj)
        self.adj_for_train_norm = utils.normalize_adj_tensor(self.adj_for_train)
        self.target_nodes_mask = self._target_nodes_mask()
        self.attack_label = None

    def _initialize_graph(self):
        """
        Initialize the graph with random selected victims and link generation rate.
        :return: Initialized graph adjacency matrix (Valid and Train) and features.
        """
        # Initialize aggressor nodes feature
        aggressor_feat = torch.zeros((self.num_victim, self.ori_feat.size(1)),
                                     dtype=self.ori_feat.dtype,
                                     device=self.device)
        # Merge feature
        feat = torch.cat((self.ori_feat, aggressor_feat), dim=0)

        # List up victim and aggressor nodes
        edges_from = self.victim_nodes
        edges_to = torch.arange(self.nnodes, self.nnodes + self.num_victim, dtype=torch.long, device=self.device)
        # Generate edge indices according to link generation rate
        mask = torch.rand(self.num_victim, device=self.device) < self.link_rate
        true_edges_from = edges_from[mask]
        true_edges_to = edges_to[mask]
        # Always connect aggressor nodes to their corresponding victim nodes for training
        train_edges_from = edges_from
        train_edges_to = edges_to
        # Merge edge indices
        true_edge_index = torch.cat((self.ori_edge_index, torch.stack((true_edges_from, true_edges_to), dim=0),
                                     torch.stack((true_edges_to, true_edges_from), dim=0)), dim=1)
        train_edge_index = torch.cat((self.ori_edge_index, torch.stack((train_edges_from, train_edges_to), dim=0),
                                      torch.stack((train_edges_to, train_edges_from), dim=0)), dim=1)
        # Update adjacency matrix
        adj = utils.edge_index_2_adj(feat.size(0), true_edge_index)
        adj_for_train = utils.edge_index_2_adj(feat.size(0), train_edge_index)

        return adj, feat, adj_for_train

    def _target_victim_nodes(self):
        # Random select victim nodes
        victim_nodes = np.random.choice(self.nnodes, self.num_victim, replace=False)
        victim_nodes = torch.tensor(victim_nodes, dtype=torch.long, device=self.device)

        return victim_nodes

    def _target_nodes_mask(self):
        """
        Generate mask tensor for training.
        :return: Data mask for target victim node.
        """
        target_nodes_mask = torch.zeros(self.nnodes + self.num_victim, dtype=torch.bool, device=self.device)
        target_nodes_mask[self.victim_nodes] = True

        return target_nodes_mask

    def _cls_loss(self, output, labels):
        """
        Loss function for training surrogate node classification model.
        :param output: Classifier output.
        :param labels: True labels.
        :return: Node classification loss.
        """
        loss = F.nll_loss(output, labels)

        return loss

    def _atk_cls_loss(self, output, labels):
        """
        Loss function for training subgraph on attacking node classification model.
        :param output: Classifier output.
        :param labels: True labels.
        :return: Attack loss on node classification.
        """
        return self._cls_loss(output, labels)

    def get_attack_label(self, label_cls_model, label, train_mask, val_mask):
        """
        Use a node classification model to detect the second-highest class on victim node.
        :param label_cls_model: Node classifier model.
        :param label: Label of original graph.
        :param train_mask: Training node mask.
        :param val_mask: Validation node mask.
        """
        # Train a node classification model.
        adj_norm = utils.normalize_adj_tensor(self.ori_adj)
        label_cls_model.fit(x=self.ori_feat,
                            adj=adj_norm,
                            label=label,
                            train_mask=train_mask,
                            val_mask=val_mask,
                            epochs=1000,
                            verbose=False)

        label_cls_model.eval()  # Set the model to evaluation mode.
        with torch.no_grad():  # Disable gradient computation.
            output = label_cls_model(self.ori_feat, self.ori_adj)  # Get the output from the GCN model.
        # Get the top 2 predicted label for the victim node.
        top2_pred = output[self.victim_nodes].topk(2, dim=1)[1]
        # Update the attack labels for the victim nodes to the second-highest label.
        self.attack_label = label.clone().detach()
        for i, victim_node in enumerate(self.victim_nodes):
            self.attack_label[victim_node] = top2_pred[i][1]  # Assign the second-highest predicted label.
        # Append zeros for aggressor nodes.
        self.attack_label = torch.cat([self.attack_label, torch.zeros(self.num_victim, dtype=self.attack_label.dtype, device=self.device)])

    def train_nodes(self,
                    label,
                    train_mask,
                    lr=1.0,
                    train_epochs_per_iter=5,
                    total_iters=200,
                    verbose=True):
        """
        Train nodes to perform NIA.
        :param label: Label of original graph.
        :param train_mask: Node mask for node classification model training.
        :param lr: Learning rate for node features optimization.
        :param train_epochs_per_iter: Inner iterations per node training loop.
        :param total_iters: Total iterations that trains the nodes.
        :param verbose: Print training information.
        """
        # Surrogate models setup.
        surrogate = self.surrogate
        # Prepare label and node mask for node classifier training that adapt to merged graph.
        train_mask = torch.cat([train_mask, torch.zeros(self.num_victim, dtype=torch.bool, device=self.device)])
        label = torch.cat([label, torch.zeros(self.num_victim, dtype=label.dtype, device=self.device)])
        # Optimizer setup.
        optimizer = optim.Adam(surrogate.parameters(), lr=0.01)
        # Default with eval mode.
        surrogate.eval()
        # Adversarial training.
        for t in range(total_iters):
            # Create tensor with gradient activated for node training.
            feat_grad_tensor = utils.grad_tensor(self.feat)
            # Inner iteration for surrogate training.
            for n in range(train_epochs_per_iter):
                # Switch to train mode.
                surrogate.train()
                # Model output.
                output = surrogate(feat_grad_tensor, self.adj_norm)
                # Loss calculation.
                loss = self._cls_loss(output[train_mask], label[train_mask])
                # Back propagation.
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            # Outer iteration for training nodes.
            # Switch to eval mode.
            surrogate.eval()
            # Output of surrogate model.
            output = surrogate(feat_grad_tensor, self.adj_for_train_norm)
            # Attack loss calculation.
            atk_loss = self._atk_cls_loss(output[self.target_nodes_mask], self.attack_label[self.target_nodes_mask])
            # Update node features based on gradient.
            feat_grad = torch.autograd.grad(atk_loss, feat_grad_tensor, retain_graph=True)[0]
            self.feat[self.nnodes:, :] = self.feat[self.nnodes:, :] - lr * feat_grad[self.nnodes:, :]
            if verbose:
                print("Train model: Cls Loss = {:.2f}.\nTrain nodes: Atk Cls Loss = {:.2f}.".format(loss, atk_loss))


if __name__ == "__main__":
    import torch_geometric.transforms as T
    from gcn import GCN
    from module.sgc import SGC
    from module.graphsage import GraphSAGE
    from torch_geometric.datasets import Planetoid, Amazon, FacebookPagePage
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split

    # Dataset Root.
    dataset_dir = r'E:\Machine_Learning\Dataset\Processing'
    # Device set.
    device = 'cuda'
    # Random seed.
    utils.setup_seed(6666)
    # Data split rate.
    train_ratio = 0.05
    val_ratio = 0.20
    test_ratio = 0.35

    # -------------------------------> Confirm settings <-------------------------------
    # Dataset Selection.
    dataset = Planetoid(dataset_dir, 'Cora', transform=T.NormalizeFeatures())
    # dataset = Amazon(dataset_dir, 'Photo', transform=T.NormalizeFeatures())
    # dataset = Planetoid(dataset_dir, 'PubMed', transform=T.NormalizeFeatures())
    # dataset = FacebookPagePage(dataset_dir, transform=T.NormalizeFeatures())

    data = dataset[0].to(device)

    # -------------------------------> Confirm settings <-------------------------------
    # Data split if required.
    # indices = list(range(data.num_nodes))
    # train_indices, remaining_indices = train_test_split(indices, train_size=train_ratio, random_state=6666)
    # val_indices, remaining_indices = train_test_split(remaining_indices, train_size=val_ratio / (1 - train_ratio), random_state=6666)
    # test_indices, _ = train_test_split(remaining_indices, train_size=test_ratio / (1 - train_ratio - val_ratio), random_state=6666)
    # train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    # val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    # test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    # train_mask[train_indices] = True
    # val_mask[val_indices] = True
    # test_mask[test_indices] = True
    # data.train_mask = train_mask
    # data.val_mask = val_mask
    # data.test_mask = test_mask

    # PCA dimensionality reduction on features.
    x_np = data.x.cpu().numpy()
    pca = PCA(n_components=256)
    x_np = pca.fit_transform(x_np)
    x = torch.tensor(x_np).to(device)
    # Node mask.
    train_cls_mask = data.train_mask
    val_cls_mask = data.val_mask
    test_cls_mask = data.test_mask
    # Classifier model for get the second-highest label.
    label_cls_model = GCN(nfeat=x.shape[1],
                          nhid=32,
                          nclass=data.y.max().item() + 1,
                          dropout=0.5,
                          lr=0.01,
                          weight_decay=5e-5).to(device)
    # Surrogate model for NIA.
    surrogate = GCN(nfeat=x.shape[1],
                    nhid=32,
                    nclass=data.y.max().item() + 1,
                    dropout=0.5,
                    lr=0.01,
                    weight_decay=5e-5).to(device)

    # -------------------------------> Confirm settings <-------------------------------
    # Model initialization.
    model = nia(surrogate=surrogate,
                nnodes=x.shape[0],
                ori_edge_index=data.edge_index,
                ori_feat=x,
                num_victim=1000,
                link_rate=0.5,
                device=device)

    # Get the second-highest label for adversarial training.
    model.get_attack_label(label_cls_model=label_cls_model,
                           label=data.y,
                           train_mask=train_cls_mask,
                           val_mask=val_cls_mask)
    # Train nodes.
    model.train_nodes(label=data.y,
                      train_mask=train_cls_mask,
                      lr=1.0,
                      train_epochs_per_iter=10,
                      total_iters=1000,
                      verbose=True)

    # -------------------------------> Confirm settings <-------------------------------
    # Classifier model.
    classifier = GCN(nfeat=x.shape[1],
                     nhid=32,
                     nclass=data.y.max().item() + 1,
                     dropout=0.5,
                     lr=0.01,
                     weight_decay=5e-5).to(device)
    # classifier = SGC(nfeat=x.shape[1],
    #                  nclass=data.y.max().item() + 1,
    #                  K=2,
    #                  lr=0.01,
    #                  weight_decay=5e-5).to(device)
    # classifier = GraphSAGE(nfeat=x.shape[1],
    #                        nhid=32,
    #                        nclass=data.y.max().item() + 1,
    #                        lr=0.01,
    #                        weight_decay=5e-5).to(device)

    # Label and node mask on merged graph.
    label = torch.cat([data.y, torch.zeros(model.num_victim, dtype=data.y.dtype, device=device)])
    train_cls_mask = torch.cat([train_cls_mask, torch.zeros(model.num_victim, dtype=torch.bool, device=device)])
    val_cls_mask = torch.cat([val_cls_mask, torch.zeros(model.num_victim, dtype=torch.bool, device=device)])
    test_cls_mask = torch.cat([test_cls_mask, torch.zeros(model.num_victim, dtype=torch.bool, device=device)])

    # -------------------------------> Confirm settings <-------------------------------
    # Train classifier and evaluate performance on victim node.
    classifier.train_model(x=model.feat,
                           adj=model.adj_norm,
                           label=label,
                           train_mask=train_cls_mask,
                           val_mask=val_cls_mask,
                           test_mask=model.target_nodes_mask,
                           epochs=1000,
                           verbose=True)
    # classifier.train_model(x=model.feat,
    #                        edge_index=utils.adj_2_edge_index(model.adj),
    #                        label=label,
    #                        train_mask=train_cls_mask,
    #                        val_mask=val_cls_mask,
    #                        test_mask=test_cls_mask,
    #                        epochs=1000,
    #                        verbose=True)

    # Print node features.
    print("Features of Injected Nodes:")
    print(model.feat[model.nnodes:, :])

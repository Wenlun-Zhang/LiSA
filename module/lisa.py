import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from tools import utils

EPS = 1e-15


class LiSA(nn.Module):
    def __init__(self,
                 surrogate_link,
                 surrogate_cls,
                 nnodes,
                 ori_edge_index,
                 ori_feat,
                 victim_node=0,
                 sub_size=5,
                 num_sub_edges=3,
                 device='cpu'
                 ):
        """
        Link Recommender-Subgraph Injection Attack.
        :param surrogate_link: Link prediction surrogate model.
        :param surrogate_cls: Node classification surrogate model.
        :param nnodes: Node number of original graph.
        :param ori_edge_index: Edge index of original graph.
        :param ori_feat: Feature of original graph.
        :param victim_node: Attack target node.
        :param sub_size: Subgraph size.
        :param num_sub_edges: Number of edges in subgraph.
        :param device: Training device: cuda or cpu or mps.
        """
        super(LiSA, self).__init__()

        assert victim_node < nnodes, "Assign a victim node in the graph!"
        assert num_sub_edges <= sub_size * (sub_size - 1) // 2, "Number of edges exceeds the maximum possible number of edges in the subgraph."

        self.surrogate_link = surrogate_link
        self.surrogate_cls = surrogate_cls
        self.nnodes = nnodes
        self.ori_edge_index = ori_edge_index
        self.ori_feat = ori_feat
        self.victim_node = victim_node
        self.sub_size = sub_size
        self.num_sub_edges = num_sub_edges
        self.device = device

        self.ori_adj = utils.edge_index_2_adj(self.nnodes, self.ori_edge_index)
        self.sub_adj, self.sub_feat = self._initialize_subgraph()
        self.target_links = self._target_links()
        self.target_node_mask = self._target_node_mask()
        self.attack_label = None

    def _initialize_subgraph(self):
        """
        Initialize a subgraph with the given parameters.
        :return: Initialized subgraph adjacency matrix and subgraph features.
        """
        # Initialize feature.
        sub_feat = torch.zeros(self.sub_size, self.ori_feat.shape[1], device=self.device)
        # Initialize adjacency matrix.
        sub_adj = torch.zeros((self.sub_size, self.sub_size), device=self.device)
        # Generate random edges until the desired number of edges is reached.
        edges_added = 0
        added_edges = set()
        while edges_added < self.num_sub_edges:
            node_a, node_b = np.random.choice(self.sub_size, 2, replace=False)
            # Ensure no self-loops and no duplicate edges.
            if node_a != node_b and (node_a, node_b) not in added_edges and (node_b, node_a) not in added_edges:
                sub_adj[node_a, node_b] = 1
                sub_adj[node_b, node_a] = 1
                added_edges.add((node_a, node_b))
                edges_added += 1

        return sub_adj, sub_feat

    def _update_edges(self, sub_adj, link_adj_grad, cls_adj_grad, beta, edge_perturbations):
        """
        Update the edges of the subgraph based on gradients.
        :param sub_adj: Adjacency matrix of the subgraph.
        :param link_adj_grad: Gradient of the adjacency matrix with respect to the link prediction loss.
        :param cls_adj_grad: Gradient of the adjacency matrix with respect to the classification loss.
        :param beta: Balance parameter between gradient of link predictor and node classifier.
        :param edge_perturbations: Number of edge perturbations to perform.
        :return: Updated adjacency matrix.
        """
        # Calculate total gradient and make it symmetric.
        total_grad = link_adj_grad + cls_adj_grad * beta
        total_grad = total_grad + total_grad.t()
        # Set diagonal to '0' to avoid self-loop.
        total_grad.fill_diagonal_(0)
        # Extract the gradient for the subgraph.
        sub_grad = total_grad[-self.sub_size:, -self.sub_size:]
        # Use only the upper triangle of the gradient matrix for perturbations.
        sub_grad_triu = torch.triu(sub_grad, diagonal=1)
        # Mask the lower triangle and diagonal of 'no_edge' and 'edge' masks to avoid considering them.
        mask_no_edge = torch.triu(sub_adj == 0, diagonal=1)
        mask_edge = torch.triu(sub_adj == 1, diagonal=1)
        # For adding edges: set entries with edges, lower triangle, and diagonal to inf.
        no_edge_grad = sub_grad_triu.clone()
        no_edge_grad[~mask_no_edge] = float('inf')
        # For removing edges: set entries without edges, lower triangle and diagonal to -inf.
        edge_grad = sub_grad_triu.clone()
        edge_grad[~mask_edge] = -float('inf')
        # Find indices of the top gradients for adding and removing edges.
        no_edge_indices = torch.topk(no_edge_grad.view(-1), edge_perturbations, largest=False).indices
        edge_indices = torch.topk(edge_grad.view(-1), edge_perturbations).indices

        # Perform edge perturbations.
        for idx in no_edge_indices:
            row = idx // sub_adj.size(0)
            col = idx % sub_adj.size(0)
            sub_adj[row, col] = 1
            sub_adj[col, row] = 1

        for idx in edge_indices:
            row = idx // sub_adj.size(0)
            col = idx % sub_adj.size(0)
            sub_adj[row, col] = 0
            sub_adj[col, row] = 0

        return sub_adj

    def _target_node_mask(self):
        """
        Generate mask tensor for training.
        :return: Data mask for target victim node.
        """
        target_node_mask = torch.zeros(self.nnodes + self.sub_size, dtype=torch.bool, device=self.device)
        target_node_mask[self.victim_node] = True

        return target_node_mask

    def _target_links(self):
        """
        Generate target links for training.
        :return: Edge index of target adversarial edges.
        """
        links = torch.zeros((2, self.sub_size), dtype=torch.int, device=self.device)
        for i in range(self.sub_size):
            links[0, i] = self.victim_node
            links[1, i] = self.nnodes + i

        return links

    def _link_loss(self, z, pos_edge_index, neg_edge_index=None):
        """
        Loss function for training surrogate link prediction model.
        :param z: Representation of graph.
        :param pos_edge_index: Positive edge target.
        :param neg_edge_index: Negative edge target.
        :return: Link prediction loss.
        """
        pos_loss = -torch.log(self.surrogate_link.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.surrogate_link.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss

    def _cls_loss(self, output, labels):
        """
        Loss function for training surrogate node classification model.
        :param output: Classifier output.
        :param labels: True labels.
        :return: Node classification loss.
        """
        loss = F.nll_loss(output, labels)

        return loss

    def _atk_link_loss(self, z, target_link, neg_edge_index=None):
        """
        Loss function for training subgraph on attacking link prediction model.
        :param z: Representation of graph.
        :param target_link: Positive edge target.
        :param neg_edge_index: Negative edge target.
        :return: Attack loss on link prediction.
        """
        return self._link_loss(z, target_link, neg_edge_index)

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
        top2_pred = output.topk(2, dim=1)[1]
        second_highest_label = top2_pred[self.victim_node][1]

        # Initialize attack_labels with true_labels.
        self.attack_label = label.clone()
        # Update the victim node's label to second-highest label.
        self.attack_label[self.victim_node] = second_highest_label
        self.attack_label = torch.cat([self.attack_label, torch.zeros(self.sub_size, dtype=self.attack_label.dtype, device=self.device)])

    def recommend_links(self, adj, z, n_links=5):
        """
        Simulate users' action towards a link recommender.
        The user would add n_links connections according to the probability from a link predictor.
        :param adj: Merged graph of isolated original graph and subgraph.
        :param z: Extracted representation from the merged graph.
        :param n_links: Number of links that the users would add by the link recommender.
        :return: Adjacency matrix after victim user add edges by the link recommender.
        """
        recommended_adj = adj.clone()
        # Decode the edge probability score of predicted adjacency matrix.
        score = torch.sigmoid(torch.matmul(z, z.t()))
        # Average score of upper and lower triangle.
        score = score + score.t()
        # Avoid self-loop.
        score.fill_diagonal_(0)
        # Ignore edges that already exist on victim node.
        score[self.victim_node, recommended_adj[self.victim_node] == 1] = 0
        score[recommended_adj[self.victim_node] == 1, self.victim_node] = 0

        # Add n_links edges with the highest probability score.
        _, top_indices = torch.topk(score[self.victim_node], n_links)
        for target_node in top_indices:
            recommended_adj[self.victim_node, target_node] = 1
            recommended_adj[target_node, self.victim_node] = 1

        return recommended_adj

    def train_subgraph(self,
                       x,
                       label,
                       train_cls_mask,
                       train_link_edge_index,
                       train_link_edge_label_index,
                       alpha=1.1,
                       beta=1.0,
                       lr=1.0,
                       edge_perturbations=1,
                       n_links=3,
                       train_epochs_per_iter=5,
                       total_iters=200,
                       verbose=True):
        """
        Train subgraph to perform SIA.
        :param x: Features for link prediction model training.
        :param label: Label of original graph.
        :param train_cls_mask: Node mask for node classification model training.
        :param train_link_edge_index: Edge index for link prediction model training.
        :param train_link_edge_label_index: Target edges for link prediction model training.
        :param alpha: Balance parameter to adjust attack loss.
        :param beta: Balance parameter for subgraph structure optimization.
        :param lr: Learning rate for subgraph features optimization.
        :param edge_perturbations: Number of edge perturbations to be performed.
        :param n_links: Links that link recommender would generate.
        :param train_epochs_per_iter: Inner iterations per subgraph training loop.
        :param total_iters: Total iterations that trains the subgraph.
        :param verbose: Print training information.
        :return: Adjacency matrix and features of subgraph for SIA.
        """
        # Surrogate models setup.
        surrogate_link = self.surrogate_link
        surrogate_cls = self.surrogate_cls
        # Derive adjacency matrix for link predictor training.
        train_adj = utils.edge_index_2_adj(self.nnodes, train_link_edge_index)
        train_adj_norm = utils.normalize_adj_tensor(train_adj)
        # Prepare label and node mask for node classifier training that adapt to merged graph.
        train_cls_mask = torch.cat([train_cls_mask, torch.zeros(self.sub_size, dtype=torch.bool, device=self.device)])
        label = torch.cat([label, torch.zeros(self.sub_size, dtype=label.dtype, device=self.device)])
        # Optimizer setup.
        optimizer_link = optim.Adam(surrogate_link.parameters(), lr=0.01)
        optimizer_cls = optim.Adam(surrogate_cls.parameters(), lr=0.01)
        # Default with eval mode.
        surrogate_link.eval()
        surrogate_cls.eval()
        # Initialize sub_adj and sub_feat for subgraph training.
        sub_adj = self.sub_adj
        sub_feat = self.sub_feat
        # Adversarial training.
        for t in range(total_iters):
            # Merge graphs.
            combined_adj = utils.merge_adj(self.ori_adj, sub_adj)
            combined_feat = utils.merge_feat(self.ori_feat, sub_feat)
            # Adjacency matrix laplacian.
            combined_adj_norm = utils.normalize_adj_tensor(combined_adj)
            # Create tensor with gradient activated for subgraph training.
            combined_adj_norm_grad = utils.grad_tensor(combined_adj_norm)
            combined_feat_grad = utils.grad_tensor(combined_feat)
            # Inner iteration for link predictor and node classifier training.
            for n in range(train_epochs_per_iter):
                # Switch to train mode.
                surrogate_link.train()
                surrogate_cls.train()
                # Model output.
                z_link = surrogate_link.encode(x, train_adj_norm)
                output_cls = surrogate_cls(combined_feat, combined_adj_norm)
                # Loss calculation.
                loss_link = self._link_loss(z_link, train_link_edge_label_index)
                loss_cls = self._cls_loss(output_cls[train_cls_mask], label[train_cls_mask])
                # Back propagation.
                optimizer_link.zero_grad()
                optimizer_cls.zero_grad()
                loss_link.backward(retain_graph=True)
                loss_cls.backward(retain_graph=True)
                optimizer_link.step()
                optimizer_cls.step()
            # Outer iteration for training subgraph.
            # Switch to eval mode.
            surrogate_link.eval()
            surrogate_cls.eval()
            # Encode representation based on merged new graph of last training iteration.
            z_link = surrogate_link.encode(combined_feat_grad, combined_adj_norm_grad)
            # Simulation of link recommender behavior.
            recommended_adj = self.recommend_links(combined_adj, z_link, n_links)
            recommended_adj_norm = utils.normalize_adj_tensor(recommended_adj)
            recommended_adj_norm_grad = utils.grad_tensor(recommended_adj_norm)
            # Output of node classifier after link recommender operation.
            output_cls = surrogate_cls(combined_feat_grad, recommended_adj_norm_grad)
            # Attack loss calculation.
            atk_link_loss = self._atk_link_loss(z_link, self.target_links)
            atk_cls_loss = self._atk_cls_loss(output_cls[self.target_node_mask], self.attack_label[self.target_node_mask])
            atk_loss = alpha * atk_link_loss + atk_cls_loss
            # Alternative training on feature and adjacency matrix.
            if t % 2 == 0:
                # For even iterations: optimize structure.
                # Adjacency matrix gradient of link predictor based on atk_loss and adjacency matrix without edge recommended.
                link_adj_grad = torch.autograd.grad(atk_link_loss, combined_adj_norm_grad, retain_graph=True)[0]
                # Adjacency matrix gradient of node classifier based on atk_loss and adjacency matrix after edge recommended.
                cls_adj_grad = torch.autograd.grad(atk_cls_loss, recommended_adj_norm_grad, retain_graph=True)[0]
                sub_adj = self._update_edges(sub_adj=self.sub_adj,
                                             link_adj_grad=link_adj_grad,
                                             cls_adj_grad=cls_adj_grad,
                                             beta=beta,
                                             edge_perturbations=edge_perturbations)
            else:
                # For odd iterations: optimize features.
                # Feature gradient based on atk_loss and merged feature.
                feat_grad = torch.autograd.grad(atk_loss, combined_feat_grad, retain_graph=True)[0]
                sub_feat = sub_feat - lr * feat_grad[self.nnodes:, :]
            if verbose:
                print("Train model: Link Loss = {:.2f}, Cls Loss = {:.2f}.\nTrain subgraph: Atk Link Loss = {:.2f}, Atk Cls Loss = {:.2f}, Atk Loss = {:.2f}.".format(
                        loss_link, loss_cls, atk_link_loss, atk_cls_loss, atk_loss))

        return sub_adj, sub_feat


if __name__ == "__main__":
    import torch_geometric.transforms as T
    from gcn import GCN
    from gae import GAE, GCNEncoder
    from torch_geometric.datasets import Planetoid
    from sklearn.decomposition import PCA

    # Random seed.
    utils.setup_seed(6666)
    # Graph transform setup.
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False)
    ])
    # Data for node classification.
    dataset_cls = Planetoid(r'E:\Machine_Learning\Dataset\Processing', 'Cora', transform=T.NormalizeFeatures())
    data_cls = dataset_cls[0].to('cuda')
    # PCA dimensionality reduction on features.
    x_np = data_cls.x.cpu().numpy()
    pca = PCA(n_components=256)
    x_np = pca.fit_transform(x_np)
    x = torch.tensor(x_np).to('cuda')
    # Edge index and adjacency matrix of original graph.
    ori_edge_index = data_cls.edge_index.to('cuda')
    ori_adj = utils.edge_index_2_adj(x.shape[0], ori_edge_index)
    # Node mask.
    train_cls_mask = data_cls.train_mask
    val_cls_mask = data_cls.val_mask
    test_cls_mask = data_cls.test_mask
    # Data for link prediction.
    dataset_link = Planetoid(r'E:\Machine_Learning\Dataset\Processing', 'Cora', transform=transform)
    data_link = dataset_link[0]
    # Edge indices for training link prediction model.
    train_link_edge_index = data_link[0].edge_index.to('cuda')
    train_link_pos_edge_label_index = data_link[0].pos_edge_label_index.to('cuda')
    test_link_edge_index = data_link[2].edge_index.to('cuda')
    test_link_pos_edge_label_index = data_link[2].pos_edge_label_index.to('cuda')
    test_link_neg_edge_label_index = data_link[2].neg_edge_label_index.to('cuda')
    # Classifier model for get the second-highest label.
    label_cls_model = GCN(nfeat=x.shape[1],
                          nhid=32,
                          nclass=data_cls.y.max().item() + 1,
                          dropout=0.5,
                          lr=0.01,
                          weight_decay=5e-5).to('cuda')
    # Surrogate link prediction model for LiSA.
    surrogate_link = GAE(GCNEncoder(x.shape[1], 16),
                         lr=0.01,
                         device='cuda').to('cuda')
    # Surrogate node classification model for LiSA.
    surrogate_cls = GCN(nfeat=x.shape[1],
                        nhid=32,
                        nclass=data_cls.y.max().item() + 1,
                        dropout=0.5,
                        lr=0.01,
                        weight_decay=5e-5).to('cuda')
    # Model initialization.
    model = LiSA(surrogate_link=surrogate_link,
                 surrogate_cls=surrogate_cls,
                 nnodes=x.shape[0],
                 ori_edge_index=ori_edge_index,
                 ori_feat=x,
                 victim_node=1000,
                 sub_size=5,
                 num_sub_edges=3,
                 device='cuda')
    # Get the second-highest label for adversarial training.
    model.get_attack_label(label_cls_model=label_cls_model,
                           label=data_cls.y,
                           train_mask=train_cls_mask,
                           val_mask=val_cls_mask)
    # Train subgraph.
    sub_adj, sub_feat = model.train_subgraph(x=x,
                                             label=data_cls.y,
                                             train_cls_mask=train_cls_mask,
                                             train_link_edge_index=train_link_edge_index,
                                             train_link_edge_label_index=train_link_pos_edge_label_index,
                                             alpha=1.1,
                                             beta=1.0,
                                             lr=1.0,
                                             edge_perturbations=1,
                                             n_links=3,
                                             train_epochs_per_iter=5,
                                             total_iters=200,
                                             verbose=True)
    # Link recommender model.
    recommender = GAE(GCNEncoder(x.shape[1], 16),
                      lr=0.01,
                      device='cuda').to('cuda')
    # Classifier model.
    classifier = GCN(nfeat=x.shape[1],
                     nhid=32,
                     nclass=data_cls.y.max().item() + 1,
                     dropout=0.5,
                     lr=0.01,
                     weight_decay=5e-5).to('cuda')
    # Link recommender training data.
    train_link_adj = utils.edge_index_2_adj(x.shape[0] + 5, train_link_edge_index)
    train_link_adj_norm = utils.normalize_adj_tensor(train_link_adj)
    train_link_neg_edge_label_index = negative_sampling(edge_index=train_link_pos_edge_label_index,
                                                        num_nodes=train_link_pos_edge_label_index.max().item() + 1,
                                                        num_neg_samples=train_link_pos_edge_label_index.size(1))
    test_link_adj = utils.edge_index_2_adj(x.shape[0] + 5, test_link_edge_index)
    test_link_adj_norm = utils.normalize_adj_tensor(test_link_adj)
    # Label and node mask on merged graph.
    label = torch.cat([data_cls.y, torch.zeros(5, dtype=data_cls.y.dtype, device='cuda')])
    train_cls_mask = torch.cat([train_cls_mask, torch.zeros(5, dtype=torch.bool, device='cuda')])
    val_cls_mask = torch.cat([val_cls_mask, torch.zeros(5, dtype=torch.bool, device='cuda')])
    test_cls_mask = torch.cat([test_cls_mask, torch.zeros(5, dtype=torch.bool, device='cuda')])
    # Merge graph.
    combined_adj = utils.merge_adj(ori_adj, sub_adj)
    combined_adj_norm = utils.normalize_adj_tensor(combined_adj)
    combined_feat = utils.merge_feat(x, sub_feat)
    # Derive adjacency matrix by link recommender.
    recommender.train_model(x=combined_feat,
                            train_adj=train_link_adj_norm,
                            train_pos_edge_label_index=train_link_pos_edge_label_index,
                            test_adj=test_link_adj_norm,
                            test_pos_edge_label_index=test_link_pos_edge_label_index,
                            test_neg_edge_label_index=test_link_neg_edge_label_index,
                            epochs=100,
                            verbose=True)
    z = recommender.encode(combined_feat, combined_adj_norm)
    recommended_adj = model.recommend_links(combined_adj, z, 3)
    recommended_adj_norm = utils.normalize_adj_tensor(recommended_adj)
    # Train classifier and evaluate performance on victim node.
    classifier.train_model(x=combined_feat,
                           adj=recommended_adj_norm,
                           label=label,
                           train_mask=train_cls_mask,
                           val_mask=val_cls_mask,
                           test_mask=test_cls_mask,
                           epochs=100,
                           verbose=True)
    # Detect new connections.
    link_users = utils.detect_new_connections(combined_adj, recommended_adj, model.victim_node)
    # Print information.
    print("Adjacency Matrix of Subgraph:")
    print(sub_adj)
    print("Features of Subgraph:")
    print(sub_feat)
    print("Victim node is {}.".format(model.victim_node))
    print("New connections by Recommendation System:")
    print(link_users)
    # Evaluate victim performance.
    output = classifier(combined_feat, recommended_adj_norm)
    pred = output[model.target_node_mask].max(1)[1]

    if any(link_users >= x.shape[0]):
        print("Link Generation Succeeds.")
    else:
        print("Link Generation Fails.")

    if pred.item() != label[model.target_node_mask].item():
        print('Classification Fails.')
    else:
        print('Classification Succeeds.')

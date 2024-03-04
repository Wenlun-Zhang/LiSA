import torch
import torch.nn as nn
from tools import utils


class GraphCopy(nn.Module):
    def __init__(self,
                 nnodes,
                 ori_edge_index,
                 ori_feat,
                 victim_node=0,
                 n_hop=2,
                 feat_perturb=0.1,
                 device='cpu'
                 ):
        """
        GraphCopy attack method: Copy target node with n_hop neighborhood and perturb the features.
        :param nnodes: Node number of original graph.
        :param ori_edge_index: Edge index of original graph.
        :param ori_feat: Feature of original graph.
        :param victim_node: Attack target node.
        :param n_hop: n_hop-hop neighborhood to be copied.
        :param feat_perturb: Percentage of standard deviation perturbation on features of copied nodes.
        :param device: Training device: cuda or cpu or mps.
        """
        super(GraphCopy, self).__init__()

        assert victim_node < nnodes, "Assign a victim node in the graph!"

        self.nnodes = nnodes
        self.ori_edge_index = ori_edge_index
        self.ori_feat = ori_feat
        self.victim_node = victim_node
        self.n_hop = n_hop
        self.feat_perturb = feat_perturb
        self.device = device

        self.ori_adj = utils.edge_index_2_adj(self.nnodes, self.ori_edge_index)
        self.sub_adj, self.sub_feat = self._generate_subgraph()
        self.target_node_mask = self._target_node_mask()

    def _generate_subgraph(self):
        """
        Generate a subgraph via GraphCopy method.
        :return: Subgraph adjacency matrix and subgraph features.
        """
        # Get n_hop neighborhood of victim node.
        hop_neighbors = self._find_n_hop_neighbors(self.victim_node, self.n_hop, self.ori_adj)
        # Include the victim node.
        subgraph_nodes = [self.victim_node] + list(hop_neighbors)
        # Generate adjacency matrix and features of the subgraph.
        sub_adj = self.ori_adj[subgraph_nodes, :][:, subgraph_nodes]
        sub_feat = self.ori_feat[subgraph_nodes]

        # Perturbation towards subgraph features.
        noise = torch.randn(sub_feat.size(), device=self.device) * self.feat_perturb
        sub_feat += noise

        return sub_adj, sub_feat

    def _find_n_hop_neighbors(self, node, n_hop, adj):
        """
        Using BFS algorithm to find the n_hop-hop nodes of target node in a given adjacency matrix.
        :param node: Target node.
        :param n_hop: The n_hop-hop structure to be searched.
        :param adj: A given adjacency matrix.
        :return: The n_hop-hop nodes of target node in a given adjacency matrix.
        """
        # Use BFS algorithm to find n_hop neighborhood.
        neighbors = set()
        current_layer = {node}
        for _ in range(n_hop):
            next_layer = set()
            for node in current_layer:
                # Get non-zero entries' index.
                neighbors_indices = adj[node].nonzero().squeeze()
                # Transform to list.
                if neighbors_indices.dim() == 0:
                    neighbors_indices = [neighbors_indices.item()]
                else:
                    neighbors_indices = neighbors_indices.tolist()
                next_layer.update(set(neighbors_indices))
            neighbors.update(next_layer)
            current_layer = next_layer

        return neighbors - {node}

    def _target_node_mask(self):
        """
        Generate mask tensor for training.
        :return: Data mask for target victim node.
        """
        target_node_mask = torch.zeros(self.nnodes + self.sub_feat.shape[0], dtype=torch.bool, device=self.device)
        target_node_mask[self.victim_node] = True

        return target_node_mask

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


if __name__ == "__main__":
    import torch_geometric.transforms as T
    from gcn import GCN
    from gae import GAE, GCNEncoder
    from torch_geometric.utils import negative_sampling
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
    # Model initialization.
    model = GraphCopy(nnodes=x.shape[0],
                      ori_edge_index=ori_edge_index,
                      ori_feat=x,
                      victim_node=0,
                      n_hop=2,
                      feat_perturb=0.1,
                      device='cuda')
    # Get subgraph.
    sub_adj, sub_feat = model.sub_adj, model.sub_feat
    # Generate data mask on target node.
    target_node_mask = torch.zeros(model.nnodes + sub_feat.shape[0], dtype=torch.bool, device='cuda')
    target_node_mask[model.victim_node] = True
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
    train_link_adj = utils.edge_index_2_adj(x.shape[0] + sub_feat.shape[0], train_link_edge_index)
    train_link_adj_norm = utils.normalize_adj_tensor(train_link_adj)
    train_link_neg_edge_label_index = negative_sampling(edge_index=train_link_pos_edge_label_index,
                                                        num_nodes=train_link_pos_edge_label_index.max().item() + 1,
                                                        num_neg_samples=train_link_pos_edge_label_index.size(1))
    test_link_adj = utils.edge_index_2_adj(x.shape[0] + sub_feat.shape[0], test_link_edge_index)
    test_link_adj_norm = utils.normalize_adj_tensor(test_link_adj)
    # Label and node mask on merged graph.
    label = torch.cat([data_cls.y, torch.zeros(sub_feat.shape[0], dtype=data_cls.y.dtype, device='cuda')])
    train_cls_mask = torch.cat([train_cls_mask, torch.zeros(sub_feat.shape[0], dtype=torch.bool, device='cuda')])
    val_cls_mask = torch.cat([val_cls_mask, torch.zeros(sub_feat.shape[0], dtype=torch.bool, device='cuda')])
    test_cls_mask = torch.cat([test_cls_mask, torch.zeros(sub_feat.shape[0], dtype=torch.bool, device='cuda')])
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
                            epochs=200,
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
                           epochs=200,
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
    pred = output[target_node_mask].max(1)[1]
    if pred.item() != label[target_node_mask].item() and any(link_users > ori_adj.shape[0]):
        print('Attack Succeeds.')
    else:
        print('Attack Fails.')
    print(pred.item())
    print(label[target_node_mask].item())

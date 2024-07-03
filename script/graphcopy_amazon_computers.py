import torch
import random
import logging
from module.gae import GAE, GCNEncoder
from module.vgae import VGAE, VariationalGCNEncoder
from module.gcn import GCN
from module.sgc import SGC
from module.graphsage import GraphSAGE
from module.graphcopy import GraphCopy
from tools import utils
from torch_geometric.datasets import Amazon
import torch_geometric.transforms as T
from sklearn.decomposition import PCA
from torch_geometric.utils import negative_sampling
from sklearn.model_selection import train_test_split

# Training parameter settings.
dataset_dir = r'E:\Machine_Learning\Dataset\Processing'
log_file = r'E:\Machine_Learning\Project\LiSA\run\graphcopy_amazon_computers.log'
logging.basicConfig(filename=log_file, filemode='a', level=logging.INFO, format='%(asctime)s - %(message)s')

train_ratio = 0.05
val_ratio = 0.20
test_ratio = 0.35

test_link_predictor = 'GAE' # GAE or VGAE
test_node_classifier = 'GCN' # GCN, SGC, or GraphSAGE

num_victims = 1000
n_hop = 2
num_recommend_users = 3
feat_perturb = 0.1
device = 'cuda'

if __name__ == "__main__":
    # Random seed.
    utils.setup_seed(6666)
    # Graph transform setup.
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False)
    ])
    # Data for node classification.
    dataset_cls = Amazon(dataset_dir, 'Computers', transform=T.NormalizeFeatures())
    data_cls = dataset_cls[0].to(device)
    # Split dataset into training set, validation set, and test set.
    train_ratio = train_ratio
    val_ratio = val_ratio
    test_ratio = test_ratio
    indices = list(range(data_cls.num_nodes))
    train_indices, remaining_indices = train_test_split(indices, train_size=train_ratio, random_state=6666)
    val_indices, remaining_indices = train_test_split(remaining_indices, train_size=val_ratio / (1 - train_ratio), random_state=6666)
    test_indices, _ = train_test_split(remaining_indices, train_size=test_ratio / (1 - train_ratio - val_ratio), random_state=6666)
    # Create mask.
    train_mask = torch.zeros(data_cls.num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(data_cls.num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(data_cls.num_nodes, dtype=torch.bool, device=device)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    # Append masks to dataset.
    data_cls.train_mask = train_mask
    data_cls.val_mask = val_mask
    data_cls.test_mask = test_mask
    # PCA dimensionality reduction on features.
    x_np = data_cls.x.cpu().numpy()
    pca = PCA(n_components=256)
    x_np = pca.fit_transform(x_np)
    x = torch.tensor(x_np).to(device)
    # Edge index and adjacency matrix of original graph.
    ori_edge_index = data_cls.edge_index.to(device)
    ori_adj = utils.edge_index_2_adj(x.shape[0], ori_edge_index)
    # Data for link prediction.
    dataset_link = Amazon(dataset_dir, 'Computers', transform=transform)
    data_link = dataset_link[0]
    # Edge indices for training link prediction model.
    train_link_edge_index = data_link[0].edge_index.to(device)
    train_link_pos_edge_label_index = data_link[0].pos_edge_label_index.to(device)
    test_link_edge_index = data_link[2].edge_index.to(device)
    test_link_pos_edge_label_index = data_link[2].pos_edge_label_index.to(device)
    test_link_neg_edge_label_index = data_link[2].neg_edge_label_index.to(device)

    # Random select victim nodes under test.
    victim_nodes = random.sample(range(x.shape[0]), num_victims)

    # Test iterations.
    miscls_count = 0
    link_count = 0
    for victim_node in victim_nodes:
        # Node mask.
        train_cls_mask = data_cls.train_mask
        val_cls_mask = data_cls.val_mask
        test_cls_mask = data_cls.test_mask
        model = GraphCopy(nnodes=x.shape[0],
                          ori_edge_index=ori_edge_index,
                          ori_feat=x,
                          victim_node=victim_node,
                          n_hop=n_hop,
                          feat_perturb=feat_perturb,
                          device=device)
        # Get subgraph.
        sub_adj, sub_feat = model.sub_adj, model.sub_feat
        # Link recommender model.
        if test_link_predictor == 'GAE':
            recommender = GAE(GCNEncoder(x.shape[1], 16),
                              lr=0.01,
                              device=device).to(device)
        elif test_link_predictor == 'VGAE':
            recommender = VGAE(VariationalGCNEncoder(x.shape[1], 16),
                               lr=0.01,
                               device=device).to(device)
        else:
            print('Please provide a link prediction model for test.')
        # Classifier model.
        if test_node_classifier == 'GCN':
            classifier = GCN(nfeat=x.shape[1],
                             nhid=32,
                             nclass=data_cls.y.max().item() + 1,
                             dropout=0.5,
                             lr=0.01,
                             weight_decay=5e-5).to(device)
        elif test_node_classifier == 'SGC':
            classifier = SGC(nfeat=x.shape[1],
                             nclass=data_cls.y.max().item() + 1,
                             K=2,
                             lr=0.01,
                             weight_decay=5e-5).to(device)
        elif test_node_classifier == 'GraphSAGE':
            classifier = GraphSAGE(nfeat=x.shape[1],
                                   nhid=32,
                                   nclass=data_cls.y.max().item() + 1,
                                   lr=0.01,
                                   weight_decay=5e-5).to(device)
        else:
            print('Please provide a node classification model for test.')
        # Link recommender training data.
        train_link_adj = utils.edge_index_2_adj(x.shape[0] + sub_feat.shape[0], train_link_edge_index)
        train_link_adj_norm = utils.normalize_adj_tensor(train_link_adj)
        train_link_neg_edge_label_index = negative_sampling(edge_index=train_link_pos_edge_label_index,
                                                            num_nodes=train_link_pos_edge_label_index.max().item() + 1,
                                                            num_neg_samples=train_link_pos_edge_label_index.size(1))
        test_link_adj = utils.edge_index_2_adj(x.shape[0] + sub_feat.shape[0], test_link_edge_index)
        test_link_adj_norm = utils.normalize_adj_tensor(test_link_adj)
        # Label and node mask on merged graph.
        label = torch.cat([data_cls.y, torch.zeros(sub_feat.shape[0], dtype=data_cls.y.dtype, device=device)])
        train_cls_mask = torch.cat([train_cls_mask, torch.zeros(sub_feat.shape[0], dtype=torch.bool, device=device)])
        val_cls_mask = torch.cat([val_cls_mask, torch.zeros(sub_feat.shape[0], dtype=torch.bool, device=device)])
        test_cls_mask = torch.cat([test_cls_mask, torch.zeros(sub_feat.shape[0], dtype=torch.bool, device=device)])
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
                                epochs=300,
                                verbose=False)
        z = recommender.encode(combined_feat, combined_adj_norm)
        recommended_adj = model.recommend_links(combined_adj, z, num_recommend_users)
        if test_node_classifier == 'GraphSAGE':
            # Train classifier and evaluate performance on victim node. GraphSAGE use edge_index.
            recommended_edge_index = utils.adj_2_edge_index(recommended_adj)
            classifier.train_model(x=combined_feat,
                                   edge_index=recommended_edge_index,
                                   label=label,
                                   train_mask=train_cls_mask,
                                   val_mask=val_cls_mask,
                                   test_mask=test_cls_mask,
                                   epochs=300,
                                   verbose=False)
            output = classifier(combined_feat, recommended_edge_index)
        else:
            recommended_adj_norm = utils.normalize_adj_tensor(recommended_adj)
            # Train classifier and evaluate performance on victim node. Others use adj.
            classifier.train_model(x=combined_feat,
                                   adj=recommended_adj_norm,
                                   label=label,
                                   train_mask=train_cls_mask,
                                   val_mask=val_cls_mask,
                                   test_mask=test_cls_mask,
                                   epochs=300,
                                   verbose=False)
            output = classifier(combined_feat, recommended_adj_norm)
        pred = output[model.target_node_mask].max(1)[1]
        link_users = utils.detect_new_connections(combined_adj, recommended_adj, model.victim_node)
        logging.info("Victim node is {}.".format(model.victim_node))
        logging.info(link_users)

        if pred.item() != label[model.target_node_mask].item():
            miscls_count += 1
            logging.info("Classification Fails.")
        else:
            logging.info("Classification Succeeds.")

        if any(link_users >= x.shape[0]):
            link_count += 1
            logging.info("Link Generation Succeeds.")
        else:
            logging.info("Link Generation Fails.")

        # Evaluate Attack Success Rate.
    miscls_rate = miscls_count / num_victims
    link_gen_rate = link_count / num_victims
    logging.info(f"Misclassification Rate: {miscls_rate:.4f}")
    logging.info(f"Link Generation Rate: {link_gen_rate:.4f}")

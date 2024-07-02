import torch
import random
import numpy as np


def setup_seed(seed=6666):
    """
    Set random seed.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)     # CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def normalize_adj_tensor(adj):
    """
    Normalize adjacency tensor matrix.
    """
    device = adj.device

    mx = adj + torch.eye(adj.shape[0]).to(device)
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv

    return mx


def edge_index_2_adj(num_nodes, edge_index):
    """
    Convert an edge_index to adjacency matrix.
    """
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = 1.0

    return adj


def adj_2_edge_index(adj):
    """
    Convert an adjacency matrix to edge_index.
    """
    row, col = adj.nonzero(as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)

    return edge_index


def merge_adj(ori_adj, sub_adj):
    """
    Merge two graphs with adjacency matrix.
    """
    # Combine adjacency matrices
    combined_adj = torch.cat([torch.cat([ori_adj, torch.zeros((ori_adj.shape[0], sub_adj.shape[1]), device=ori_adj.device)], dim=1),
                              torch.cat([torch.zeros((sub_adj.shape[0], ori_adj.shape[1]), device=ori_adj.device), sub_adj], dim=1)],
                             dim=0)

    return combined_adj


def merge_edge_index(ori_edge_index, sub_edge_index):
    """
    Merge two graphs with edge_index.
    """
    combined_edge_index = torch.cat([ori_edge_index, sub_edge_index], dim=1)

    return combined_edge_index


def merge_feat(ori_feat, sub_feat):
    """
    Merge two graphs with node features.
    """
    combined_feat = torch.cat([ori_feat, sub_feat], dim=0)

    return combined_feat


def grad_tensor(tensor):
    """
    Copy a tensor and activate the gradient.
    """
    tensor_clone = torch.zeros_like(tensor, requires_grad=True)
    tensor_clone.data = tensor.data.clone()

    return tensor_clone


def detect_new_connections(combined_adj, recommended_adj, victim_node):
    """
    Compare two graphs in structure and extract new links on target node by recommendation system.
    """
    # Find new edges: '1' in recommended_adj but '0' in combined_adj.
    new_edges_mask = (recommended_adj == 1) & (combined_adj == 0)
    # Get new edge indices.
    new_edge_indices = new_edges_mask.nonzero(as_tuple=False)
    # Extract all related node number.
    nodes = torch.unique(new_edge_indices)
    # Remove target node number.
    nodes = nodes[nodes != victim_node]

    return nodes


def accuracy(output, labels):
    """
    Return accuracy of output compared to labels.
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def cosine_similarity(tensor1, tensor2):
    """
    Calculate the Cosine similarity between two tensor
    """
    # Calculate L2 norm
    norm1 = torch.norm(tensor1, p=2)
    norm2 = torch.norm(tensor2, p=2)
    # Calculate dot product
    dot_product = torch.dot(tensor1, tensor2)
    # Calculate Cosine similarity
    cosine_sim = dot_product / (norm1 * norm2)
    return cosine_sim
    

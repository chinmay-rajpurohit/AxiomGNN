import torch
import torch.nn.functional as Fnn
import numpy as np
from deeprobust.graph.data import Dataset
from scipy.sparse import load_npz


def load_cora(device=None, custom_adj_path: str = None):
    if device is None:
        device = torch.device("cpu")

    print("Loading cora dataset...")
    data = Dataset(root="data", name="cora")

    features = data.features
    if not isinstance(features, np.ndarray):
        features = features.toarray()
    X = torch.tensor(features, dtype=torch.float32, device=device)

    
    X = Fnn.normalize(X, p=2, dim=1)

    base_adj = data.adj

    if custom_adj_path is not None:
        print(f"Loading custom perturbed adjacency from: {custom_adj_path}")
        adj_sp = load_npz(custom_adj_path)
        A_bar = torch.tensor(adj_sp.toarray(), dtype=torch.float32, device=device)
    else:
        print("Using original Cora adjacency as A_bar")
        A_bar = torch.tensor(base_adj.toarray(), dtype=torch.float32, device=device)

    
    A_bar = 0.5 * (A_bar + A_bar.T)


    A_bar = torch.clamp(A_bar, min=0.0)

    deg = A_bar.sum(dim=1)
    D = torch.diag(deg)
    L_tilde = D - A_bar

    return X, A_bar, L_tilde

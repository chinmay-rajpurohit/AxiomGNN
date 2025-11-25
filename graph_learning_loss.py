import torch
from operators import L_from_w, A_from_w


def sigma_with_grad(S: torch.Tensor):
    S_sigma = torch.sigmoid(S)
    sigma_prime = S_sigma * (1.0 - S_sigma)
    return S_sigma, sigma_prime


def graph_learning_loss(
    F: torch.Tensor,
    w: torch.Tensor,
    X: torch.Tensor,
    L_tilde: torch.Tensor,
    edge_index: torch.Tensor,
    reg_lambda: float = 1e-2,
) -> torch.Tensor:
    n, d = F.shape

    L_w = L_from_w(w, num_nodes=n, edge_index=edge_index)

    term1 = torch.norm(F - X, p="fro") ** 2

    term2 = torch.trace(F.T @ L_w @ F)

    S = F @ F.T
    S_sigma, _ = sigma_with_grad(S)

    A_w = A_from_w(w, num_nodes=n, edge_index=edge_index)

    term3 = torch.norm(S_sigma - A_w, p="fro") ** 2

    term4 = torch.norm(L_tilde - L_w, p="fro") ** 2

    term5 = torch.trace(X.T @ L_w @ X)

    term6 = reg_lambda * torch.norm(w, p=2) ** 2

    loss = term1 + term2 + term3 + term4 + term5 + term6
    return loss

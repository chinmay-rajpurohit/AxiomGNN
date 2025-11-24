import torch
from operators import L_from_w, L_adj
from graph_learning_loss import sigma_with_grad


def manual_gradients(
    F: torch.Tensor,
    w: torch.Tensor,
    X: torch.Tensor,
    L_tilde: torch.Tensor,
    edge_index: torch.Tensor,
    reg_lambda: float = 1e-2,
):
    n, d = F.shape
    L_w = L_from_w(w, num_nodes=n, edge_index=edge_index)

    diff_FX = F - X
    term1 = torch.norm(diff_FX, p="fro") ** 2
    term2 = torch.trace(F.T @ L_w @ F)

    S = F @ F.T
    S_sigma, sigma_prime = sigma_with_grad(S)

    D_w = torch.diag(torch.diag(L_w))
    A_w = D_w - L_w
    E = S_sigma - A_w
    term3 = torch.norm(E, p="fro") ** 2

    diff_LL = L_tilde - L_w
    term4 = torch.norm(diff_LL, p="fro") ** 2
    term5 = torch.trace(X.T @ L_w @ X)
    term6 = reg_lambda * torch.norm(L_w, p="fro") ** 2

    loss = term1 + term2 + term3 + term4 + term5 + term6

    grad_F_term1 = 2.0 * diff_FX
    grad_F_term2 = 2.0 * (L_w @ F)

    dL_dE = 2.0 * E
    G = dL_dE * sigma_prime
    grad_F_term3 = (G + G.T) @ F

    grad_F = grad_F_term1 + grad_F_term2 + grad_F_term3

    dL_term2 = F @ F.T
    dL_dA = -2.0 * E

    dL_term3 = torch.zeros_like(L_w)
    n = L_w.size(0)
    off_mask = ~torch.eye(n, dtype=torch.bool, device=L_w.device)
    dL_term3[off_mask] = -dL_dA[off_mask]

    dL_term4 = 2.0 * (L_w - L_tilde)
    dL_term5 = X @ X.T
    dL_term6 = 2.0 * reg_lambda * L_w

    dL_total = dL_term2 + dL_term3 + dL_term4 + dL_term5 + dL_term6
    grad_w = L_adj(dL_total, edge_index)

    return loss, grad_F, grad_w

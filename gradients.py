import torch
from graph_learning_loss import graph_learning_loss, sigma_with_grad  


def manual_gradients(
    F: torch.Tensor,
    w: torch.Tensor,
    X: torch.Tensor,
    L_tilde: torch.Tensor,
    edge_index: torch.Tensor,
    reg_lambda: float = 1e-2,
):
    F_req = F.detach().clone().requires_grad_(True)
    w_req = w.detach().clone().requires_grad_(True)

    loss = graph_learning_loss(
        F=F_req,
        w=w_req,
        X=X,
        L_tilde=L_tilde,
        edge_index=edge_index,
        reg_lambda=reg_lambda,
    )

    loss.backward()

    grad_F = F_req.grad.detach()
    grad_w = w_req.grad.detach()

    return loss.detach(), grad_F, grad_w

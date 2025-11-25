import warnings
warnings.filterwarnings("ignore", message="Please install pytorch geometric")

import torch
import torch.nn.functional as Fnn

from operators import build_edge_index_from_adj, L_from_w
from data_utils import load_cora
from gradients import manual_gradients


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    custom_adj_path = "data/cora_meta_adj_0.05.npz"

    X, A_bar, L_tilde = load_cora(device=device, custom_adj_path=custom_adj_path)
    n, d = X.shape
    print(f"Cora loaded: n={n}, d={d}")

    edge_index = build_edge_index_from_adj(A_bar)
    m = edge_index.shape[1]
    print(f"Number of edges (m) represented in w: {m}")

    F = X.clone()

    i, j = edge_index
    with torch.no_grad():
        init_w = A_bar[i, j].clone()
    w_raw = init_w.clone()

    lr_F = 1e-4
    lr_w = 5e-4
    reg_lambda = 1e-2
    num_epochs = 130
    inner_F_steps = 5
    inner_w_steps = 5

    for epoch in range(1, num_epochs + 1):
        for _ in range(inner_F_steps):
            w_pos = Fnn.softplus(w_raw)
            loss_F, grad_F, _ = manual_gradients(
                F=F,
                w=w_pos,
                X=X,
                L_tilde=L_tilde,
                edge_index=edge_index,
                reg_lambda=reg_lambda,
            )
            grad_F = torch.clamp(grad_F, -1e3, 1e3)
            F = F - lr_F * grad_F

        for _ in range(inner_w_steps):
            w_pos = Fnn.softplus(w_raw)
            loss_w, _, grad_w = manual_gradients(
                F=F,
                w=w_pos,
                X=X,
                L_tilde=L_tilde,
                edge_index=edge_index,
                reg_lambda=reg_lambda,
            )
            sigma_wraw = torch.sigmoid(w_raw)
            grad_w_raw = grad_w * sigma_wraw
            grad_w_raw = torch.clamp(grad_w_raw, -1e3, 1e3)
            w_raw = w_raw - lr_w * grad_w_raw
            w_raw = torch.clamp(w_raw, min=0.0)

        print(
            f"Epoch {epoch:03d} | Loss(F-step): {loss_F.item():.6f} | "
            f"Loss(w-step): {loss_w.item():.6f}"
        )

    with torch.no_grad():
        w_pos = Fnn.softplus(w_raw)
        L_star = L_from_w(w_pos, num_nodes=n, edge_index=edge_index)
        D_star = torch.diag(torch.diag(L_star))
        A_star = D_star - L_star

    torch.save(
        {
            "F": F.detach().cpu(),
            "w_raw": w_raw.detach().cpu(),
            "w": w_pos.detach().cpu(),
            "L_star": L_star.detach().cpu(),
            "A_star": A_star.detach().cpu(),
        },
        "learned_graph_cora.pt",
    )

    print("Training finished. Saved learned graph to learned_graph_cora.pt")


if __name__ == "__main__":
    main()

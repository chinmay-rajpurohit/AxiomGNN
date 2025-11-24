import warnings
warnings.filterwarnings("ignore", message="Please install pytorch geometric")

import torch
import torch.nn.functional as Fnn
from operators import build_edge_index_from_adj, L_from_w
from graph_learning_loss import graph_learning_loss
from data_utils import load_cora


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

    F = torch.nn.Parameter(X.clone())

    i, j = edge_index
    with torch.no_grad():
        init_w = A_bar[i, j].clone()
    w_raw = torch.nn.Parameter(init_w)

    params = [F, w_raw]
    optimizer = torch.optim.Adam(params, lr=1e-3)

    num_epochs = 1000
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        w_pos = Fnn.softplus(w_raw)

        loss = graph_learning_loss(
            F=F,
            w=w_pos,
            X=X,
            L_tilde=L_tilde,
            edge_index=edge_index,
            reg_lambda=1e-2,
        )

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")

    with torch.no_grad():
        w_pos = Fnn.softplus(w_raw)
        L_star = L_from_w(w_pos, num_nodes=n, edge_index=edge_index)
        D_star = torch.diag(torch.diag(L_star))
        A_star = D_star - L_star

    torch.save({
        'F': F.detach().cpu(),
        'w': w_pos.detach().cpu(),
        'L_star': L_star.detach().cpu(),
        'A_star': A_star.detach().cpu(),
    }, "learned_graph_cora.pt")

    print("Training finished. Saved learned graph to learned_graph_cora.pt")


if __name__ == "__main__":
    main()

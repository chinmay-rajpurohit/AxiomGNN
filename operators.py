import torch


def build_complete_edge_index(num_nodes: int, device=None) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    j, i = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
    edge_index = torch.stack([i, j], dim=0)
    return edge_index


def build_edge_index_from_adj(A: torch.Tensor) -> torch.Tensor:
    device = A.device
    ei = (A > 0).nonzero(as_tuple=False).T
    mask = ei[0] > ei[1]
    return ei[:, mask].to(device)


def L_from_w(
    w: torch.Tensor,
    num_nodes: int,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    device = w.device
    dtype = w.dtype

    i, j = edge_index
    assert w.shape[0] == i.shape[0], "w and edge_index size mismatch"

    L = torch.zeros((num_nodes, num_nodes), device=device, dtype=dtype)

    L[i, j] -= w
    L[j, i] -= w

    L[i, i] += w
    L[j, j] += w

    return L


def L_adj(
    Q: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    i, j = edge_index
    return Q[i, i] - Q[i, j] - Q[j, i] + Q[j, j]


def A_from_w(
    w: torch.Tensor,
    num_nodes: int,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    device = w.device
    dtype = w.dtype

    A = torch.zeros((num_nodes, num_nodes), device=device, dtype=dtype)
    i, j = edge_index

    A[i, j] = w
    A[j, i] = w

    return A

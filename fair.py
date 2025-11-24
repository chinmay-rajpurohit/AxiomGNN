import torch
import torch.nn.functional as F


def _cost_matrix(num_classes: int, p: int = 2, device=None):
    idx = torch.arange(num_classes, device=device).float()
    x = idx.view(-1, 1)
    y = idx.view(1, -1)
    return torch.abs(x - y) ** p


def sinkhorn_distance(p, q, eps=0.1, n_iters=50):
    p = p.view(-1)
    q = q.view(-1)
    assert p.numel() == q.numel()
    device = p.device
    K_dim = p.numel()
    C = _cost_matrix(K_dim, p=2, device=device)
    K = torch.exp(-C / eps)

    u = torch.ones_like(p) / K_dim
    v = torch.ones_like(q) / K_dim
    eps_denom = 1e-8

    for _ in range(n_iters):
        Kv = K @ v + eps_denom
        u = p / Kv
        KTu = K.t() @ u + eps_denom
        v = q / KTu

    T = torch.diag(u) @ K @ torch.diag(v)
    dist = torch.sum(T * C)
    return dist


def sinkhorn_divergence(p, q, eps=0.1, n_iters=50):
    d_pq = sinkhorn_distance(p, q, eps=eps, n_iters=n_iters)
    d_pp = sinkhorn_distance(p, p, eps=eps, n_iters=n_iters)
    d_qq = sinkhorn_distance(q, q, eps=eps, n_iters=n_iters)
    return d_pq - 0.5 * d_pp - 0.5 * d_qq


def conditional_class_distributions(logits, groups, num_groups=None):
    if logits.dim() != 2:
        raise ValueError("logits must be [n, num_classes]")
    n, num_classes = logits.shape
    if num_groups is None:
        num_groups = int(groups.max().item()) + 1
    probs = F.softmax(logits, dim=-1)
    dists = []
    for g in range(num_groups):
        mask = (groups == g)
        if mask.sum() == 0:
            continue
        p_g = probs[mask].mean(dim=0)
        p_g = p_g / (p_g.sum() + 1e-8)
        dists.append(p_g)
    return dists


def sinkhorn_fairness_loss(logits, groups, eps=0.1, n_iters=50):
    dists = conditional_class_distributions(logits, groups)
    if len(dists) <= 1:
        return torch.tensor(0.0, device=logits.device)
    total = 0.0
    count = 0
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            total = total + sinkhorn_divergence(dists[i], dists[j], eps=eps, n_iters=n_iters)
            count += 1
    return total / max(count, 1)

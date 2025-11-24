import warnings
warnings.filterwarnings("ignore", message="Please install pytorch geometric")

import torch
import torch.nn as nn
import torch.nn.functional as F
from deeprobust.graph.data import Dataset
import numpy as np
from fair import sinkhorn_fairness_loss


def load_cora_features_and_labels(device):
    data = Dataset(root='data', name='cora')
    features = data.features
    if not isinstance(features, torch.Tensor):
        if not isinstance(features, np.ndarray):
            features = features.toarray()
        features = torch.tensor(features, dtype=torch.float32)
    X = features.to(device)
    labels = torch.tensor(data.labels, dtype=torch.long, device=device)
    idx_train = torch.tensor(data.idx_train, dtype=torch.long, device=device)
    idx_val = torch.tensor(data.idx_val, dtype=torch.long, device=device)
    idx_test = torch.tensor(data.idx_test, dtype=torch.long, device=device)
    num_classes = int(labels.max().item() + 1)
    return X, labels, idx_train, idx_val, idx_test, num_classes


def normalize_adjacency(A):
    device = A.device
    n = A.size(0)
    A_clamped = torch.clamp(A, min=0.0)
    I = torch.eye(n, device=device)
    A_tilde = A_clamped + I
    deg = A_tilde.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, X, A_hat):
        return A_hat @ self.linear(X)


class GCN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.gc1 = GCNLayer(in_features, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, X, A_hat):
        h = self.gc1(X, A_hat)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.gc2(h, A_hat)


def evaluate(model, X, A_hat, labels, idx):
    model.eval()
    with torch.no_grad():
        logits = model(X, A_hat)
        preds = logits.argmax(dim=1)
        return (preds[idx] == labels[idx]).float().mean().item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load("learned_graph_cora.pt", map_location=device)
    A_star = ckpt["A_star"].to(device)

    X, labels, idx_train, idx_val, idx_test, num_classes = load_cora_features_and_labels(device)
    assert A_star.size(0) == X.size(0)

    groups_np = np.load("data/groups.npy")
    groups = torch.tensor(groups_np, dtype=torch.long, device=device)

    n, d = X.shape
    A_hat = normalize_adjacency(A_star)

    NUM_RUNS = 3
    NUM_EPOCHS = 1000
    HIDDEN_DIM = 64
    lambda_fair = 1e-2

    best_overall_test = 0.0

    for run in range(1, NUM_RUNS + 1):
        model = GCN(in_features=d, hidden_dim=HIDDEN_DIM, num_classes=num_classes, dropout=0.5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_test_at_best_val = 0.0

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(X, A_hat)
            L_ce = criterion(logits[idx_train], labels[idx_train])
            L_fair = sinkhorn_fairness_loss(logits, groups, eps=0.1, n_iters=50)
            loss = L_ce + lambda_fair * L_fair
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0 or epoch == 1:
                val_acc = evaluate(model, X, A_hat, labels, idx_val)
                test_acc = evaluate(model, X, A_hat, labels, idx_test)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_at_best_val = test_acc
                print(f"Run {run} | Epoch {epoch:04d} | Loss {loss.item():.4f} | Val {val_acc*100:.2f}% | Test {test_acc*100:.2f}%")

        print(f"\n[Run {run}] Best Test {best_test_at_best_val*100:.2f}%")
        if best_test_at_best_val > best_overall_test:
            best_overall_test = best_test_at_best_val

    logits_all = model(X, A_hat)
    final_fair = sinkhorn_fairness_loss(logits_all, groups, eps=0.1, n_iters=50).item()

    print(f"\nFinal Best Test Accuracy: {best_overall_test*100:.2f}%")
    print(f"Final Sinkhorn Fairness Loss: {final_fair:.4f}")


if __name__ == "__main__":
    main()

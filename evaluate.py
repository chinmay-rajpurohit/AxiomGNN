import warnings
warnings.filterwarnings("ignore", message="Please install pytorch geometric")
import torch
from deeprobust.graph.data import Dataset


def load_cora_labels(device):
    print("Loading Cora labels and splits...")
    data = Dataset(root='data', name='cora')

    labels = torch.tensor(data.labels, dtype=torch.long, device=device)
    idx_train = torch.tensor(data.idx_train, dtype=torch.long, device=device)
    idx_val = torch.tensor(data.idx_val, dtype=torch.long, device=device)
    idx_test = torch.tensor(data.idx_test, dtype=torch.long, device=device)

    num_classes = int(labels.max().item() + 1)
    print(f"Cora: num_nodes={labels.shape[0]}, num_classes={num_classes}")
    print(f"Train size={idx_train.shape[0]}, Val size={idx_val.shape[0]}, Test size={idx_test.shape[0]}")
    return labels, idx_train, idx_val, idx_test, num_classes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    
    ckpt = torch.load("learned_graph_cora.pt", map_location=device)
    F = ckpt["F"].to(device)  
    n, d = F.shape
    print(f"Loaded learned embeddings F with shape: {F.shape}")

    
    labels, idx_train, idx_val, idx_test, num_classes = load_cora_labels(device)

    
    classifier = torch.nn.Linear(d, num_classes).to(device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-2, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def evaluate(split_idx):
        classifier.eval()
        with torch.no_grad():
            logits = classifier(F[split_idx])
            preds = logits.argmax(dim=1)
            acc = (preds == labels[split_idx]).float().mean().item()
        return acc

    
    num_epochs = 200
    for epoch in range(1, num_epochs + 1):
        classifier.train()
        optimizer.zero_grad()

        logits = classifier(F[idx_train])
        loss = criterion(logits, labels[idx_train])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            train_acc = evaluate(idx_train)
            val_acc = evaluate(idx_val)
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"Train Acc: {train_acc*100:.2f}% | "
                f"Val Acc: {val_acc*100:.2f}%"
            )

    
    test_acc = evaluate(idx_test)
    print(f"\nFinal Test Accuracy (linear probe on F): {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()

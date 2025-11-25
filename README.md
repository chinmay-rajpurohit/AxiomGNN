AxiomGNN
A self-supervised graph learning framework that refines graph topology (A*) and learns embeddings (F) without labels.

Usage:

Train AxiomGNN (learn A* + F) python train.py

Evaluate embedding quality (linear probe on F) python evaluate.py

Train GCN using refined graph A* python test.py

Notes:
- Supports perturbed graphs (.npz) placed in data/ folder
- Select file inside train_graph_learning.py using:
  custom_adj_path = "data/your_file.npz"

Requirements:
pip install torch scipy deeprobust numpy

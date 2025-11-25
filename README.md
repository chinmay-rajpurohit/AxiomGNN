AxiomGNN
A self-supervised graph learning framework that refines graph topology (A*) and learns node embeddings (F) without labels. The model reconstructs a cleaner adjacency structure directly from node features and then uses the refined graph A* for downstream GNN training.

Usage:
1) Train AxiomGNN (learn A* + F)
   python train.py

2) Evaluate embedding quality (linear probe on F)
   python evaluate.py

3) Train GCN using refined graph A*
   python test.py

Notes:
- Supports perturbed graphs (.npz) placed in data/ folder
- Select file inside train_graph_learning.py using:
  custom_adj_path = "data/your_file.npz"
- The topology A* is learned from a feature-driven objective instead of using labels.

Requirements:
pip install torch scipy deeprobust numpy

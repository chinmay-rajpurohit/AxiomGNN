import numpy as np
import torch

ckpt = torch.load("learned_graph_cora.pt", map_location="cpu")
n = ckpt["A_star"].shape[0]

groups = np.random.randint(0, 2, size=n)  
np.save("data/groups.npy", groups)

print("groups.npy created with shape:", groups.shape)
print("Saved inside: data/groups.npy")

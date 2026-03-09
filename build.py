# ============================
# ONE-CELL LiDAR DL Pipeline
# ============================

import laspy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -------- Step 1: Load LAS --------
las = laspy.read(r"D:\veera\lidarrrrr\DX3035724 S.GIUSTO000001.laz")   # change path
points = np.vstack((las.x, las.y, las.z)).T

# -------- Step 2: Create simple labels (demo) --------
labels = np.zeros(len(points))
labels[points[:,2] > 20] = 1   # towers
labels[(points[:,2] > 5) & (points[:,2] <= 20)] = 2   # buildings
labels[points[:,2] <= 5] = 3   # wires

X = torch.tensor(points, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)

# -------- Step 3: Model --------
model = nn.Sequential(
    nn.Linear(3, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 4)
)

# -------- Step 4: Train --------
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epoch:", epoch, "Loss:", loss.item())

# -------- Step 5: Predict --------
with torch.no_grad():
    predictions = model(X).argmax(dim=1).numpy()

print("Buildings:", np.sum(predictions==2))
print("Towers:", np.sum(predictions==1))
print("Wires:", np.sum(predictions==3))

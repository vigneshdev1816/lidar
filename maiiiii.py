# =========================================================
# INDUSTRY STYLE LIDAR AI PIPELINE
# Height normalization + geometric features + DL training
# =========================================================

import os
import laspy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset, DataLoader

BASE_DIR = r"D:\veera\lidarrrrr"
input_file  = os.path.join(BASE_DIR, "DX3035724 S.GIUSTO000001.laz")
output_file = os.path.join(BASE_DIR, "industry_classified_output.laz")

# -------------------------
# LOAD
# -------------------------
las = laspy.read(input_file)
points = np.vstack((las.x, las.y, las.z)).T

print("Points loaded:", len(points))

# -------------------------
# STEP 1 — Ground estimation (lowest 5%)
# -------------------------
z_ground = np.percentile(points[:,2], 5)
height_above_ground = points[:,2] - z_ground

# -------------------------
# STEP 2 — Neighborhood geometric features
# -------------------------
tree = KDTree(points[:,:3])
idx = tree.query(points[:,:3], k=10, return_distance=False)

density = np.zeros(len(points))
verticality = np.zeros(len(points))

for i in range(len(points)):
    nbrs = points[idx[i]]
    density[i] = len(nbrs)

    cov = np.cov(nbrs.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    normal = eigvecs[:, np.argmin(eigvals)]
    verticality[i] = abs(normal[2])

# -------------------------
# FEATURE MATRIX
# -------------------------
features = np.column_stack((
    points[:,0],
    points[:,1],
    height_above_ground,
    density,
    verticality
))

# -------------------------
# AUTO LABELS (demo rule based)
# -------------------------
labels = np.zeros(len(points))
labels[height_above_ground > 20] = 2     # towers
labels[(height_above_ground > 5) & (height_above_ground <= 20)] = 1  # buildings
labels[(height_above_ground <= 5) & (verticality < 0.3)] = 3   # wires
labels = labels.astype(int)

num_classes = 4

# -------------------------
# DATASET
# -------------------------
class PCDataset(Dataset):
    def __init__(self,x,y):
        self.x=torch.tensor(x,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.long)
    def __len__(self): return len(self.x)
    def __getitem__(self,i): return self.x[i],self.y[i]

loader = DataLoader(PCDataset(features,labels), batch_size=4096, shuffle=True)

# -------------------------
# MODEL
# -------------------------
model = nn.Sequential(
    nn.Linear(5,64), nn.ReLU(),
    nn.Linear(64,128), nn.ReLU(),
    nn.Linear(128,256), nn.ReLU(),
    nn.Linear(256,num_classes)
)

opt = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# -------------------------
# TRAIN
# -------------------------
for epoch in range(5):
    total=0
    for x,y in loader:
        pred=model(x)
        loss=loss_fn(pred,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total+=loss.item()
    print("Epoch",epoch+1,"Loss",total)

# -------------------------
# -------------------------
# INFERENCE (memory safe)
# -------------------------
model.eval()

features_tensor = torch.tensor(features, dtype=torch.float32)
infer_loader = DataLoader(features_tensor, batch_size=4096, shuffle=False)

pred_list = []

with torch.no_grad():
    for batch in infer_loader:
        logits = model(batch)
        pred = torch.argmax(logits, dim=1)
        pred_list.append(pred.cpu().numpy())

preds = np.concatenate(pred_list)

# -------------------------
# COLORS
# -------------------------        
colors = np.zeros((len(points),3),dtype=np.uint8)
colors[preds==0]=[0,255,0]    # ground
colors[preds==1]=[255,0,0]    # buildings
colors[preds==2]=[255,255,0]  # towers
colors[preds==3]=[0,0,255]    # wires

# -------------------------
# SAVE OUTPUT
# -------------------------
header = las.header.copy()
header.point_format = laspy.PointFormat(3)

out = laspy.LasData(header)
out.x, out.y, out.z = las.x, las.y, las.z
out.classification = preds.astype(np.uint8)

out.red   = (colors[:,0].astype(np.uint16)*256)
out.green = (colors[:,1].astype(np.uint16)*256)
out.blue  = (colors[:,2].astype(np.uint16)*256)

out.write(output_file)

print("Finished:", output_file)

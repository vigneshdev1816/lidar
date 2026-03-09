import laspy
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# -------------------------
# LOAD LAZ FILE
# -------------------------
input_file = r"D:\veera\lidarrrrr\DX3035724 S.GIUSTO000001.laz"   # change to your file
output_file = "building_classified.laz"

print("Loading LAZ...")
las = laspy.read(input_file)

points = np.vstack((las.x, las.y, las.z)).transpose()
print("Total points:", len(points))

# -------------------------
# FEATURE PREPARATION
# -------------------------
scaler = StandardScaler()
features = scaler.fit_transform(points)

X_tensor = torch.tensor(features, dtype=torch.float32)

# Simple rule-based labels (example: height > threshold = building)
z = points[:, 2]
labels = (z > np.percentile(z, 70)).astype(int)
y_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

# -------------------------
# DEFINE MODEL
# -------------------------
class BuildingNet(nn.Module):
    def __init__(self):
        super(BuildingNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = BuildingNet()

# -------------------------
# TRAIN MODEL
# -------------------------
print("Training model...")
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5
batch_size = 50000

for epoch in range(epochs):
    for i in range(0, X_tensor.shape[0], batch_size):
        batch_x = X_tensor[i:i+batch_size]
        batch_y = y_tensor[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} completed")

print("Training completed")

# -------------------------
# MEMORY-SAFE PREDICTION
# -------------------------
print("Predicting in batches...")

pred_list = []
model.eval()

with torch.no_grad():
    for i in range(0, X_tensor.shape[0], batch_size):
        batch = X_tensor[i:i + batch_size]
        out = model(batch)
        pred_list.append(out)

pred = torch.cat(pred_list, dim=0)
pred_binary = (pred > 0.5).int().numpy().flatten()

print("Prediction completed")

# -------------------------
# WRITE OUTPUT LAZ
# -------------------------
classification = np.ones(len(points), dtype=np.uint8)

# Building class = 6 (standard LAS classification)
classification[pred_binary == 1] = 6

las.classification = classification
las.write(output_file)

print("Building classification saved to:", output_file)

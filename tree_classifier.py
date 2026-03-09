import laspy
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# LOAD INPUT FILE
# -------------------------
input_file = r"D:\veera\lidarrrrr\DX3035724 S.GIUSTO000001.laz"
las = laspy.read(input_file)

points = np.vstack((las.x, las.y, las.z)).transpose()

# -------------------------
# STEP 1: Height Normalization
# -------------------------
z = points[:, 2]
z_norm = (z - z.min()) / (z.max() - z.min())

features = np.column_stack((points[:, 0],
                            points[:, 1],
                            z_norm))

# -------------------------
# STEP 2: Create Tree Labels (Height rule)
# -------------------------
labels = (z_norm > 0.35).astype(int)

# -------------------------
# STEP 3: Train Random Forest (FAST TRAINING)
# -------------------------
print("Total points:", len(features))

sample_size = 100000

if len(features) > sample_size:
    idx = np.random.choice(len(features), sample_size, replace=False)
    X_train = features[idx]
    y_train = labels[idx]
else:
    X_train = features
    y_train = labels

clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
clf.fit(X_train, y_train)

print("Training completed")

# Predict on full dataset
pred = clf.predict(features)
tree_mask = pred == 1

# -------------------------
# STEP 4: Write Output LAZ
# -------------------------
out_las = laspy.create(point_format=las.header.point_format,
                       file_version=las.header.version)

out_las.points = las.points
out_las.classification[:] = 1

# Assign tree class (5)
out_las.classification[tree_mask] = 5

out_las.write("tree_classified_output.laz")

print("Tree classification completed successfully.")

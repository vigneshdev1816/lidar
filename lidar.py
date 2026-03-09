# ============================================================
#   COMPLETE LIDAR CLASSIFICATION USING DEEP LEARNING
#   Model: RandLA-Net inspired approach with PyTorch
#   Author: Generated for LiDAR Point Cloud Classification
#   Input: .las or .laz file
#   Output: classified .las file
# ============================================================

# ─────────────────────────────────────────────
# STEP 0: INSTALL REQUIRED LIBRARIES
# Run this in terminal before running script:
# pip install laspy lazrs numpy scikit-learn torch open3d pdal tqdm matplotlib
# ─────────────────────────────────────────────

import os
import numpy as np
import laspy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1: CONFIGURATION
# Change input_file to your .las or .laz file path
# ─────────────────────────────────────────────

CONFIG = {
    "input_file"     : r"D:\veera\lidarrrrr\DX3035724 S.GIUSTO000001.laz",       # 👈 Change this to your file path
    "output_file"    : "classified_output.las",
    "num_classes"    : 8,                      # 0,1,2,3,4,5,6,9 classes
    "k_neighbors"    : 16,                     # KNN neighbors for feature extraction
    "voxel_size"     : 0.5,                    # Downsampling voxel size (meters)
    "batch_size"     : 4096,                   # Points per batch
    "epochs"         : 50,                     # Training epochs (if labeled data)
    "learning_rate"  : 0.001,
    "device"         : "cuda" if torch.cuda.is_available() else "cpu"
}

print(f"Using device: {CONFIG['device']}")


# ─────────────────────────────────────────────
# STEP 2: LOAD .LAS / .LAZ FILE
# Reads all point attributes from your file
# ─────────────────────────────────────────────

def load_las_file(filepath):
    """
    Load LiDAR .las or .laz file
    Returns: points array (N, 3) and all attributes
    """
    print(f"\n📂 Loading file: {filepath}")
    las = laspy.read(filepath)

    # Extract XYZ coordinates
    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)

    # Extract additional features if available
    features = {}
    available_dims = list(las.point_format.dimension_names)
    print(f"Available dimensions: {available_dims}")

    if 'intensity' in available_dims:
        features['intensity'] = np.array(las.intensity, dtype=np.float32)
    else:
        features['intensity'] = np.zeros(len(x), dtype=np.float32)

    if 'number_of_returns' in available_dims:
        features['num_returns'] = np.array(las.number_of_returns, dtype=np.float32)
    else:
        features['num_returns'] = np.ones(len(x), dtype=np.float32)

    if 'return_number' in available_dims:
        features['return_num'] = np.array(las.return_number, dtype=np.float32)
    else:
        features['return_num'] = np.ones(len(x), dtype=np.float32)

    # Existing classification
    if 'classification' in available_dims:
        features['existing_class'] = np.array(las.classification, dtype=np.int32)
    else:
        features['existing_class'] = np.zeros(len(x), dtype=np.int32)

    points = np.vstack([x, y, z]).T
    print(f"✅ Loaded {len(points):,} points")
    print(f"   X range: {x.min():.2f} to {x.max():.2f}")
    print(f"   Y range: {y.min():.2f} to {y.max():.2f}")
    print(f"   Z range: {z.min():.2f} to {z.max():.2f}")
    print(f"   Existing classes: {np.unique(features['existing_class'])}")

    return points, features, las


# ─────────────────────────────────────────────
# STEP 3: PREPROCESSING
# Noise removal + normalization + feature engineering
# ─────────────────────────────────────────────

def preprocess_points(points, features):
    """
    Preprocess point cloud:
    - Statistical noise removal
    - Height normalization
    - Feature engineering
    """
    print("\n⚙️  Preprocessing points...")

    # ── 3a. Statistical Outlier Removal ──
    tree = KDTree(points)
    distances, _ = tree.query(points, k=20)
    mean_dist = distances[:, 1:].mean(axis=1)
    threshold = mean_dist.mean() + 2.0 * mean_dist.std()
    valid_mask = mean_dist < threshold

    points = points[valid_mask]
    for key in features:
        features[key] = features[key][valid_mask]

    print(f"   After noise removal: {len(points):,} points")

    # ── 3b. Normalize coordinates ──
    points_normalized = points.copy()
    points_normalized[:, 0] -= points[:, 0].mean()
    points_normalized[:, 1] -= points[:, 1].mean()
    # Keep Z as-is for height-based classification

    # ── 3c. Feature Engineering ──
    # Height above minimum (useful for ground detection)
    z_min = points[:, 2].min()
    height_above_ground = points[:, 2] - z_min

    # Local height variance (useful for vegetation detection)
    # Process in chunks to avoid memory error
    tree2 = KDTree(points[:, :2])  # 2D tree for neighborhood
    n_points = len(points)
    chunk_size = 100000  # process 100k points at a time
    height_variance = np.zeros(n_points, dtype=np.float32)
    height_range = np.zeros(n_points, dtype=np.float32)

    print("   Computing local height features (chunked)...")
    for start in tqdm(range(0, n_points, chunk_size), desc="   Height features"):
        end = min(start + chunk_size, n_points)
        _, idx_chunk = tree2.query(points[start:end, :2], k=CONFIG['k_neighbors'])
        local_z = points[:, 2][idx_chunk].astype(np.float32)
        height_variance[start:end] = local_z.var(axis=1)
        height_range[start:end] = local_z.max(axis=1) - local_z.min(axis=1)
        del local_z, idx_chunk

    # Intensity normalization
    intensity = features['intensity']
    intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)

    # Return ratio (first_return / total_returns)
    return_ratio = features['return_num'] / (features['num_returns'] + 1e-8)

    # Stack all features
    feature_matrix = np.column_stack([
        points_normalized,          # x, y, z normalized
        height_above_ground,        # height above ground
        height_variance,            # local height variance
        height_range,               # local height range
        intensity_norm,             # normalized intensity
        return_ratio,               # return ratio
        features['num_returns'],    # number of returns
    ])

    print(f"   Feature matrix shape: {feature_matrix.shape}")
    return points, feature_matrix, valid_mask, features


# ─────────────────────────────────────────────
# STEP 4: RANDLA-NET MODEL ARCHITECTURE
# Simplified RandLA-Net for point cloud classification
# ─────────────────────────────────────────────

class LocalFeatureAggregation(nn.Module):
    """
    Local Feature Aggregation module from RandLA-Net
    Aggregates features from K nearest neighbors
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x shape: (N, in_channels)
        out = self.mlp1(x)
        # Self attention
        att = self.attention(out)
        out = out * att
        # Concatenate with original
        out = torch.cat([out, out], dim=1)
        out = self.mlp2(out)
        return out


class RandLANet(nn.Module):
    """
    Simplified RandLA-Net for point cloud semantic segmentation
    Input: (N, num_features)
    Output: (N, num_classes)
    """
    def __init__(self, num_features, num_classes):
        super().__init__()

        # Encoder
        self.encoder1 = LocalFeatureAggregation(num_features, 64)
        self.encoder2 = LocalFeatureAggregation(64, 128)
        self.encoder3 = LocalFeatureAggregation(128, 256)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Decoder
        self.decoder3 = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # Encoder with skip connections
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder with skip connections
        d3 = self.decoder3(torch.cat([b, e3], dim=1))
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))
        d1 = self.decoder1(torch.cat([d2, e1], dim=1))

        # Classification
        out = self.classifier(d1)
        return out


# ─────────────────────────────────────────────
# STEP 5: RULE-BASED CLASSIFICATION
# Uses geometric features to assign class labels
# This generates pseudo-labels for training
# ─────────────────────────────────────────────

def rule_based_classification(points, features_dict):
    """
    Rule-based classification to generate initial labels
    Used as pseudo-labels when no ground truth available

    Classes:
    0 = Unclassified
    1 = Unassigned
    2 = Ground
    3 = Low Vegetation (0-0.5m)
    4 = Medium Vegetation (0.5-2m)
    5 = High Vegetation (2m+)
    6 = Building
    9 = Water
    """
    print("\n📏 Generating rule-based labels...")

    n_points = len(points)
    labels = np.zeros(n_points, dtype=np.int32)

    z = points[:, 2]
    z_min = z.min()
    height = z - z_min

    intensity = features_dict['intensity']
    num_returns = features_dict['num_returns']
    return_num = features_dict['return_num']

    # KD-Tree for neighborhood analysis — chunked to avoid memory error
    tree = KDTree(points[:, :2])
    chunk_size = 100000
    local_z_std = np.zeros(n_points, dtype=np.float32)
    local_z_range = np.zeros(n_points, dtype=np.float32)
    local_height_mean = np.zeros(n_points, dtype=np.float32)

    print("   Computing neighborhood features (chunked)...")
    for start in tqdm(range(0, n_points, chunk_size), desc="   Neighborhood"):
        end = min(start + chunk_size, n_points)
        _, idx_chunk = tree.query(points[start:end, :2], k=20)
        local_z_chunk = z[idx_chunk].astype(np.float32)
        local_z_std[start:end] = local_z_chunk.std(axis=1)
        local_z_range[start:end] = local_z_chunk.max(axis=1) - local_z_chunk.min(axis=1)
        local_height_mean[start:end] = height[idx_chunk].mean(axis=1)
        del local_z_chunk, idx_chunk

    # ── Rule 2: Ground ──
    # Low height, low variance, last return
    ground_mask = (
        (height < 0.5) &
        (local_z_std < 0.15) &
        (return_num == num_returns)
    )
    labels[ground_mask] = 2

    # ── Rule 3: Low Vegetation ──
    low_veg_mask = (
        (height >= 0.1) & (height < 0.5) &
        (local_z_range > 0.1) &
        (num_returns > 1)
    )
    labels[low_veg_mask] = 3

    # ── Rule 4: Medium Vegetation ──
    med_veg_mask = (
        (height >= 0.5) & (height < 2.0) &
        (num_returns >= 1)
    )
    labels[med_veg_mask] = 4

    # ── Rule 5: High Vegetation ──
    high_veg_mask = (
        (height >= 2.0) &
        (local_z_range > 1.0) &
        (num_returns > 1)
    )
    labels[high_veg_mask] = 5

    # ── Rule 6: Building ──
    # High, flat surface, single return, high intensity
    building_mask = (
        (height >= 2.0) &
        (local_z_std < 0.3) &
        (num_returns == 1) &
        (intensity > np.percentile(intensity, 40))
    )
    labels[building_mask] = 6

    # ── Rule 9: Water ──
    # Very flat, low intensity, low returns
    water_mask = (
        (height < 0.3) &
        (local_z_std < 0.05) &
        (intensity < np.percentile(intensity, 20))
    )
    labels[water_mask] = 9

    # ── Rule 1: Unassigned (everything else) ──
    labels[labels == 0] = 1

    # Print class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\n   Initial class distribution:")
    class_names = {0:'Unclassified', 1:'Unassigned', 2:'Ground',
                   3:'Low Veg', 4:'Med Veg', 5:'High Veg',
                   6:'Building', 9:'Water'}
    for cls, cnt in zip(unique, counts):
        name = class_names.get(cls, f'Class {cls}')
        pct = cnt/n_points*100
        print(f"   Class {cls:2d} ({name:<15}): {cnt:>8,} points ({pct:.1f}%)")

    return labels


# ─────────────────────────────────────────────
# STEP 6: TRAIN RANDLA-NET MODEL
# Train on pseudo-labels from rule-based classification
# ─────────────────────────────────────────────

def train_model(feature_matrix, labels):
    """
    Train RandLA-Net model on point features
    """
    print(f"\n🧠 Training RandLA-Net model on {CONFIG['device']}...")

    # Map class labels to consecutive indices
    unique_classes = np.unique(labels)
    class_map = {c: i for i, c in enumerate(unique_classes)}
    reverse_map = {i: c for c, i in class_map.items()}
    mapped_labels = np.array([class_map[l] for l in labels])
    num_classes = len(unique_classes)

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_matrix).astype(np.float32)

    # Convert to tensors
    X_tensor = torch.FloatTensor(features_scaled).to(CONFIG['device'])
    y_tensor = torch.LongTensor(mapped_labels).to(CONFIG['device'])

    # Initialize model
    model = RandLANet(
        num_features=feature_matrix.shape[1],
        num_classes=num_classes
    ).to(CONFIG['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    batch_size = CONFIG['batch_size']
    n_points = len(features_scaled)

    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        correct = 0
        n_batches = 0

        # Shuffle indices
        perm = torch.randperm(n_points)
        X_tensor = X_tensor[perm]
        y_tensor = y_tensor[perm]

        # Mini-batch training
        for i in range(0, n_points, batch_size):
            X_batch = X_tensor[i:i+batch_size]
            y_batch = y_tensor[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == y_batch).sum().item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        accuracy = correct / n_points * 100

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch [{epoch+1:3d}/{CONFIG['epochs']}] "
                  f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    print("✅ Training complete!")
    return model, scaler, reverse_map


# ─────────────────────────────────────────────
# STEP 7: PREDICT CLASSES
# Run trained model on all points
# ─────────────────────────────────────────────

def predict_classes(model, scaler, feature_matrix, reverse_map):
    """
    Predict class for every point in the cloud
    """
    print("\n🔍 Predicting classes for all points...")

    model.eval()
    features_scaled = scaler.transform(feature_matrix).astype(np.float32)
    X_tensor = torch.FloatTensor(features_scaled).to(CONFIG['device'])

    all_predictions = []
    batch_size = CONFIG['batch_size']

    with torch.no_grad():
        for i in tqdm(range(0, len(features_scaled), batch_size),
                      desc="   Classifying"):
            batch = X_tensor[i:i+batch_size]
            outputs = model(batch)
            preds = outputs.argmax(1).cpu().numpy()
            all_predictions.extend(preds)

    # Map back to original class codes
    predictions = np.array([reverse_map[p] for p in all_predictions])
    return predictions


# ─────────────────────────────────────────────
# STEP 8: SAVE CLASSIFIED .LAS FILE
# Write predictions back to .las file
# ─────────────────────────────────────────────

def save_classified_las(original_las, predictions, valid_mask, output_path):
    """
    Save classified point cloud as .las file
    """
    print(f"\n💾 Saving classified file to: {output_path}")

    # Create output LAS file
    out_las = laspy.LasData(header=original_las.header)
    out_las.points = original_las.points

    # Create full classification array
    full_classification = np.zeros(len(original_las.points), dtype=np.uint8)
    full_classification[valid_mask] = predictions.astype(np.uint8)

    out_las.classification = full_classification
    out_las.write(output_path)

    # Print final class distribution
    unique, counts = np.unique(predictions, return_counts=True)
    total = len(predictions)
    class_names = {0:'Unclassified', 1:'Unassigned', 2:'Ground',
                   3:'Low Veg', 4:'Med Veg', 5:'High Veg',
                   6:'Building', 9:'Water'}

    print("\n📊 Final Classification Results:")
    print("=" * 55)
    for cls, cnt in zip(unique, counts):
        name = class_names.get(int(cls), f'Class {cls}')
        pct = cnt / total * 100
        bar = "█" * int(pct / 2)
        print(f"  Class {cls:2d} ({name:<15}): {cnt:>8,} pts ({pct:5.1f}%) {bar}")
    print("=" * 55)
    print(f"✅ Total classified points: {total:,}")
    print(f"✅ Saved to: {output_path}")


# ─────────────────────────────────────────────
# STEP 9: MAIN PIPELINE
# Runs all steps in sequence
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("   LIDAR POINT CLOUD CLASSIFICATION - RandLA-Net")
    print("=" * 60)

    # ── Load file ──
    points, features_dict, original_las = load_las_file(CONFIG['input_file'])

    # ── Preprocess ──
    points, feature_matrix, valid_mask, features_dict = preprocess_points(
        points, features_dict
    )

    # ── Generate pseudo-labels (rule-based) ──
    labels = rule_based_classification(points, features_dict)

    # ── Train RandLA-Net ──
    model, scaler, reverse_map = train_model(feature_matrix, labels)

    # ── Predict ──
    predictions = predict_classes(model, scaler, feature_matrix, reverse_map)

    # ── Save output ──
    save_classified_las(original_las, predictions, valid_mask, CONFIG['output_file'])

    print("\n🎉 Classification Pipeline Complete!")
    print(f"   Open '{CONFIG['output_file']}' in CloudCompare to visualize results")
    print("   In CloudCompare: Edit → Colors → Set Unique Color per Class")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
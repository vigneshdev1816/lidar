import laspy
import numpy as np
import CSF

# -------------------------
# INPUT FILE
# -------------------------
input_file = r"D:\veera\lidarrrrr\DX3035724 S.GIUSTO000001.laz" 
output_file = "tree_classified.laz"

las = laspy.read(input_file)

points = np.vstack((las.x, las.y, las.z)).transpose()

# -------------------------
# STEP 1 — Ground detection using CSF
# -------------------------
csf = CSF.CSF()
csf.setPointCloud(points)

ground_idx = []
non_ground_idx = []

csf.do_filtering(ground_idx, non_ground_idx)

ground_idx = np.array(ground_idx)
non_ground_idx = np.array(non_ground_idx)

# -------------------------
# STEP 2 — Height above ground
# -------------------------
ground_z = np.mean(las.z[ground_idx])
height = las.z - ground_z

# -------------------------
# STEP 3 — Classification
# -------------------------
classification = np.zeros(len(las.points), dtype=np.uint8)

classification[ground_idx] = 2          # Ground
classification[non_ground_idx] = 1      # Default non-ground

# Tree classification (vegetation)
tree_mask = (height > 2) & (height < 50)
classification[tree_mask] = 5           # Vegetation

# -------------------------
# STEP 4 — Save output
# -------------------------
las.classification = classification
las.write(output_file)

print("Tree classification completed and saved:", output_file)

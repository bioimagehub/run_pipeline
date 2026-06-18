#%% Imports
from bioio import BioImage
import matplotlib.pyplot as plt
import numpy as np




#%% Input

video_path = r"E:\Oyvind\BIP-hub-scratch\train_macropinosome_model\input_drift_corrected\Input_crest__KiaWee__250225_RPE-mNG-Phafin2_BSD_10ul_001_1_drift.ome.tif"
mask_path = r"E:\Oyvind\BIP-hub-scratch\train_macropinosome_model\deep_learning_output_2\masks\Input_crest__KiaWee__250225_RPE-mNG-Phafin2_BSD_10ul_001_1_drift.ome_mask.tif"


# %%
img = BioImage(video_path)
print(img.shape, img.dtype)
mask = BioImage(mask_path) 
print(mask.shape, mask.dtype) 

mask_time =  0 #img.dims.T // 2
print(mask_time)

# %%
size = 2000
img_data = img.get_image_data("YX", T=mask_time, C=0, Z=0)[:size, :size]
print(img_data.shape, img_data.dtype)
mask_data = mask.get_image_data("YX", T=mask_time, C=0, Z=0)[:size, :size]
mask_data = (mask_data > 0).astype(np.uint8)





# %% Find centroids
from scipy.ndimage import label, center_of_mass
labeled_mask, num_features = label(mask_data)
centroids = center_of_mass(mask_data, labeled_mask, range(1, num_features + 1))
print(centroids)

# %% Make a rectangular region around each centroid
regions = []
half_size = 20
for centroid in centroids:
    y, x = int(centroid[0]), int(centroid[1])
    y_min = max(y - half_size, 0)
    y_max = min(y + half_size, img_data.shape[0])
    x_min = max(x - half_size, 0)
    x_max = min(x + half_size, img_data.shape[1])
    regions.append((y_min, y_max, x_min, x_max))

#%% Plot the image with centroids and regions
plt.figure(figsize=(8, 8))  
plt.imshow(img_data, cmap='gray')
for centroid in centroids:
    plt.plot(centroid[1], centroid[0], 'ro')  # Centroid as red dot
for region in regions:
    y_min, y_max, x_min, x_max = region
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         edgecolor='blue', facecolor='none', linewidth=1)
    plt.gca().add_patch(rect)  # Add rectangle to the plot
plt.show()  # Display the plot

# %% Use sam to predict the next frame
centroid_selected = centroids[0]
print(centroid_selected)
centroid_selected_img = img_data[int(centroid_selected[0]) - half_size:int(centroid_selected[0]) + half_size,
                                 int(centroid_selected[1]) - half_size:int(centroid_selected[1]) + half_size]
centroid_selected_mask = mask_data[int(centroid_selected[0]) - half_size:int(centroid_selected[0]) + half_size,
                                 int(centroid_selected[1]) - half_size:int(centroid_selected[1]) + half_size]   

# make a panel with t=mask_time and t=mask_time+1
next_time = mask_time + 1
centroid_selected_img_next = img.get_image_data("YX", T=next_time, C=0, Z=0)[
    int(centroid_selected[0]) - half_size:int(centroid_selected[0]) + half_size,
    int(centroid_selected[1]) - half_size:int(centroid_selected[1]) + half_size
]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title(f"Time {mask_time}")
plt.imshow(centroid_selected_img, cmap='gray')
plt.imshow(centroid_selected_mask, cmap='gray', alpha=0.5)  
plt.subplot(1, 2, 2)
plt.title(f"Time {next_time}")
plt.imshow(centroid_selected_img_next, cmap='gray')
plt.show()  # Display the plot

# %% Load SAM2 and predict mask for next frame
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sam2_utils import load_sam2_image_predictor, find_similar_object

predictor = load_sam2_image_predictor()

predicted_mask = find_similar_object(
    source_image=centroid_selected_img,
    source_mask=centroid_selected_mask,
    target=centroid_selected_img_next,
    predictor=predictor,
)

# Show source frame with mask and next frame with predicted mask
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].set_title(f"Time {mask_time} (source mask)")
axes[0].imshow(centroid_selected_img, cmap='gray')
overlay_src = np.zeros((*centroid_selected_img.shape, 4), dtype=np.float32)
overlay_src[centroid_selected_mask > 0] = [0.0, 1.0, 0.0, 0.5]
axes[0].imshow(overlay_src)
axes[0].axis("off")

axes[1].set_title(f"Time {next_time} (SAM2 predicted mask)")
axes[1].imshow(centroid_selected_img_next, cmap='gray')
overlay_pred = np.zeros((*centroid_selected_img_next.shape, 4), dtype=np.float32)
overlay_pred[predicted_mask > 0] = [1.0, 0.4, 0.0, 0.5]
axes[1].imshow(overlay_pred)
axes[1].axis("off")

plt.tight_layout()
plt.show()

# %% Load SAM2 and predict mask for next 5 frames
import sys
import os
import importlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sam2_utils
importlib.reload(sam2_utils)
from sam2_utils import load_sam2_predictor, find_similar_object

predictor = load_sam2_predictor()

y0 = int(centroid_selected[0]) - half_size
y1 = int(centroid_selected[0]) + half_size
x0 = int(centroid_selected[1]) - half_size
x1 = int(centroid_selected[1]) + half_size

n_future = 15
future_frames = np.stack([
    img.get_image_data("YX", T=mask_time + i + 1, C=0, Z=0)[y0:y1, x0:x1]
    for i in range(n_future)
])  # (5, H, W)

predicted_masks = find_similar_object(
    source_image=centroid_selected_img,
    source_mask=centroid_selected_mask,
    target=future_frames,
    predictor=predictor,
)  # (5, H, W) int32

# Plot: source frame + 5 predicted frames
fig, axes = plt.subplots(1, n_future + 1, figsize=(4 * (n_future + 1), 4))

axes[0].set_title(f"T={mask_time}\n(source mask)")
axes[0].imshow(centroid_selected_img, cmap='gray')
overlay_src = np.zeros((*centroid_selected_img.shape, 4), dtype=np.float32)
overlay_src[centroid_selected_mask > 0] = [0.0, 1.0, 0.0, 0.5]
axes[0].imshow(overlay_src)
axes[0].axis("off")

for i in range(n_future):
    ax = axes[i + 1]
    ax.set_title(f"T={mask_time + i + 1}\n(SAM2 predicted)")
    ax.imshow(future_frames[i], cmap='gray')
    overlay = np.zeros((*future_frames[i].shape, 4), dtype=np.float32)
    overlay[predicted_masks[i] > 0] = [1.0, 0.4, 0.0, 0.5]
    ax.imshow(overlay)
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%

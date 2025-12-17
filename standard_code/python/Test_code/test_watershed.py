import tifffile
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# Load the image
path = r"E:\Coen\Sarah\6849908-IMB-Coen-Sarah-Photoconv_global\masks_nuc\SP20250625__L58R__R2__SP20250625_PC_R2_L58R_001_bleach_corrected_filled_Probabilities.tif"
mask = tifffile.imread(path)
print(f"Original mask shape: {mask.shape}")

# Apply watershed to each slice in the stack
watershed_result = np.zeros_like(mask)

for i in range(mask.shape[0]):
    slice_mask = mask[i]
    
    # Compute distance transform
    distance = ndi.distance_transform_edt(slice_mask)
    
    # Find local maxima (markers for watershed)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=slice_mask)
    markers = np.zeros(distance.shape, dtype=bool)
    markers[tuple(coords.T)] = True
    markers = ndi.label(markers)[0]
    
    # Apply watershed
    watershed_result[i] = watershed(-distance, markers, mask=slice_mask)

# Save the result
output_path = Path(path).parent / f"{Path(path).stem}_watershed.tif"

#tifffile.imwrite(output_path, watershed_result.astype(mask.dtype))
#print(f"Watershed result saved to: {output_path}")
print(f"Result shape: {watershed_result.shape}")

# Show comparison of original and watershed result
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Show a slice from the middle of the stack
slice_idx = mask.shape[0] // 2 if mask.ndim == 3 else 0
original_slice = mask[slice_idx] if mask.ndim == 3 else mask
watershed_slice = watershed_result[slice_idx] if watershed_result.ndim == 3 else watershed_result

axes[0].imshow(original_slice, cmap='gray')
axes[0].set_title('Original Mask')
axes[0].axis('off')

axes[1].imshow(watershed_slice, cmap='gray')
axes[1].set_title('After Watershed')
axes[1].axis('off')

plt.tight_layout()
plt.show()

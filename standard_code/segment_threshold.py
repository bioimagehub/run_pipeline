import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from skimage.filters import (
    threshold_otsu, threshold_yen, threshold_li, threshold_triangle,
    threshold_mean, threshold_minimum, threshold_isodata,
    threshold_niblack, threshold_sauvola
)
from skimage.measure import label

from scipy.ndimage import binary_fill_holes, median_filter, distance_transform_edt

from roifile import ImagejRoi
from skimage.measure import find_contours
from skimage.segmentation import watershed




# local imports
import run_pipeline_helper_functions as rp


# Mapping of threshold methods
threshold_methods = {
    "otsu": threshold_otsu,
    "yen": threshold_yen,
    "li": threshold_li,
    "triangle": threshold_triangle,
    "mean": threshold_mean,
    "minimum": threshold_minimum,
    "isodata": threshold_isodata,
    "niblack": lambda img: threshold_niblack(img, window_size=25),
    "sauvola": lambda img: threshold_sauvola(img, window_size=25)
}

class LabelInfo:
    def __init__(self, label_id: int, x_center: float, y_center: float, npixels: int, frame: int, channel: int, z_plane: int):
        self.label_id = label_id
        self.x_center = x_center
        self.y_center = y_center
        self.npixels = npixels
        self.frame = frame
        self.channel = channel
        self.z_plane = z_plane

    def __repr__(self):
        return (f"LabelInfo(ID={self.label_id}, "
                f"x_center={self.x_center:.2f}, "
                f"y_center={self.y_center:.2f}, "
                f"npixels={self.npixels}, "
                f"frame={self.frame}, "
                f"channel={self.channel}, "
                f"z_plane={self.z_plane})")

    def to_dict(self):
        """Convert the LabelInfo instance to a dictionary for easier access."""
        return {
            "ID": self.label_id,
            "x_center": self.x_center,
            "y_center": self.y_center,
            "npixels": self.npixels,
            "frame": self.frame,
            "channel": self.channel,
            "z_plane": self.z_plane
        }

def apply_median_filter(image: np.ndarray, size: int = 10, channels: list[int] = None, frames: slice = None) -> np.ndarray:
    """Apply a median filter to a 5D TCZYX image in the XY plane for specified timepoints and channels, reducing unnecessary dimensions.

    Args:
        image: 5D TCZYX image (T, C, Z, Y, X).
        size: Size of the median filter kernel in the XY plane (size x size).
        channels: List of channels to filter (defaults to all).
        frames: Slice of frames to filter (defaults to all).

    Returns:
        filtered_image: 5D image after applying the median filter with unnecessary channels removed.
    """
    
    # Check the dimensions of the input image
    if len(image.shape) != 5:
        raise ValueError("Input image must be 5D (T, C, Z, Y, X)")

    t, c, z, y, x = image.shape  # Assuming image shape is (T, C, Z, Y, X)

    # Set default frames if None
    if frames is None:
        frames = list(range(t))  # Use all frames
    if channels is None:
        channels = list(range(c))  # Use all channels

    # Create an output image with the appropriate shape
    # Output shape is (T, num_selected_channels, Z, Y, X)

    # Apply median filter for the specified frames and channels
    for frame_idx in frames:
        for selected_channel_idx, channel_idx in enumerate(channels):
            for z_idx in range(z):
                img_c = image[frame_idx, channel_idx, z_idx]  # Select T=frame_idx, C=channel_idx, Z=z_idx

                # Apply median filter in the XY plane
                image[frame_idx, channel_idx, z_idx] = median_filter(img_c, size=size)

    return image

def apply_threshold(image: np.ndarray, method: str = "otsu", channels: list[int] = None, frames: slice = None) -> tuple[np.ndarray, list[LabelInfo]]:
    """Apply thresholding method to a 5D TCZYX image and return a labeled 5D mask and a list of LabelInfo."""
    
    if len(image.shape) != 5:
        raise ValueError("Input image must be 5D (T, C, Z, Y, X)")
    
    t, c, z, y, x = image.shape  # Assuming image is in the shape (T, C, Z, Y, X)
    
    if frames is None:
        frames = list(range(t))  # Use all frames
    if channels is None:
        channels = list(range(c))  # Use all channels

    # Create an output mask with only the selected channels
    # num_selected_channels = len(channels)
    mask_all = np.zeros((t, c, z, y, x), dtype=np.uint8)  # Ensure mask_all is only for selected channels
    label_info_list = []  # List to store LabelInfo instances
    
    threshold_fn = threshold_methods.get(method)
    if threshold_fn is None:
        raise ValueError(f"Unsupported method: {method}")
    
    for frame_idx in frames:
        for selected_channel_idx, channel_idx in enumerate(channels):
            img_c = image[frame_idx, channel_idx]  # Select T=frame_idx, channel=channel_idx

            for z_idx in range(z):
                plane = img_c[z_idx]
                try:
                    thresh = threshold_fn(plane)
                    binary = plane > thresh
                except Exception as e:
                    print(f"Threshold failed on T={frame_idx}, C={channel_idx}, Z={z_idx}: {e}")
                    continue

                labeled = label(binary)

                # Count unique labels and check if max exceeds 255
                unique_labels = np.unique(labeled)
                if len(unique_labels) > 256:
                    raise ValueError(f"Error: More than 256 labels found in T={frame_idx}, C={channel_idx}, Z={z_idx}. Please check the input image.")

                # Store labels in the appropriate position in the mask
                mask_all[frame_idx, selected_channel_idx, z_idx] = labeled.astype(np.uint8)

                # Calculate properties for each labeled region
                for label_num in unique_labels:
                    if label_num > 0:  # Ignore background label
                        # Calculate number of pixels in the label region
                        npixels = np.sum(labeled == label_num)

                        # Get the coordinates of the labeled pixels
                        y_indices, x_indices = np.where(labeled == label_num)

                        # Calculate the center of mass (x, y)
                        if npixels > 0:
                            x_center = np.mean(x_indices)
                            y_center = np.mean(y_indices)
                        else:
                            x_center, y_center = 0, 0

                        # Create a LabelInfo instance and add it to the list
                        label_info = LabelInfo(label_id=label_num, 
                                               x_center=x_center, 
                                               y_center=y_center, 
                                               npixels=npixels,
                                               frame=frame_idx,
                                               channel=channel_idx,
                                               z_plane=z_idx)
                        label_info_list.append(label_info)

    return mask_all, label_info_list

def remove_small_or_large_labels(mask: np.ndarray, label_info_list: list[LabelInfo], min_size: int = 100, max_size: int = np.inf, new_index: bool = True) -> tuple[np.ndarray, list[LabelInfo]]:
    """
    Remove labels outside [min_size, max_size] range from a 5D mask and optionally reindex them.

    Args:
        mask: 5D mask with original labels.
        label_info_list: List of LabelInfo instances containing label data.
        min_size: Minimum size of labels to keep.
        max_size: Maximum size of labels to keep.
        new_index: If True, the labels in the output mask will have new sequential indices.

    Returns:
        new_mask: The updated mask with small/large labels removed.
        kept_labels: List of LabelInfo instances for the kept labels.
    """
    # Create a new mask
    new_mask = np.zeros_like(mask, dtype=mask.dtype)
    kept_labels = []

    # Filter label_info_list based on size criteria
    valid_labels = [label_info for label_info in label_info_list if min_size <= label_info.npixels <= max_size]

    if new_index:
        for new_id, label_info in enumerate(valid_labels, start=1):
            # Move the mask for this label from the original mask to the new mask
            new_mask[label_info.frame, label_info.channel, label_info.z_plane][mask[label_info.frame, label_info.channel, label_info.z_plane] == label_info.label_id] = new_id
            
            # Append the new LabelInfo with new ID
            kept_labels.append(LabelInfo(label_id=new_id,
                                          x_center=label_info.x_center,
                                          y_center=label_info.y_center,
                                          npixels=label_info.npixels,
                                          frame=label_info.frame,
                                          channel=label_info.channel,
                                          z_plane=label_info.z_plane))
    else:
        for label_info in valid_labels:
            # Move the mask for this label from the original mask to the new mask
            new_mask[label_info.frame, label_info.channel, label_info.z_plane][mask[label_info.frame, label_info.channel, label_info.z_plane] == label_info.label_id] = label_info.label_id
            
            # Append the same LabelInfo
            kept_labels.append(label_info)

    return new_mask, kept_labels


def split_large_labels_with_watershed(mask: np.ndarray, label_info_list: list[LabelInfo], max_size: int = np.inf) -> tuple[np.ndarray, list[LabelInfo]]:
    """
    Split large labels using the watershed algorithm and return updated label information.

    Args:
        mask: 5D mask with original labels.
        label_info_list: List of LabelInfo instances containing label data.
        max_size: Maximum size of labels to split.

    Returns:
        new_mask: The updated mask with large labels split.
        updated_label_info: List of updated LabelInfo instances for the new labels.
    """
    new_mask = np.copy(mask)  # Create a new mask for updated labels
    updated_label_info = []  # List to store updated LabelInfo instances


    # Iterate through the provided label_info_list
    for label_info in label_info_list:
        if label_info.npixels > max_size:
            # Extract the specific slice for this label
            t_idx = label_info.frame
            c_idx = label_info.channel
            z_idx = label_info.z_plane
            
            # Create binary mask for the current label
            slice_ = mask[t_idx, c_idx, z_idx]
            binary_mask = (slice_ == label_info.label_id)

            # Distance Transform
            distance = distance_transform_edt(binary_mask)

            # Generate markers for the watershed
            markers = label(binary_mask)

            # Apply watershed algorithm
            new_labels = watershed(-distance, markers, mask=binary_mask)

            # Update the new_mask and record the new label information
            unique_new_labels = np.unique(new_labels)
            for new_label in unique_new_labels:
                if new_label > 0:
                    # Assign the new label in the new_mask
                    new_mask[t_idx, c_idx, z_idx][new_labels == new_label] = new_label

                    # Record the new label info
                    npixels = np.sum(new_labels == new_label)
                    y_indices, x_indices = np.where(new_labels == new_label)

                    # Calculate center
                    x_center = np.mean(x_indices) if npixels > 0 else 0
                    y_center = np.mean(y_indices) if npixels > 0 else 0

                    # Create and append a new LabelInfo instance
                    updated_label_info.append(LabelInfo(label_id=new_label,
                                                         x_center=x_center,
                                                         y_center=y_center,
                                                         npixels=npixels,
                                                         frame=t_idx,
                                                         channel=c_idx,
                                                         z_plane=z_idx))
    return new_mask, updated_label_info

def mask_to_rois(mask: np.ndarray) -> list[ImagejRoi]:
    """Convert labeled 5D mask into a list of ImageJ-compatible ROI objects."""
    rois = []
    
    # Iterate through the time and channel dimensions
    for t_idx in range(mask.shape[0]):
        for c_idx in range(mask.shape[1]):
            for z_idx in range(mask.shape[2]):
                labeled = mask[t_idx, c_idx, z_idx]
                
                # Find contours on the labeled image
                contours = find_contours(labeled, level=0.5)  # Use 0.5 as the threshold level for binary contours
                for contour in contours:  # Each contour corresponds to a different labeled region
                    # Convert the contour to ImageJ ROI
                    roi = ImagejRoi.frompoints(np.round(contour)[:, ::-1])  # Reverse the order for ImageJ compatibility
                    rois.append(roi)  # Append ROI objects directly

    return rois


def fill_holes_indexed(mask: np.ndarray) -> np.ndarray:
    """Fill holes within each labeled region independently."""
    filled = np.zeros_like(mask, dtype=mask.dtype)
    
    for t_idx in range(mask.shape[0]):
        for c_idx in range(mask.shape[1]):
            for z_idx in range(mask.shape[2]):
                slice_ = mask[t_idx, c_idx, z_idx]
                for label_id in np.unique(slice_):
                    if label_id == 0:
                        continue  # Skip background
                    region_mask = slice_ == label_id
                    filled_region = binary_fill_holes(region_mask)
                    filled[t_idx, c_idx, z_idx][filled_region] = label_id
                    
    return filled


def plot_mask(mask: np.ndarray, label_sizes:dict[int, int] ) -> None:
    """Plot the mask with a color map."""
    cmap = ListedColormap(np.vstack((np.array([[1, 1, 1]]),
                                     np.random.rand(len(label_sizes), 3))))  

    plt.imshow(mask[0, 1, 0], cmap=cmap)  # Plotting the first time point, channel, and z-slice
    plt.colorbar()
    plt.title("Mask at T=0, C=1, Z=0")
    plt.show()

def process_file(path: str, method: str = "otsu", channels: list = [1], median_filter_size: int = 10) -> tuple[np.ndarray, list[ImagejRoi]]:
    """Main function to load image, segment it, and generate ROIs."""
    print("Loading image...")
    img = rp.load_bioio(path)
    print("Input image shape:", img.data.shape)  

    print("Median filtering...")
    mask = apply_median_filter(img.data, size=median_filter_size, channels=channels)  
    print("Mask shape after filter:", mask.shape)

    print("Applying threshold...")
    mask, labelinfo = apply_threshold(mask, method=method, channels=channels) 
    print("Mask shape affter threshold:", mask.shape)
    print("ROIs generated:", len(labelinfo))
    plot_mask(mask, labelinfo)

    print("Removing small labels...")
    mask, labelinfo = remove_small_or_large_labels(mask, labelinfo, min_size=10_000, max_size=np.inf)
    print("Mask shape after removing small labels:", mask.shape)
    print("ROIs generated:", len(labelinfo))
    plot_mask(mask, labelinfo)

    # print("Split large labels with waterhed...")
    # TODO Implement watershed splitting of large labels
    # mask, labelinfo = split_large_labels_with_watershed(mask, labelinfo, max_size=np.inf)

    # print("Remove masks that touch xy edges Z is optional")
    # TODO Implement remove masks that touch xy edges or z edges
    # mask, labelinfo = remove_on_xy_edges(mask, labelinfo, remove_xy_edges=True, remove_z_edges=False)
    
    # Fill holes in the mask
    print("Filling holes...")
    mask = fill_holes_indexed(mask)
    print("Mask shape after filling holes:", mask.shape)

    print(labelinfo)
    
    rois = mask_to_rois(mask)
    print("ROIs generated:", len(rois))

    plot_mask(mask, labelinfo)

    return mask, rois


image_path = r"Z:\Schink\Oyvind\colaboration_user_data\20250124_Viola\output_cellpose\230705_93_mNG-DFCP1_LT_LC3_CMvsLPDS__CM__CM_chol_2h__2023-07-07__230705_mNG-DFCP1_LT_LC3_CM_chol_2h_1_input_cellpose.tif"
masks, rois = process_file(image_path, method="otsu", channels=[1])
print("Final mask shape:", masks.shape)

# Uncomment the following lines for command line interface
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("image_path", type=str, help="Path to .npy file containing TCZYX image")
#     parser.add_argument("--method", type=str, default="otsu", help="Thresholding method")
#     parser.add_argument("--test", action="store_true", help="Run test panel of all methods")
#     args = parser.parse_args()

#     image = np.load(args.image_path)

#     if args.test:
#         test_threshold_methods(image)
#     else:
#         mask, roi_zip_path = process_file(args.image_path, method=args.method)  # Ensure function call is accurate
#         print(f"Segmentation complete. Mask shape: {mask.shape}, ROIs saved to: {roi_zip_path}")

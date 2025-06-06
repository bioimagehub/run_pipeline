import argparse
import os
from tqdm import tqdm
# import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from skimage.filters import (
    threshold_otsu, threshold_yen, threshold_li, threshold_triangle,
    threshold_mean, threshold_minimum, threshold_isodata,
    threshold_niblack, threshold_sauvola
)
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from scipy.ndimage import binary_fill_holes, median_filter, distance_transform_edt

from cv2 import findContours, RETR_EXTERNAL, CHAIN_APPROX_NONE

from roifile import ImagejRoi, roiwrite, roiread

from bioio.writers import OmeTiffWriter

from joblib import Parallel, delayed

import yaml

# local imports
import run_pipeline_helper_functions as rp


# TODO alow min and max size to be set in pixels or physical units

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
        return (f"LabelInfo(label_id={self.label_id}, "
                f"x_center={self.x_center:.2f}, "
                f"y_center={self.y_center:.2f}, "
                f"npixels={self.npixels}, "
                f"frame={self.frame}, "
                f"channel={self.channel}, "
                f"z_plane={self.z_plane})")

    def to_dict(self):
        return {
            "label_id": int(self.label_id),
            "x_center": float(self.x_center),
            "y_center": float(self.y_center),
            "npixels": int(self.npixels),
            "frame": int(self.frame),
            "channel": int(self.channel),
            "z_plane": int(self.z_plane)
        }
    @staticmethod
    def save(label_info_list, filepath):
        data = [label.to_dict() for label in label_info_list]
        with open(filepath, 'w') as f:
            yaml.dump(data, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return [LabelInfo(**item) for item in data]

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> list:
        """Create LabelInfo objects from a labeled 5D mask (TCZYX)."""
        t, c, z, y, x = mask.shape
        label_info_list = []

        for frame in range(t):
            for channel in range(c):
                for z_plane in range(z):
                    labeled = mask[frame, channel, z_plane]
                    unique_labels = np.unique(labeled)
                    for label_num in unique_labels:
                        if label_num == 0:
                            continue
                        y_indices, x_indices = np.where(labeled == label_num)
                        npixels = len(x_indices)
                        if npixels > 0:
                            x_center = np.mean(x_indices)
                            y_center = np.mean(y_indices)
                        else:
                            x_center = y_center = 0
                        label_info_list.append(
                            cls(label_id=label_num,
                                x_center=x_center,
                                y_center=y_center,
                                npixels=npixels,
                                frame=frame,
                                channel=channel,
                                z_plane=z_plane)
                        )
        return label_info_list

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
    mask_out:np.uint16 = np.zeros((t, c, z, y, x), dtype=np.uint16)  # 16 bit since it can be a lot of signals. Can change to 8bit later in the code
    
    threshold_fn = threshold_methods.get(method)
    if threshold_fn is None:
        raise ValueError(f"Unsupported method: {method}")
    
    for frame in frames: # Iterate over defined frames to process only
        for channel in channels: # Iterate over defined channels to process only
            for z_plane in range(z):
                img_plane = image[frame, channel, z_plane]
                try:
                    thresh = threshold_fn(img_plane)
                    binary = img_plane > thresh
                except Exception as e:
                    print(f"Threshold failed on T={frame}, C={channel}, Z={z_plane}: {e}")
                    continue

                labeled = label(binary)

                # Count unique labels and check if max exceeds 255
                unique_labels = np.unique(labeled)
                if len(unique_labels) > (2 ** 16) - 1:
                    raise ValueError(f"Error: More than {(2 ** 16) - 1}] labels found inT={frame}, C={channel}, Z={z_plane}. Please check the input image or consider upgrading from 16bit mask.")

                # Store labels in the appropriate position in the mask
                mask_out[frame, channel, z_plane] = labeled.astype(np.uint16)

                
    return mask_out, LabelInfo.from_mask(mask_out)

def remove_small_or_large_labels(mask: np.ndarray, label_info_list: list[LabelInfo], min_size: int = 100, max_size: int = np.inf) -> tuple[np.ndarray, list[LabelInfo]]:
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
    mask_out = np.zeros_like(mask, dtype=mask.dtype)

    # Filter label_info_list based on size criteria
    valid_labels = [label_info for label_info in label_info_list if min_size <= label_info.npixels <= max_size]

    
    for new_id, label_info in enumerate(valid_labels, start=1):
        # Move the mask for this label from the original mask to the new mask
        mask_out[label_info.frame, label_info.channel, label_info.z_plane][mask[label_info.frame, label_info.channel, label_info.z_plane] == label_info.label_id] = new_id
    return mask_out, LabelInfo.from_mask(mask_out)

def remove_on_edges(mask: np.ndarray, label_info_list: list[LabelInfo], remove_xy_edges: bool = True, remove_z_edges: bool = False) -> tuple[np.ndarray, list[LabelInfo]]:
    """
    Remove labels that touch the edges of the image in XY or Z dimensions.

    Args:
        mask: 5D mask with original labels.
        label_info_list: List of LabelInfo instances containing label data.
        remove_xy_edges: If True, remove labels touching XY edges.
        remove_z_edges: If True, remove labels touching Z edges.

    Returns:
        new_mask: The updated mask with labels touching edges removed.
        updated_label_info: List of updated LabelInfo instances for the remaining labels.
    """
    mask_out = np.zeros_like(mask, dtype=mask.dtype)
    
    # Iterate through the provided label_info_list
    for label_info in label_info_list:
        if label_info.npixels > 0:
            # Extract the specific slice for this label
            t_idx = label_info.frame
            c_idx = label_info.channel
            z_idx = label_info.z_plane

            # Create binary mask for the current label
            slice_ = mask[t_idx, c_idx, z_idx]
            binary_mask = (slice_ == label_info.label_id)

            # Check if the label touches the edges
            if remove_xy_edges and (np.any(binary_mask[0, :]) or np.any(binary_mask[-1, :]) or np.any(binary_mask[:, 0]) or np.any(binary_mask[:, -1])):
                continue  # Skip this label

            if remove_z_edges and (np.any(binary_mask[:, :, 0]) or np.any(binary_mask[:, :, -1])):
                continue  # Skip this label

            # If it doesn't touch the edges, keep it in the new mask and record its info
            mask_out[t_idx, c_idx, z_idx][binary_mask] = label_info.label_id

    return mask_out, LabelInfo.from_mask(mask_out)

def split_large_labels_with_watershed(mask: np.ndarray, label_info_list: list[LabelInfo], max_size: int = np.inf) -> tuple[np.ndarray, list[LabelInfo]]:
    """
    Split large labels using the watershed algorithm and return updated label information.
    This function will only split labels larger than max_size.
    It uses the distance transform and watershed algorithm to split large labels into smaller ones.

    Args:
        mask: 5D mask with original labels.
        label_info_list: List of LabelInfo instances containing label data.
        max_size: Maximum size of labels to split.

    Returns:
        new_mask: The updated mask with large labels split.
        updated_label_info: List of updated LabelInfo instances for the new labels.
    """
    mask_out = np.zeros_like(mask, dtype=mask.dtype)
    next_available_label = np.max(mask) + 1  # Start new labels from the max existing label + 1

    # Iterate through the provided label_info_list
    for label_info in label_info_list:
        # Extract the specific slice for this label
        t_idx = label_info.frame
        c_idx = label_info.channel
        z_idx = label_info.z_plane
        slice_ = mask[t_idx, c_idx, z_idx]


        if label_info.npixels > 2*max_size:
            # remove very big labels
            continue  # Skip this label
        elif  label_info.npixels > max_size:
            # Create binary mask for the current label
            label_mask = (slice_ == label_info.label_id)


            # Distance Transform
            distance = distance_transform_edt(label_mask)

            # Find local maxima
            all_coords = peak_local_max(
                distance, 
                labels=label_mask,
                num_peaks=2,
                min_distance=10,  # Minimum distance between peaks 
                exclude_border=False
            ) # Will return more than 2 peaks if there are more than 2 local maxima witht the same value
            sorted_coords = sorted(all_coords, key=lambda c: distance[tuple(c)], reverse=True)
            coords = sorted_coords[:2]  # Take the top 2 coordinates furthest apart


            markers = np.zeros_like(label_mask, dtype=np.int32)
            for i, coord in enumerate(coords):
                markers[coord[0], coord[1]] = next_available_label
                next_available_label += 1

            # Apply watershed algorithm
            new_labels = watershed(-distance, markers, mask=label_mask, connectivity=2)
            # Update the new_mask and record the new label information
            unique_new_labels = np.unique(new_labels)
            for new_label in unique_new_labels:
                if new_label > 0:
                    # Assign the new label in the new_mask
                    mask_out[t_idx, c_idx, z_idx][new_labels == new_label] = new_label
        
        else:
            mask_out[t_idx, c_idx, z_idx][slice_ == label_info.label_id] = label_info.label_id
    return mask_out, LabelInfo.from_mask(mask_out)

def plot_image_overlays(image, overlays, **kwargs):
    """Plot image and overlays (bytes) using matplotlib."""
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    if not isinstance(overlays, list):
        overlays = [overlays]
    for overlay in overlays:
        roi = ImagejRoi.frombytes(overlay)
        roi.plot(ax, **kwargs)
    plt.show()

def plot_mask(mask: np.ndarray, label_sizes:dict[int, int] ) -> None:
    """Plot the mask with a color map."""
    cmap = ListedColormap(np.vstack((np.array([[1, 1, 1]]),
                                     np.random.rand(len(label_sizes), 3))))  

    plt.imshow(mask[0, 1, 0], cmap=cmap)  # Plotting the first time point, channel, and z-slice
    plt.colorbar()
    plt.title("Mask at T=0, C=1, Z=0")
    plt.show()

def outlines_list_single(masks):
    """Get outlines of masks as a list to loop over for plotting.

    Args:
        masks (ndarray): masks (0=no cells, 1=first cell, 2=second cell,...)

    Returns:
        list: List of outlines as pixel coordinates.

    """
    outpix = []
    for n in np.unique(masks)[1:]:
        mn = masks == n
        if mn.sum() > 0:
            contours = findContours(mn.astype(np.uint8), mode=RETR_EXTERNAL,
                                        method=CHAIN_APPROX_NONE)
            contours = contours[-2]
            cmax = np.argmax([c.shape[0] for c in contours])
            pix = contours[cmax].astype(int).squeeze()
            if len(pix) > 4:
                outpix.append(pix)
            else:
                outpix.append(np.zeros((0, 2)))
    return outpix

def mask_to_rois(mask: np.ndarray, label_info_list: list[LabelInfo]) -> list[ImagejRoi]:
    """Convert labeled 5D mask into a list of ImageJ-compatible ROI objects."""

    outlines = []

    for label_info in label_info_list:
        # Extract the specific slice for this label
        t_idx = label_info.frame
        c_idx = label_info.channel
        z_idx = label_info.z_plane
        slice_ = mask[t_idx, c_idx, z_idx]
        label_mask = (slice_ == label_info.label_id)

        if label_mask.sum() == 0:
            continue  # Skip if label has no pixels

        # Find contours
        contours, _ = findContours(label_mask.astype(np.uint8), mode=RETR_EXTERNAL, method=CHAIN_APPROX_NONE)

        if not contours:
            outlines.append(np.zeros((0, 2)))
            continue

        # Take the largest contour by number of points
        cmax = np.argmax([c.shape[0] for c in contours])
        pix = contours[cmax].astype(int).squeeze()

        if pix.ndim != 2 or pix.shape[0] <= 4:
            outlines.append(np.zeros((0, 2)))
        else:
            outlines.append(pix)

    # Filter out empty outlines
    nonempty_outlines = [outline for outline in outlines if outline.shape[0] > 0]

    if len(nonempty_outlines) < len(outlines):
        print(f"Empty outlines found, saving {len(nonempty_outlines)} ImageJ ROIs to .zip archive.")

    rois = [ImagejRoi.frompoints(outline) for outline in nonempty_outlines]

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

def save_intermediate(mask: np.ndarray=None, labelinfo=None, rois=None, path=None, physical_pixel_sizes=None):
    """Utility function to save the images and label info.
    if mask is None or if labelinfo is {} they willnot be saved    
    """
    if not isinstance(path, str):
        return
    if mask is not None:
        OmeTiffWriter.save(mask, path + ".tif", dim_order="TCZYX", physical_pixel_sizes=physical_pixel_sizes) 
    if labelinfo is not None:
        LabelInfo.save(labelinfo, path + "_labelinfo.yaml")
    if rois is not None:
        rois_path = path + "_rois.zip"
        if os.path.exists(rois_path): # Will just append to folder not replace
            os.remove(rois_path)
        roiwrite(rois_path, rois)

def process_file(
    input_path: str,
    output_name: str,
    channels: list[int] = [1],
    tracking_channel: int | None = None,
    median_filter_size: list[int] = [10],
    threshold_method: list[str] = ["otsu"],
    min_size: int = 10_000,
    max_size: int = 55_000,
    watershed_large_labels: bool = True,
    remove_xy_edges: bool = True,
    remove_z_edges: bool = False,
    tmp_output_folder: str | None = None,
    yaml_file_extension: str = "_metadata.yaml",
) -> tuple["np.ndarray", list, "LabelInfo"]:
    """
    Main function to load an image, segment it, and generate ROIs.

    Parameters
    ----------
    channels : list[int]
        One-based channel indices to process.
    median_filter_size : list[int]
        Per-channel median filter sizes (same length as `channels`).
    threshold_method : list[str]
        Per-channel thresholding methods (same length as `channels`).
    tracking_channel : int | None
        If > -1, channel index (one-based) used for tracking; -1 skips tracking.
        If None, defaults to channels[0].
    ...
    Returns
    -------
    mask : np.ndarray
    rois : list
    labelinfo : LabelInfo
    """
    

    # ---------------------------------------------------------------------
    # 0.  Attach segmentation parameters to the YAML metadata  ------------
    # ---------------------------------------------------------------------
    yaml_path_input = os.path.splitext(input_path)[0] + yaml_file_extension
    if os.path.isfile(yaml_path_input):
        with open(yaml_path_input, "r") as f:
            metadata = yaml.safe_load(f)
        metadata["Threshold segmentation"] = {
            "Channels": channels,
            "Tracking channel": tracking_channel,
            "Median filter size": dict(zip(channels, median_filter_size)),
            "Threshold method": dict(zip(channels, threshold_method)),
            "Min size": min_size,
            "Max size": max_size,
            "Remove xy edges": remove_xy_edges,
            "Remove z edges": remove_z_edges,
        }
        yaml_path_output = os.path.splitext(output_name)[0] + yaml_file_extension
        with open(yaml_path_output, "w") as f:
            yaml.dump(metadata, f)
    

    # ---------------------------------------------------------------------
    # 1.  Validate / set tracking_channel  --------------------------------
    # ---------------------------------------------------------------------
    if tracking_channel is None:
        tracking_channel = channels[0]
        print(
            f"Warning: no tracking channel set — using channels[0] ({tracking_channel}); "
            "use --tracking-channel -1 to skip tracking."
        )
    elif tracking_channel != -1 and tracking_channel not in channels:
        raise ValueError(
            f"tracking_channel ({tracking_channel}) must be one of {channels} or -1"
        )

    # ---------------------------------------------------------------------
    # 2.  Load image & make tmp dir  --------------------------------------
    # ---------------------------------------------------------------------
    img = rp.load_bioio(input_path)  # raises on failure
    if tmp_output_folder is not None:
        os.makedirs(tmp_output_folder, exist_ok=True)
    channels_str = "_ch" + "-".join(map(str, channels))

    # ---------------------------------------------------------------------
    # 3.  MEDIAN FILTER  ---------------------------------------------------
    #     `apply_median_filter` is expected to accept a list for `size`.
    # ---------------------------------------------------------------------
    tmp_path = (
        None
        if tmp_output_folder is None
        else os.path.join(
            tmp_output_folder,
            f"{os.path.splitext(os.path.basename(input_path))[0]}_{channels_str}_01_med",
        )
    )
    if tmp_path and os.path.exists(tmp_path):
        mask = rp.load_bioio(tmp_path).data
    else:
        mask = apply_median_filter(
            img.data, size=median_filter_size, channels=channels
        )
        save_intermediate(
            mask=mask,
            labelinfo=None,
            rois=None,
            path=tmp_path,
            physical_pixel_sizes=img.physical_pixel_sizes,
        )

    # ---------------------------------------------------------------------
    # 4.  THRESHOLD  -------------------------------------------------------
    #     `apply_threshold` is expected to accept a list for `method`.
    # ---------------------------------------------------------------------
    tmp_path = (
        None
        if tmp_output_folder is None
        else os.path.join(
            tmp_output_folder,
            f"{os.path.splitext(os.path.basename(input_path))[0]}_{channels_str}_02_thresh",
        )
    )
    if tmp_path and os.path.exists(tmp_path + ".tif"):
        mask = rp.load_bioio(tmp_path + ".tif").data
        labelinfo = LabelInfo.load(tmp_path + "_labelinfo.yaml")
    else:
        mask, labelinfo = apply_threshold(
            mask, method=threshold_method, channels=channels
        )
        save_intermediate(
            mask=mask,
            labelinfo=labelinfo,
            rois=None,
            path=tmp_path,
            physical_pixel_sizes=img.physical_pixel_sizes,
        )

    # ---------------------------------------------------------------------
    # 5.  Remaining pipeline (unchanged)  ---------------------------------
    #     All later steps use `mask` / `labelinfo` exactly as before.
    # ---------------------------------------------------------------------


    #### remaining code as before

    
    try: # Remove small labels
        tmp_path = os.path.join(tmp_output_folder, os.path.splitext(os.path.basename(input_path))[0] + f"_{channels_str}_03_rm_small") if tmp_output_folder is not None else None
        if tmp_path is not None and os.path.exists(tmp_path + ".tif"):
            mask = rp.load_bioio(tmp_path + ".tif").data
            labelinfo = LabelInfo.load(tmp_path + "_labelinfo.yaml")
        else:
            max_size_filter = np.inf if watershed_large_labels else max_size
            mask, labelinfo = remove_small_or_large_labels(mask, labelinfo, min_size=min_size, max_size=max_size_filter)
            save_intermediate(mask=mask, labelinfo=labelinfo, rois=None, path=tmp_path, physical_pixel_sizes=img.physical_pixel_sizes)
    except Exception as e:
            print(f"Error Remove small labels {input_path}: {e}")
    
    try: # Remove cells touching edges
        tmp_path = os.path.join(tmp_output_folder, os.path.splitext(os.path.basename(input_path))[0] + f"_{channels_str}_04_rm_edges") if tmp_output_folder is not None else None
        if tmp_path is not None and os.path.exists(tmp_path + ".tif"):
            mask = rp.load_bioio(tmp_path + ".tif").data
            labelinfo = LabelInfo.load(tmp_path + "_labelinfo.yaml")
        else:
            mask, labelinfo = remove_on_edges(mask, labelinfo, remove_xy_edges=remove_xy_edges, remove_z_edges=remove_z_edges)
            save_intermediate(mask=mask, labelinfo=labelinfo, rois=None, path=tmp_path, physical_pixel_sizes=img.physical_pixel_sizes)
    except Exception as e:
                print(f"Error Remove cells touching edges {input_path}: {e}")

    try: # Fill holes in the mask
        tmp_path = os.path.join(tmp_output_folder, os.path.splitext(os.path.basename(input_path))[0] + f"_{channels_str}_05_fill_holes") if tmp_output_folder is not None else None
        if tmp_path is not None and os.path.exists(tmp_path + ".tif"):
            mask = rp.load_bioio(tmp_path + ".tif").data
            labelinfo = LabelInfo.load(tmp_path + "_labelinfo.yaml")
        
        else:
            mask = fill_holes_indexed(mask)
            save_intermediate(mask=mask, labelinfo=labelinfo, rois=None, path=tmp_path, physical_pixel_sizes=img.physical_pixel_sizes)
    except Exception as e:
                    print(f"Error Remove cells touching edges {input_path}: {e}")
            
        
    try: # Split large labels with watershed
        if watershed_large_labels and max_size != np.inf:
            tmp_path = os.path.join(tmp_output_folder, os.path.splitext(os.path.basename(input_path))[0] + f"_{channels_str}_06_split_watershed") if tmp_output_folder is not None else None
            if tmp_path is not None and os.path.exists(tmp_path + ".tif"):
                mask = rp.load_bioio(tmp_path + ".tif").data
                labelinfo = LabelInfo.load(tmp_path + "_labelinfo.yaml")
            else:
                mask, labelinfo = split_large_labels_with_watershed(mask, labelinfo, max_size=max_size)
                save_intermediate(mask=mask, labelinfo=labelinfo, rois=None, path=tmp_path, physical_pixel_sizes=img.physical_pixel_sizes)
        elif max_size == np.inf:
            print("Warning: max_size is set to np.inf, watershed will not be applied.")
    except Exception as e:
                    print(f"Error Watershed {input_path}: {e}")


    try:
        if img.dims.T > 1 and tracking_channel != -1:
            tmp_path = os.path.join(tmp_output_folder, os.path.splitext(os.path.basename(input_path))[0] + f"_{channels_str}_07_tracking") if tmp_output_folder is not None else None

            from track_indexed_mask import track_labels_with_trackpy    
            df_tracked, mask = track_labels_with_trackpy(mask, channel_zero_base= tracking_channel)
            labelinfo = LabelInfo.from_mask(mask)
            save_intermediate(mask=mask, labelinfo=labelinfo, rois=None, path=tmp_path, physical_pixel_sizes=img.physical_pixel_sizes)
            df_tracked.to_csv(tmp_path + ".tsv", sep='\t', index=False)  # index=False to not include row indices

    except Exception as e:
        print(f"Error tracking {input_path}: {e}")

    try: # Generate ROIs
        tmp_path = os.path.join(tmp_output_folder, os.path.splitext(os.path.basename(input_path))[0] + f"_{channels_str}_08_rois.zip") if tmp_output_folder is not None else None
        if tmp_path is not None and os.path.exists(tmp_path):   
            rois = roiread(tmp_path)
        else:
            rois = mask_to_rois(mask, labelinfo)
            save_intermediate(mask=None, labelinfo=None, rois=rois, path=tmp_path, physical_pixel_sizes=img.physical_pixel_sizes)
            
    except Exception as e:
            print(f"Error Generate ROIs {input_path}: {e}")
        
    # And then finally save mask, labelinfo and ROIS in the output folder
    save_intermediate(mask=mask, labelinfo=labelinfo, rois=rois, path=output_name, physical_pixel_sizes=img.physical_pixel_sizes)

    return mask, rois, labelinfo


def process_folder(args: argparse.Namespace) -> None: # Paralel not working yet
    # Find files to process
    files_to_process = rp.get_files_to_process(args.input_file_or_folder, ".tif", search_subfolders=False)

    
    
    # Make output folder
    os.makedirs(args.output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

    if args.use_parallel:  # Process each file in parallel
        raise NotImplementedError
        
        Parallel(n_jobs=-1)(
            delayed(process_file)(
                input_path = input_file_path,
                output_name=os.path.join(args.output_folder, os.path.splitext(os.path.basename(input_file_path))[0]),
                channels = args.channels,
                median_filter_size = args.median_filter_size,
                threshold_method = args.threshold_method,
                min_size = args.min_size,
                max_size = args.max_size,
                watershed_large_labels = args.watershed_large_labels,
                remove_xy_edges = args.remove_xy_edges,
                remove_z_edges = args.remove_z_edges,
                tmp_output_folder = args.tmp_output_folder
            )
            for input_file_path in tqdm(files_to_process, desc="Processing files", unit="file")
        )

    else:  # Process each file sequentially        
        # print(f"Process {len(files_to_process)} file(s) sequentially ")
        for input_file_path in tqdm(files_to_process, desc="Processing files", unit="file"):
            # Define output file name
            output_tif_file_name: str = os.path.join(args.output_folder, os.path.splitext(os.path.basename(input_file_path))[0])
            # Process file
            try:
                '''
                path: str, 
                 channels: list = [1], 
                 median_filter_size: int = 10,
                 method: str = "otsu",   
                 min_size=10_000, max_size=55_000, watershed_large_labels=True,
                 remove_xy_edges=True, remove_z_edges=False,
                 tmp_output_folder: str = None) -> tuple[np.ndarray, list[ImagejRoi], list[LabelInfo]]:
                '''
                process_file(
                    input_path = input_file_path,
                    output_name = output_tif_file_name,
                    channels = args.channels,
                    tracking_channel = args.tracking_channel,
                    median_filter_size = args.median_filter_size,
                    threshold_method = args.threshold_method,
                    min_size = args.min_size,
                    max_size = args.max_size,
                    watershed_large_labels = args.watershed_large_labels,
                    remove_xy_edges = args.remove_xy_edges,
                    remove_z_edges = args.remove_z_edges,
                    tmp_output_folder = args.tmp_output_folder,
                    yaml_file_extension = parsed_args.yaml_file_extension
                    )
                  # Process each file

            except Exception as e:
                print(f"Error processing {input_file_path}: {e}")
                continue
def main(parsed_args: argparse.Namespace):
    
    # Check if the input is a file or a folder
    if os.path.isfile(parsed_args.input_file_or_folder):
        print(f"Processing single file: {parsed_args.input_file_or_folder}")
        output_file_name = os.path.splitext(os.path.basename(parsed_args.input_folder))[0] + "_segmented.tif"  # Example output naming
        output_file_path = os.path.join(parsed_args.output_folder, output_file_name)
        process_file(input_path = parsed_args.input_file_or_folder,
                 output_name = output_file_path,
                 channels = parsed_args.channels,
                 tracking_channel = parsed_args.tracking_channel, 
                 median_filter_size = parsed_args.median_filter_size,
                 threshold_method = parsed_args.threshold_method,   
                 min_size = parsed_args.min_size, max_size = parsed_args.max_size, watershed_large_labels = parsed_args.watershed_large_labels,
                 remove_xy_edges = parsed_args.remove_xy_edges, remove_z_edges = parsed_args.remove_z_edges,
                 tmp_output_folder = parsed_args.mp_output_folder,
                 yaml_file_extension = parsed_args.yaml_file_extension)
        
    elif os.path.isdir(parsed_args.input_file_or_folder):
        print(f"Processing folder: {parsed_args.input_file_or_folder}")
        process_folder(parsed_args)
        
    else:
        print("Error: The specified path is neither a file nor a folder.")
        return
    
def split_comma_separated_string(value):
    return list(map(int, value.split(',')))    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file-or-folder", type=str, help="Path to the file or folder to be processed.")
    parser.add_argument("--output-folder", type=str, help="Output folder for processed files")
    parser.add_argument("--channels", type=split_comma_separated_string, default=[0], help="List of channels to process (comma-separated, e.g., 0,1,2)")
    parser.add_argument("--tracking-channel", type=int, default=None, help="Channel used for tracking if frames > 1. Must be one of the --channels. default None -> channels[0], -1 skips tracking")
    parser.add_argument("--median-filter-size", type=split_comma_separated_string, default=[10], help="Size(s) of the median filter (comma-separated, e.g., 10,20)")
    parser.add_argument("--threshold-method", type=str, nargs='*', default=["li"], help="Thresholding method(s) (comma-separated, e.g., li,otsu)")  # Use nargs='*' to accept space-separated
    parser.add_argument("--min-size", type=int, default=10_000, help="Minimum size for processing")
    parser.add_argument("--max-size", type=int, default=55_000, help="Maximum size for processing")
    parser.add_argument("--watershed-large-labels", action="store_true", help="Split large cells with watershed")
    parser.add_argument("--remove-xy-edges", action="store_true", help="Remove edges in XY")
    parser.add_argument("--remove-z-edges", action="store_true", help="Remove edges in Z")
    parser.add_argument("--tmp-output-folder", type=str, help="Save intermediate steps in tmp_output_folder")
    parser.add_argument("--use-parallel", action="store_true", help="Use parallel computing")
    parser.add_argument("--yaml-file-extension", type=str, default="_metadata.yaml", help="Extension relative to basename of input image name")

    parsed_args = parser.parse_args()


    # Expand to match number of channels if needed
    num_channels = len(parsed_args.channels)
    
    # Adjust median filter size
    parsed_args.median_filter_size = [
        parsed_args.median_filter_size[i] if i < len(parsed_args.median_filter_size) else parsed_args.median_filter_size[0]
        for i in range(num_channels)
    ]
    
    # Adjust threshold methods
    parsed_args.threshold_method = [
        parsed_args.threshold_method[i] if i < len(parsed_args.threshold_method) else parsed_args.threshold_method[0]
        for i in range(num_channels)
    ]

    main(parsed_args)
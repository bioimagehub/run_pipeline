from __future__ import annotations
import argparse
import os
import sys
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

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union, Literal, Dict, Any

from collections.abc import Sequence
import sys
import yaml
import numpy as np
import os


# local imports
import run_pipeline_helper_functions as rp

# ---------------------------------------------------------------------------
# Things to implement in the future
# ---------------------------------------------------------------------------
# TODO alow min and max size to be set in pixels or physical units


# ---------------------------------------------------------------------------
# Type aliases & constants
# ---------------------------------------------------------------------------
SysExitStep = Literal[
    "median_filter",
    "background_subtract",
    "apply_threshold",
    "thresholding",
    "remove_small_labels",
    "remove_edges",
    "fill_holes",
    "watershed",
    "tracking",
    "generate_rois",
]
SysExitStepList =[None, "median_filter", "background_subtract", "apply_threshold", "thresholding", "remove_small_labels", "remove_edges", "fill_holes", "watershed", "tracking", "generate_rois"]


Mask = np.ndarray  # semantic alias for readability
ROIs = List[Any]  # adjust to concrete ROI type if available


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
    def from_mask(cls, mask: Mask) -> list[LabelInfo]:
        """Create LabelInfo objects from a labeled 5D mask (TCZYX)."""
        t, c, z, y, x = mask.shape
        label_info_list = []

        for frame in range(t):
            for channel in range(c):
                for z_plane in range(z):
                    labeled = mask[frame, channel, z_plane]
                    unique_labels = np.unique(np.asarray(labeled))
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
                                x_center=float(x_center),
                                y_center=float(y_center),
                                npixels=npixels,
                                frame=frame,
                                channel=channel,
                                z_plane=z_plane)
                        )
        return label_info_list


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

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

from typing import Optional

def save_intermediate(mask: Optional[np.ndarray] = None, labelinfo=None, rois=None, path=None, physical_pixel_sizes=None):
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

def split_comma_separated_strstring(value):
    return list(map(str, value.split(',')))    

def split_comma_separated_intstring(value):
    return list(map(int, value.split(',')))    

def _metadata_path(path: Path, suffix: str) -> Path:
    """
    Return *path* with its stem followed by *suffix*.

    >>> _metadata_path(Path("image1.tif"), "_metadata.yaml")
    PosixPath('image1_metadata.yaml')
    """
    return path.with_name(f"{path.stem}{suffix}")

def _exit_if_requested(step: str, exit_after: Optional[SysExitStep]) -> None:
    """Terminate the program if the user requested exit after *step*."""
    if step == exit_after:
        print(f"Exiting after '{step}' step as requested.")
        sys.exit(0)

def _safe_return(
    message: str,
    exc: Exception,
    current: Tuple[Optional[Mask], Optional[ROIs], Optional[List[LabelInfo]]],
) -> Tuple[Optional[Mask], Optional[ROIs], Optional[list[LabelInfo]]]:
    """Log *exc* together with *message* and return current results early."""
    print(f"{message}: {exc}")
    return current

def _make_cache_path(
    base: Path,
    channels: List[int],
    step_idx: int,
    step_name: str,
    tmp_dir: Optional[Path],
) -> Optional[Path]:
    if tmp_dir is None:
        return None
    ch_str = "-".join(map(str, channels))
    stem = f"{base.stem}_ch{ch_str}_{step_idx:02d}_{step_name}"
    return tmp_dir / stem

# ---------------------------------------------------------------------------
# Image segmentation functions
# ---------------------------------------------------------------------------

def apply_median_filter(image: np.ndarray, sizes: list[int] = [10], channels: Optional[list[int]] = None, frames: Optional[list[int]] = None) -> np.ndarray:
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
                image[frame_idx, channel_idx, z_idx] = median_filter(img_c, size=sizes[selected_channel_idx])

    return image

def subtract_background_median_filter(image: np.ndarray, kernel_sizes: Optional[list[int]], channels: Optional[list[int]] = None) -> np.ndarray:
    """Estimate background with a large median filter and subtract it for specified channels.

    Parameters
    ----------
    image : np.ndarray
        The input 5D TCZYX image (T, C, Z, Y, X).
    kernel_size : int
        The kernel size for the median filter. If ``kernel_size <= 0`` the
        function returns the image unchanged (skip).
    channels : list[int], optional
        List of channels to apply the background subtraction (defaults to all channels).

    Returns
    -------
    np.ndarray
        The processed image after background subtraction.
    """

    if kernel_sizes is None:
        return image  # No background subtraction needed

    if all(kernel_size <= 0 for kernel_size in kernel_sizes): 
        return image

    # Check the dimensions of the input image
    if len(image.shape) != 5:
        raise ValueError("Input image must be 5D (T, C, Z, Y, X)")

    t, c, z, y, x = image.shape  # Assuming image shape is (T, C, Z, Y, X)

    # Set default channels if None
    if channels is None:
        channels = list(range(c))  # Use all channels

    # Create an output image as a copy of the input
    result = np.empty_like(image, dtype=np.float32)

    # Estimate background and subtract for each selected channel
    for selected_channel_idx, channel_idx in enumerate(channels):
        for frame_idx in range(t):
            for z_idx in range(z):
                kernel_size = kernel_sizes[selected_channel_idx]
                if kernel_size <= 0:
                    # If kernel size is less than or equal to 0, skip background subtraction for this channel
                    result[frame_idx, channel_idx, z_idx] = image[frame_idx, channel_idx, z_idx]
                    continue

                img_c = image[frame_idx, channel_idx, z_idx]  # Select T=frame_idx, C=channel_idx, Z=z_idx

                # Apply large median filter to estimate the background in the XY plane
                background = median_filter(img_c, size=kernel_size)

                # Subtract the background
                result[frame_idx, channel_idx, z_idx] = img_c.astype(np.float32) - background
                result[frame_idx, channel_idx, z_idx] = np.clip(result[frame_idx, channel_idx, z_idx], 0, None)

    return result.astype(image.dtype)


def apply_threshold(
    image: np.ndarray,
    methods: Optional[Sequence[str]] = None,
    channels: Optional[Sequence[int]] = None,
    frames: Optional[Sequence[int]] = None
) -> Tuple[np.ndarray, list]:
    """Apply thresholding method to a 5D TCZYX image and return a labeled 5D mask and a list of LabelInfo."""

    if methods is None:
        raise ValueError("No thresholding methods provided. Please specify at least one method.")

    if len(image.shape) != 5:
        raise ValueError("Input image must be 5D (T, C, Z, Y, X)")

    t, c, z, y, x = image.shape

    if frames is None:
        frames = list(range(t))
    if channels is None:
        channels = list(range(c))

    if len(methods) != len(channels):
        raise ValueError("Length of 'methods' must match the number of selected channels.")

    mask_out: np.ndarray = np.zeros((t, c, z, y, x), dtype=np.uint16)

    # Compute one threshold per channel using all frames
    channel_thresholds = {}
    for channel_idx, channel in enumerate(channels):
        method_name = methods[channel_idx]
        threshold_fn = threshold_methods.get(method_name)
        if threshold_fn is None:
            raise ValueError(f"Unsupported method: {method_name}. Supported methods are: {list(threshold_methods.keys())}")

        # Stack all Z-slices from all frames into a single flat array
        data_for_threshold = image[frames, channel].reshape(-1)
        try:
            channel_thresholds[channel] = threshold_fn(data_for_threshold)
        except Exception as e:
            raise RuntimeError(f"Thresholding failed for channel {channel} with method {method_name}: {e}")

    # Apply threshold per channel across all timepoints
    for frame in frames:
        for channel in channels:
            thresh = channel_thresholds[channel]

            for z_plane in range(z):
                img_plane = image[frame, channel, z_plane]
                binary = img_plane > thresh
                labeled = label(binary)

                if isinstance(labeled, tuple):  # safety check
                    labeled = labeled[0]

                labeled = np.asarray(labeled)  # ensure it's an array
                if labeled.max() > (2 ** 16 - 1):
                    raise ValueError(f"Too many labels in T={frame}, C={channel}, Z={z_plane}. Use higher bit depth.")

                mask_out[frame, channel, z_plane] = labeled.astype(np.uint16)

    return mask_out, LabelInfo.from_mask(mask_out)

# def remove_small_or_large_labels(mask: np.ndarray, label_info_list: list[LabelInfo], min_sizes: int = 100, max_sizes: int = np.inf) -> tuple[np.ndarray, list[LabelInfo]]:
#     """
#     Remove labels outside [min_sizes, max_sizes] range from a 5D mask and optionally reindex them.

#     Args:
#         mask: 5D mask with original labels.
#         label_info_list: List of LabelInfo instances containing label data.
#         min_size: Minimum size of labels to keep.
#         max_size: Maximum size of labels to keep.
#         new_index: If True, the labels in the output mask will have new sequential indices.

#     Returns:
#         new_mask: The updated mask with small/large labels removed.
#         kept_labels: List of LabelInfo instances for the kept labels.
#     """
#     # Create a new mask
#     mask_out = np.zeros_like(mask, dtype=mask.dtype)

#     # Filter label_info_list based on size criteria
#     valid_labels = [label_info for label_info in label_info_list if min_sizes <= label_info.npixels <= max_sizes]

    
#     for new_id, label_info in enumerate(valid_labels, start=1):
#         # Move the mask for this label from the original mask to the new mask
#         mask_out[label_info.frame, label_info.channel, label_info.z_plane][mask[label_info.frame, label_info.channel, label_info.z_plane] == label_info.label_id] = new_id
#     return mask_out, LabelInfo.from_mask(mask_out)

def remove_small_or_large_labels(mask: np.ndarray, label_info_list: Optional[list[LabelInfo]], channels: Optional[list[int]] = None, frames: Optional[list[int]] = None,
                                 min_sizes: list[float] = [-np.inf], max_sizes: list[float] = [np.inf]) -> tuple[Mask, list[LabelInfo]]:
    """
    Remove labels outside [min_sizes, max_sizes] range from a 5D mask and optionally reindex them.

    Args:
        mask: 5D mask with original labels.
        label_info_list: List of LabelInfo instances containing label data.
        min_sizes: List of minimum sizes of labels to keep for each channel.
        max_sizes: List of maximum sizes of labels to keep for each channel.

    Returns:
        new_mask: The updated mask with small/large labels removed.
        kept_labels: List of LabelInfo instances for the kept labels.
    """
   
    if len(mask.shape) != 5:
        raise ValueError("Input image must be 5D (T, C, Z, Y, X)")
    
    if label_info_list is None:
        print("No label_info_list provided, generating from mask.")
        label_info_list = LabelInfo.from_mask(mask)

    # print(f"Before removing minsizes {min_sizes} and max sizes {max_sizes}: {len(label_info_list)} labels found in mask.")
    
    if channels is None:
        print("No channels specified, returning original mask and label info.")
        return mask, label_info_list  # No channels specified, return original mask and label info
    
    if len(min_sizes) != len(channels) or len(max_sizes) != len(channels):
        raise ValueError("min_sizes and max_sizes must match the number of specified channels.")
    
    if all(min_size <= 0 for min_size in min_sizes) and all(max_size >= np.inf for max_size in max_sizes):
        print("No size filtering needed, returning original mask and label info.") 
        return mask, label_info_list  # No size filtering needed, return original mask and label info
    

    t, c, z, y, x = mask.shape  # Assuming image is in the shape (T, C, Z, Y, X)
    
    if frames is None:
        frames = list(range(t))  # Use all frames if None
    if channels is None:
        channels = list(range(c))  # Use all channels if None
    
    # Ensure min_sizes and max_sizes are lists of the same length as channels and apply to all channels if len(max_sizes) == 1
    if len(min_sizes) != len(channels) and len(min_sizes) == 1: # only one min_size provided, apply it to all channels
        min_sizes = min_sizes * len(channels)
    elif len(min_sizes) != len(channels): # len(max_sizes) != len(channels) and len(max_sizes) != 1: -> raise error
        raise ValueError("min_sizes and max_sizes must match the number of specified channels. or be a single value. that will be applied to all channels.")
    # Else it should be a list of the same length as channels

    # Ensure min_sizes and max_sizes are lists of the same length as channels and apply to all channels if len(max_sizes) == 1
    if len(max_sizes) != len(channels) and len(max_sizes) == 1: # only one min_size provided, apply it to all channels
        max_sizes = max_sizes * len(channels)
    elif len(max_sizes) != len(channels): # len(max_sizes) != len(channels) and len(max_sizes) != 1: -> raise error
        raise ValueError("max_sizes and max_sizes must match the number of specified channels. or be a single value. that will be applied to all channels.")
    # Else it should be a list of the same length as channels

    # Create a new mask
    mask_out = np.zeros_like(mask, dtype=mask.dtype)
    # print("mask out shape" ,mask_out.shape)

    # look in the labelinfo file and move over the labels (from mask) that are within the min and max size range
    
    for label_info in label_info_list:
        ch_to_process = label_info.channel
        
        if ch_to_process not in channels:
            continue
        
        
        npixels = label_info.npixels
        # print(npixels, max_sizes[ch], min_sizes[ch])
        # get the index of the channel in the channels list
        ch_index = channels.index(ch_to_process)  # Get the index of the channel in the channels list

        if max_sizes[ch_index] < npixels:
            continue
        if npixels < min_sizes[ch_index]:
            continue

        # Move the mask with this ID from the original mask to the new mask
        frame = label_info.frame
        if frame not in frames:
            continue
        plane = label_info.z_plane
        if plane >= mask.shape[2]:
            print(f"Warning: Skipping label {label_info.label_id} in frame {frame}, channel {ch_to_process}, plane {plane} due to out-of-bounds index.")
            continue

        label_id = label_info.label_id
        # print(f"Moving label {label_id} from frame {frame}, channel {ch_to_process}, plane {plane} with size {npixels} to new mask.")
        # This ID should be foved from input maksk to output mask
        mask_out[frame, ch_to_process, plane][mask[frame, ch_to_process, plane] == label_id] = label_id

    labelinfo_out = LabelInfo.from_mask(mask_out)
    # print(f"After removing: {len(labelinfo_out)} labels found in mask.")
    return mask_out, labelinfo_out


def remove_on_edges(mask: np.ndarray, label_info_list: Optional[list[LabelInfo]], remove_xy_edges: bool = True, remove_z_edges: bool = False) -> tuple[np.ndarray, list[LabelInfo]]:
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
    if label_info_list is None:
        label_info_list = LabelInfo.from_mask(mask)
    
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



def split_large_labels_with_watershed(
    mask: np.ndarray,
    label_info_list: Optional[list[LabelInfo]],
    *,
    channels: Optional[list[int]] = None,
    frames: Optional[list[int]] = None,
    max_sizes: list[float] = [np.inf],
    min_peak_distance: int = 10,
    max_num_peaks: int = 2,
) -> tuple[np.ndarray, list["LabelInfo"]]:
    """Split every label whose voxel-count exceeds a per-channel threshold."""
    # -------------------- sanity checks & bookkeeping --------------------
    if mask.ndim != 5:
        raise ValueError("mask must be 5-D (T, C, Z, Y, X)")
    
    if label_info_list is None:
        label_info_list = LabelInfo.from_mask(mask)
    
    T, C, Z, Y, X = mask.shape
    frames   = list(range(T)) if frames   is None else frames
    channels = list(range(C)) if channels is None else channels

    if len(max_sizes) == 1:
        max_sizes *= len(channels)
    if len(max_sizes) != len(channels):
        raise ValueError("max_sizes must be length-1 or equal to len(channels)")

    ch_thresh = {ch: max_sizes[i] for i, ch in enumerate(channels)}

    mask_out = np.zeros_like(mask, dtype=mask.dtype)
    next_id  = int(mask.max()) + 1

    # ---------------------------- main loop -----------------------------
    for info in label_info_list:
        if info.channel not in channels or info.frame not in frames:
            continue

        thr = ch_thresh[info.channel]
        f, ch, zp = info.frame, info.channel, info.z_plane

        src_slice = mask[f, ch, zp]
        dst_slice = mask_out[f, ch, zp]

        if info.npixels <= thr:                       # keep as-is
            dst_slice[src_slice == info.label_id] = info.label_id
            continue

        lbl_mask = src_slice == info.label_id
        if lbl_mask.sum() == 0:
            continue                                  # nothing to do

        # ---------------------- distance-TX & peak pick ------------------
        dist = distance_transform_edt(lbl_mask)       # ndarray, may return None
        if dist is None or not isinstance(dist, np.ndarray):
            print(f"Warning: Distance transform failed for label {info.label_id}. Skipping.")
            dst_slice[lbl_mask] = info.label_id
            continue

        peak_xy = peak_local_max(
            dist,
            labels=lbl_mask,
            num_peaks=max_num_peaks,
            min_distance=min_peak_distance,
            exclude_border=False,
        )
        if peak_xy.shape[0] < 2:                      # not enough peaks
            dst_slice[lbl_mask] = info.label_id
            continue

        # rank peaks by distance value (vectorised – silences Pylance)
        if peak_xy is not None and peak_xy.size > 0:
            peak_vals = dist[tuple(peak_xy.T)]
        else:
            peak_vals = np.array([])  # Handle the case where no peaks are found
        peak_xy   = peak_xy[np.argsort(peak_vals)[::-1][:max_num_peaks]]

        # -------------------------- watershed ---------------------------
        markers = np.zeros_like(lbl_mask, dtype=np.int32)
        for yx in peak_xy:
            markers[tuple(yx)] = next_id
            next_id += 1

        split = watershed(-dist, markers=markers, mask=lbl_mask, connectivity=2)
        for new_lbl in np.unique(split):
            if new_lbl > 0:
                dst_slice[split == new_lbl] = new_lbl

    # ---------------------- rebuild LabelInfo list ----------------------
    return mask_out, LabelInfo.from_mask(mask_out)

# def split_large_labels_with_watershed(mask: np.ndarray, label_info_list: list[LabelInfo], max_size: list[float] = [np.inf]) -> tuple[np.ndarray, list[LabelInfo]]:
#     """
#     Split large labels using the watershed algorithm and return updated label information.
#     This function will only split labels larger than max_size.
#     It uses the distance transform and watershed algorithm to split large labels into smaller ones.

#     Args:
#         mask: 5D mask with original labels.
#         label_info_list: List of LabelInfo instances containing label data.
#         max_size: Maximum size of labels to split.

#     Returns:
#         new_mask: The updated mask with large labels split.
#         updated_label_info: List of updated LabelInfo instances for the new labels.
#     """
#     mask_out = np.zeros_like(mask, dtype=mask.dtype)
#     next_available_label = np.max(mask) + 1  # Start new labels from the max existing label + 1

#     # Iterate through the provided label_info_list
#     for label_info in label_info_list:
#         # Extract the specific slice for this label
#         t_idx = label_info.frame
#         c_idx = label_info.channel
#         z_idx = label_info.z_plane
#         slice_ = mask[t_idx, c_idx, z_idx]


#         if label_info.npixels > max_sizes:
#             # remove very big labels
#             continue  # Skip this label
#         elif  label_info.npixels > max_size:
#             # Create binary mask for the current label
#             label_mask = (slice_ == label_info.label_id)


#             # Distance Transform
#             distance = distance_transform_edt(label_mask)

#             # Find local maxima
#             all_coords = peak_local_max(
#                 distance, 
#                 labels=label_mask,
#                 num_peaks=2,
#                 min_distance=10,  # Minimum distance between peaks 
#                 exclude_border=False
#             ) # Will return more than 2 peaks if there are more than 2 local maxima witht the same value
#             sorted_coords = sorted(all_coords, key=lambda c: distance[tuple(c)], reverse=True)
#             coords = sorted_coords[:2]  # Take the top 2 coordinates furthest apart


#             markers = np.zeros_like(label_mask, dtype=np.int32)
#             for i, coord in enumerate(coords):
#                 markers[coord[0], coord[1]] = next_available_label
#                 next_available_label += 1

#             # Apply watershed algorithm
#             new_labels = watershed(-distance, markers, mask=label_mask, connectivity=2)
#             # Update the new_mask and record the new label information
#             unique_new_labels = np.unique(new_labels)
#             for new_label in unique_new_labels:
#                 if new_label > 0:
#                     # Assign the new label in the new_mask
#                     mask_out[t_idx, c_idx, z_idx][new_labels == new_label] = new_label
        
#         else:
#             mask_out[t_idx, c_idx, z_idx][slice_ == label_info.label_id] = label_info.label_id
#     return mask_out, LabelInfo.from_mask(mask_out)


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

    # if len(nonempty_outlines) < len(outlines):
    #     print(f"Empty outlines found, saving {len(nonempty_outlines)} ImageJ ROIs to .zip archive.")

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

# -----------------------------------------------------------------------------
# Main processing functions
# -----------------------------------------------------------------------------


def process_file(
    input_path: Union[str, Path],
    output_name: Union[str, Path],
    *,  # Enforce keyword-only arguments
    channels: List[int],
    threshold_methods: List[str],
    median_filter_sizes: Optional[List[int]] = None,  # [-1] means no filter
    background_median_filter: Optional[List[int]] = None,  # [-1] means no background
    min_sizes: Optional[List[float]] = None,
    max_sizes: Optional[List[float]] = None,
    watershed_large_labels: Optional[List[int]] = None,  # [-1] means disabled
    remove_xy_edges: bool = True,
    remove_z_edges: bool = False,
    tmp_output_folder: Optional[Union[str, Path]] = None,
    yaml_file_extension: str = "_metadata.yaml",
    sys_exit_after_step: Optional[SysExitStep] = None,
) -> Tuple[Optional[Mask], Optional[ROIs], Optional[List[LabelInfo]]]:
    """Segment *input_path* and write results to *output_name*.

    Returns the final ``(mask, rois, labelinfo)`` tuple. Any step that raises an
    exception will be logged and cause an early return with the partial results
    obtained up to that point.

    Pylance/pyright strict‑mode compatibility.
    """

    # -------------------------------------------------------------------------
    # Normalise & validate inputs
    # -------------------------------------------------------------------------
    if not channels:
        raise ValueError("'channels' must contain at least one channel index.")
    if not threshold_methods:
        raise ValueError("'threshold_methods' must contain at least one method.")
    
    # Replace ``None`` with sentinel defaults.
    median_filter_sizes = median_filter_sizes or [-1]
    background_median_filter = background_median_filter or [-1]
    min_sizes = min_sizes or [-float('inf')]
    max_sizes = max_sizes or [float('inf')]
    watershed_large_labels = watershed_large_labels or [-1]
    
    # Length consistency checks.
    def _check_len(name: str, value: List[Any]) -> None:
        if len(value) not in (1, len(channels)):
            raise ValueError(
                f"{name} must have length 1 or the same length as 'channels' "
                f"({len(channels)}); received {len(value)} values."
            )

    _check_len("median_filter_sizes", median_filter_sizes)
    _check_len("background_median_filter", background_median_filter)
    _check_len("threshold_methods", threshold_methods)
    _check_len("min_sizes", min_sizes)
    _check_len("max_sizes", max_sizes)
    _check_len("watershed_large_labels", watershed_large_labels)

    # Expand length‑1 parameter lists so that the rest of the code can treat
    # them positionally per channel. This preserves the original behaviour of
    # "broadcasting" a single value.
    def _expand(value: List[Any]) -> List[Any]:
        return value * len(channels) if len(value) == 1 else value

    median_filter_sizes = _expand(median_filter_sizes)
    background_median_filter = _expand(background_median_filter)
    threshold_methods = _expand(threshold_methods)
    min_sizes = _expand(min_sizes)
    max_sizes = _expand(max_sizes)
    watershed_large_labels = _expand(watershed_large_labels)

    # Paths.
    input_path_p = Path(input_path).expanduser().resolve()
    output_path_p = Path(output_name).expanduser().resolve()
    tmp_dir_p = Path(tmp_output_folder).expanduser().resolve() if tmp_output_folder else None
    if tmp_dir_p:
        tmp_dir_p.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 0. Copy & update YAML metadata
    # -------------------------------------------------------------------------
    current_results: Tuple[Optional[Mask], Optional[ROIs], Optional[List[LabelInfo]]] = (
        None,
        None,
        None,
    )

    try:
        yaml_in = _metadata_path(input_path_p, yaml_file_extension)
        if yaml_in.exists():
            with yaml_in.open() as fh:
                metadata: Dict[str, Any] = yaml.safe_load(fh)
            metadata["Threshold segmentation"] = {
                "Channels": channels,
                "Median filter sizes": dict(zip(channels, median_filter_sizes)),
                "Threshold methods": dict(zip(channels, threshold_methods)),
                "Min size": min_sizes,
                "Max size": max_sizes,
                "Remove xy edges": remove_xy_edges,
                "Remove z edges": remove_z_edges,
            }
            yaml_out = _metadata_path(output_path_p, yaml_file_extension)
            with yaml_out.open("w") as fh:
                yaml.dump(metadata, fh)
    except Exception as e:
        return _safe_return("Error updating YAML metadata", e, current_results)

    # -------------------------------------------------------------------------
    # 1. Load image
    # -------------------------------------------------------------------------
    try:
        img = rp.load_bioio(str(input_path_p))  # External lib
    except Exception as e:
        return _safe_return("Error loading image", e, current_results)

    rois: Optional[ROIs] = None

    # Convenience closure to save intermediates
    def _save(mask_: Optional[Mask], label_: Optional[List[LabelInfo]], rois_: Optional[ROIs], path_: Optional[Path]) -> None:
        if path_ is None:
            return
        save_intermediate(
            mask=mask_,
            labelinfo=label_,
            rois=rois_,
            path=str(path_),
            physical_pixel_sizes=img.physical_pixel_sizes,
        )

    # -------------------------------------------------------------------------
    # STEP helpers
    # -------------------------------------------------------------------------
    def _load_cached(path_base: Path) -> Tuple[Optional[Mask], Optional[List[LabelInfo]]]:
        tif = path_base.with_suffix(".tif")
        if not tif.exists():
            return None, None
        mask_local = rp.load_bioio(str(tif)).data  # Load mask
        li_path = tif.with_name(f"{path_base.name}_labelinfo.yaml")
        li_local = LabelInfo.load(str(li_path)) if li_path.exists() else None  # Load label info
        return mask_local, li_local

    # Map of processing steps
    steps: List[Tuple[str, Callable[[Mask, Optional[List[LabelInfo]]], Tuple[Mask, Optional[List[LabelInfo]]]], bool]] = []  

    # 1. median filter
    def _median_filter(m: Mask, _: Optional[List[LabelInfo]]) -> Tuple[Mask, Optional[List[LabelInfo]]]:
        return apply_median_filter(m, sizes=median_filter_sizes, channels=channels), None

    steps.append(("median_filter", _median_filter, False))

    # 2. background subtraction
    def _background_subtract(m: Mask, _: Optional[List[LabelInfo]]) -> Tuple[Mask, Optional[List[LabelInfo]]]:
        return subtract_background_median_filter(m, background_median_filter), None
    steps.append(("background_subtract", _background_subtract, False))

    # 3. thresholding
    def _threshold(m: Mask, _: Optional[List[LabelInfo]]) -> Tuple[Mask, Optional[List[LabelInfo]]]:
        return apply_threshold(m, methods=threshold_methods, channels=channels)

    steps.append(("apply_threshold", _threshold, True))

    # 4. remove small / large labels
    def _remove_small_large(m: Mask, li: Optional[List[LabelInfo]]) -> Tuple[Mask, Optional[List[LabelInfo]]]:
        watershed_large_labels_bool: List[bool] = [ws > 0 for ws in watershed_large_labels]
        max_filters = [float('inf') if ws else max_sizes[i] for i, ws in enumerate(watershed_large_labels_bool)]
        return remove_small_or_large_labels(m, li, min_sizes=min_sizes, max_sizes=max_filters, channels=channels)

    steps.append(("remove_small_labels", _remove_small_large, True))

    # 5. remove edges
    def _remove_edges(m: Mask, li: Optional[List[LabelInfo]]) -> Tuple[Mask, Optional[List[LabelInfo]]]:
        return remove_on_edges(m, li, remove_xy_edges=remove_xy_edges, remove_z_edges=remove_z_edges)

    steps.append(("remove_edges", _remove_edges, True))

    # 6. fill holes
    def _fill_holes(m: Mask, _: Optional[List[LabelInfo]]) -> Tuple[Mask, Optional[List[LabelInfo]]]:
        return fill_holes_indexed(m), None

    steps.append(("fill_holes", _fill_holes, False))

    # 7. watershed (if applicable)
    if watershed_large_labels and max_sizes[0] < float('inf'):  # A check to ensure proper conditions
        def _watershed(m: Mask, li: Optional[List[LabelInfo]]) -> Tuple[Mask, Optional[List[LabelInfo]]]:
            return split_large_labels_with_watershed(m, li)  # Handle the actual logic
        steps.append(("watershed", _watershed, True))
    else:
        print("Warning: watershed step skipped (either disabled or max size conditions not met)")

    # -------------------------------------------------------------------------
    # Execute linear pipeline
    # -------------------------------------------------------------------------
    current_mask: Mask = img.data  # Start with raw image data
    current_labelinfo: Optional[List[LabelInfo]] = None

    for idx, (step_name, func, uses_li) in enumerate(steps, start=1):
        # print(f"Processing step {idx}/{len(steps)}: {step_name}")
        cache_base = _make_cache_path(input_path_p, channels, idx, step_name, tmp_dir_p)
        try:
            cached_mask, cached_li = (None, None) if cache_base is None else _load_cached(cache_base)
            if cached_mask is not None:
                current_mask, current_labelinfo = cached_mask, cached_li if uses_li else current_labelinfo
            else:
                current_mask, produced_li = func(current_mask, current_labelinfo)
                if produced_li is not None:
                    current_labelinfo = produced_li
                    # print(f"Produced {len(current_labelinfo)} labels in step '{step_name}'")
                _save(current_mask, current_labelinfo if uses_li else None, None, cache_base)
        except Exception as e:
            return _safe_return(f"Error during '{step_name}'", e, (current_mask, rois, current_labelinfo))

        _exit_if_requested(step_name, sys_exit_after_step)

    # -------------------------------------------------------------------------
    # 8. tracking (if applicable)
    if img.dims.T > 1:
        step_name = "tracking"
        cache_base = _make_cache_path(input_path_p, channels, len(steps) + 1, step_name, tmp_dir_p)
        try:
            if cache_base and cache_base.with_suffix(".tif").exists():
                current_mask = rp.load_bioio(str(cache_base.with_suffix(".tif"))).data  # Load cached mask
                current_labelinfo = LabelInfo.load(str(cache_base.with_name(f"{cache_base.name}_labelinfo.yaml")))  # Load cached label info
            else:
                from track_indexed_mask import track_labels_with_trackpy  # Local import
                df_tracked, current_mask = track_labels_with_trackpy(current_mask)  # Tracking processing
                current_labelinfo = LabelInfo.from_mask(current_mask)  # Convert mask to label info
                _save(current_mask, current_labelinfo, None, cache_base)
                if cache_base:
                    df_tracked.to_csv(str(cache_base) + ".tsv", sep="\t", index=False)  # Save tracking data

        except Exception as e:
            return _safe_return("Error during 'tracking'", e, (current_mask, rois, current_labelinfo))

        _exit_if_requested(step_name, sys_exit_after_step)

    # -------------------------------------------------------------------------
    # 9. generate ROIs
    step_name = "generate_rois"
    cache_base = _make_cache_path(input_path_p, channels, len(steps) + 2, step_name, tmp_dir_p)
    try:
        if cache_base and cache_base.exists():
            rois_raw = roiread(str(cache_base))  # Load cached ROIs
            rois = rois_raw if isinstance(rois_raw, list) else [rois_raw]
        else:
            if current_labelinfo is not None:
                rois = mask_to_rois(current_mask, current_labelinfo)  # Convert mask to ROIs
            else:
                rois = None
            _save(None, None, rois, cache_base)  # Save ROIs
    except Exception as e:
        return _safe_return("Error generating ROIs", e, (current_mask, rois, current_labelinfo))

    _exit_if_requested(step_name, sys_exit_after_step)

    # -------------------------------------------------------------------------
    # 10. final write
    try:
        save_intermediate(
            mask=current_mask,
            labelinfo=current_labelinfo,
            rois=rois,
            path=str(output_path_p),
            physical_pixel_sizes=img.physical_pixel_sizes,
        )
    except Exception as e:
        return _safe_return("Error saving final results", e, (current_mask, rois, current_labelinfo))

    return current_mask, rois, current_labelinfo


def process_folder(args: argparse.Namespace, parallel:bool) -> None:     
    # Find files to process
    files_to_process = rp.get_files_to_process(args.input_file_or_folder, ".tif", search_subfolders=False)

    # Make output folder
    os.makedirs(args.output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

    if parallel:  # Process each file in parallel
        # raise NotImplementedError
        
        Parallel(n_jobs=-1)(
            delayed(process_file)(
                input_path = input_file_path,
                output_name=os.path.join(args.output_folder, os.path.splitext(os.path.basename(input_file_path))[0] + "_mask"),  # Example output naming
                channels = args.channels,
                median_filter_sizes = args.median_filter_sizes,
                background_median_filter = args.background_median_filter_sizes,
                threshold_methods = args.threshold_methods,
                min_sizes = args.min_sizes,
                max_sizes = args.max_sizes,
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
            output_tif_file_name: str = os.path.join(args.output_folder, os.path.splitext(os.path.basename(input_file_path))[0] + "_mask")  # Example output naming
            # Process file
            try:

                process_file(
                    input_path = input_file_path,
                    output_name = output_tif_file_name,
                    channels = args.channels,
                    median_filter_sizes = args.median_filter_sizes,
                    threshold_methods = args.threshold_methods,
                    background_median_filter = args.background_median_filter_sizes,
                    min_sizes = args.min_sizes,
                    max_sizes = args.max_sizes,
                    watershed_large_labels = args.watershed_large_labels,
                    remove_xy_edges = args.remove_xy_edges,
                    remove_z_edges = args.remove_z_edges,
                    tmp_output_folder = args.tmp_output_folder,
                    yaml_file_extension = parsed_args.yaml_file_extension,
                    sys_exit_after_step = args.sys_exit_after_step
                    )
                  # Process each file

            except Exception as e:
                print(f"Error processing {input_file_path}: {e}")
                continue

def main(parsed_args: argparse.Namespace, parallel: bool = True) -> None:
    # -------------------------------------------------------------------------
    # Check if its a file or a folder and process accordingly
    # -----------------------------------

    # Check if the input is a file or a folder
    if os.path.isfile(parsed_args.input_file_or_folder):
        print(f"Processing single file: {parsed_args.input_file_or_folder}")
        output_file_name = os.path.splitext(os.path.basename(parsed_args.input_folder))[0] + "_mask"  # Example output naming
        output_file_path = os.path.join(parsed_args.output_folder, output_file_name)
        process_file(input_path = parsed_args.input_file_or_folder,
                 output_name = output_file_path,
                 channels = parsed_args.channels,
                 median_filter_sizes = parsed_args.median_filter_sizes,
                 background_median_filter= parsed_args.background_median_filter_sizes,
                 threshold_methods = parsed_args.threshold_methods,   
                 min_sizes= parsed_args.min_sizes, max_sizes = parsed_args.max_size, watershed_large_labels = parsed_args.watershed_large_labels,
                 remove_xy_edges = parsed_args.remove_xy_edges, remove_z_edges = parsed_args.remove_z_edges,
                 tmp_output_folder = parsed_args.mp_output_folder,
                 yaml_file_extension = parsed_args.yaml_file_extension,
                 sys_exit_after_step = parsed_args.sys_exit_after_step)
        
    elif os.path.isdir(parsed_args.input_file_or_folder):
        print(f"Processing folder: {parsed_args.input_file_or_folder}")
        process_folder(parsed_args, parallel)
        
    else:
        print("Error: The specified path is neither a file nor a folder.")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file-or-folder", type=str, help="Path to the file or folder to be processed.")
    parser.add_argument("--output-folder", type=str, help="Output folder for processed files")
    parser.add_argument("--channels", type=split_comma_separated_intstring, default=[0], help="List of channels to process (comma-separated, e.g., 0,1,2)")
    parser.add_argument("--median-filter-sizes", type=split_comma_separated_intstring, default=[10], help="Size(s) of the median filter (comma-separated, e.g., 10,20)")
    parser.add_argument("--background-median-filter-sizes", type=split_comma_separated_intstring, default=[-1], help="Size(s) of the background median filter (comma-separated, e.g., 50,100). If <= 0, background subtraction is skipped.")
    parser.add_argument("--threshold-methods", type=split_comma_separated_strstring, default=["li"], help="Thresholding method(s) (comma-separated, e.g., li,otsu)")  # Use nargs='*' to accept space-separated
    parser.add_argument("--min-sizes", type=split_comma_separated_intstring, default=[10_000], help="Minimum size for processing. If a list is provided, it must match the number of channels. If only one value is provided, it will be used for all channels.")
    parser.add_argument("--max-sizes", type=split_comma_separated_intstring, default=[55_000], help="Maximum size for processing. If a list is provided, it must match the number of channels. If only one value is provided, it will be used for all channels.")
    parser.add_argument("--watershed-large-labels",type=split_comma_separated_strstring, default=[-1], help="If True, large labels will be split using watershed. If a list is provided, it must match the number of channels. If only one value is provided, it will be used for all channels. Use 'True' or 'False' as strings.")
    parser.add_argument("--remove-xy-edges", action="store_true", help="Remove edges in XY")
    parser.add_argument("--remove-z-edges", action="store_true", help="Remove edges in Z")
    parser.add_argument("--tmp-output-folder", type=str, help="Save intermediate steps in tmp_output_folder")
    parser.add_argument("--no-parallel", action="store_true", help="Do not use parallel processing")
    parser.add_argument("--yaml-file-extension", type=str, default="_metadata.yaml", help="Extension relative to basename of input image name")
    parser.add_argument("--sys-exit-after-step", type=str, default=None, help="Exit after a specific step (e.g., median_filter, background_subtract, thresholding, remove_small_labels, remove_edges, fill_holes, watershed, tracking, generate_rois) for debugging purposes. If None, will process all steps.")
    parsed_args = parser.parse_args()

    # -------------------------------------------------------------------------
    # All processing of input arguments happens here in __name__ == "__main__" block
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Check if the input is correct -------------------------------------------
    parallel = parsed_args.no_parallel == False # inverse

    # -------------------------------------------------------------------------
    # Valid sys_exit_after_step input
    if parsed_args.sys_exit_after_step not in SysExitStepList:
        sys.exit(f"Error: --sys_exit_after_step must be one of {SysExitStepList}. You provided {parsed_args.sys_exit_after_step}.")
    
    # Check if channels are provided
    if not parsed_args.channels:
        raise ValueError("At least one channel must be specified in --channels.")

    # -------------------------------------------------------------------------
    # Convert single values to lists if necessary
    # TODO I started fixing this in the actual functoins, but it is not yet complete
    num_channels = len(parsed_args.channels)
    # if only one median filter size is provided use it for all channels
    parsed_args.median_filter_sizes = [
        parsed_args.median_filter_sizes[i] if i < len(parsed_args.median_filter_sizes) else parsed_args.median_filter_sizes[0]
        for i in range(num_channels)
    ]

    # print(parsed_args.background_median_filter_sizes)
    # if only one background median filter size is provided use it for all channels
    parsed_args.background_median_filter_sizes = [
        parsed_args.background_median_filter_sizes[i] if i < len(parsed_args.background_median_filter_sizes) else parsed_args.background_median_filter_sizes[0]
        for i in range(num_channels)
    ]
    
    # if only one median filter size is provided use it for all channels
    parsed_args.threshold_methods = [
        parsed_args.threshold_methods[i] if i < len(parsed_args.threshold_methods) else parsed_args.threshold_methods[0]
        for i in range(num_channels)
    ]

    # if only one min_size is provided use it for all channels
    parsed_args.min_sizes = [
        parsed_args.min_sizes[i] if i < len(parsed_args.min_sizes) else parsed_args.min_sizes[0]
        for i in range(num_channels)
    ]
    
    # if only one max_size is provided use it for all channels
    parsed_args.max_sizess = [
        parsed_args.max_sizes[i] if i < len(parsed_args.max_sizes) else parsed_args.max_sizes[0]
        for i in range(num_channels)
    ]

    parsed_args.watershed_large_labels = [ # Also changes from string to boolean values
        parsed_args.watershed_large_labels[i] == "True" 
        if i < len(parsed_args.watershed_large_labels) 
        else parsed_args.watershed_large_labels[0] == "True" 
        for i in range(num_channels)
    ]

    # print(f"Channels: {parsed_args.channels}")
    # Run
    main(parsed_args, parallel=parallel)

from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.segmentation import watershed

import bioimage_pipeline_utils as rp

Mask = np.ndarray


def split_comma_separated_intstring(value: str) -> List[int]:
    return list(map(int, value.split(',')))


def split_comma_separated_strstring(value: str) -> List[str]:
    return list(map(str, value.split(',')))


class LabelInfo:
    def __init__(self, label_id: int, x_center: float, y_center: float, npixels: int, frame: int, channel: int, z_plane: int):
        self.label_id = label_id
        self.x_center = x_center
        self.y_center = y_center
        self.npixels = npixels
        self.frame = frame
        self.channel = channel
        self.z_plane = z_plane

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label_id": int(self.label_id),
            "x_center": float(self.x_center),
            "y_center": float(self.y_center),
            "npixels": int(self.npixels),
            "frame": int(self.frame),
            "channel": int(self.channel),
            "z_plane": int(self.z_plane),
        }

    @staticmethod
    def save(label_info_list: List["LabelInfo"], filepath: str) -> None:
        data = [label.to_dict() for label in label_info_list]
        with open(filepath, "w") as f:
            yaml.dump(data, f)

    @staticmethod
    def load(filepath: str) -> List["LabelInfo"]:
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return [LabelInfo(**item) for item in data]

    @classmethod
    def from_mask(cls, mask: Mask) -> List["LabelInfo"]:
        t, c, z, y, x = mask.shape
        label_info_list: List[LabelInfo] = []
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
                            cls(
                                label_id=label_num,
                                x_center=float(x_center),
                                y_center=float(y_center),
                                npixels=npixels,
                                frame=frame,
                                channel=channel,
                                z_plane=z_plane,
                            )
                        )
        return label_info_list


def _metadata_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}")


def _filter_watershed_cuts(boundary_mask: np.ndarray, max_length: int) -> np.ndarray:
    if max_length <= 0:
        return boundary_mask

    labeled = label(boundary_mask, connectivity=2)
    if labeled.max() == 0:
        return boundary_mask

    sizes = np.bincount(labeled.ravel())
    keep_ids = np.where(sizes <= max_length)[0]
    keep_ids = keep_ids[keep_ids != 0]
    if keep_ids.size == 0:
        return np.zeros_like(boundary_mask, dtype=bool)

    return np.isin(labeled, keep_ids)


def split_large_labels_with_watershed(
    mask: np.ndarray,
    label_info_list: Optional[List[LabelInfo]],
    *,
    channels: Optional[List[int]] = None,
    frames: Optional[List[int]] = None,
    max_sizes: List[float] = [np.inf],
    min_peak_distance: int = 10,
    max_num_peaks: int = 2,
    max_watershed_cut_length: Optional[int] = None,
) -> Tuple[np.ndarray, List[LabelInfo]]:
    if mask.ndim != 5:
        raise ValueError("mask must be 5-D (T, C, Z, Y, X)")

    if label_info_list is None:
        label_info_list = LabelInfo.from_mask(mask)

    t, c, z, y, x = mask.shape
    frames = list(range(t)) if frames is None else frames
    channels = list(range(c)) if channels is None else channels

    if len(max_sizes) == 1:
        max_sizes *= len(channels)
    if len(max_sizes) != len(channels):
        raise ValueError("max_sizes must be length-1 or equal to len(channels)")

    ch_thresh = {ch: max_sizes[i] for i, ch in enumerate(channels)}

    mask_out = mask.copy()
    next_id = int(mask.max()) + 1

    for info in label_info_list:
        if info.channel not in channels or info.frame not in frames:
            continue

        thr = ch_thresh[info.channel]
        f, ch, zp = info.frame, info.channel, info.z_plane

        src_slice = mask[f, ch, zp]
        dst_slice = mask_out[f, ch, zp]

        if info.npixels <= thr:
            continue

        lbl_mask = src_slice == info.label_id
        if lbl_mask.sum() == 0:
            continue

        dist = distance_transform_edt(lbl_mask)
        if dist is None or not isinstance(dist, np.ndarray):
            print(f"Warning: Distance transform failed for label {info.label_id}. Skipping.")
            continue

        peak_xy = peak_local_max(
            dist,
            labels=lbl_mask,
            num_peaks=max_num_peaks,
            min_distance=min_peak_distance,
            exclude_border=False,
        )
        if peak_xy.shape[0] < 2:
            continue

        if peak_xy is not None and peak_xy.size > 0:
            peak_vals = dist[tuple(peak_xy.T)]
        else:
            peak_vals = np.array([])
        peak_xy = peak_xy[np.argsort(peak_vals)[::-1][:max_num_peaks]]

        markers = np.zeros_like(lbl_mask, dtype=np.int32)
        for yx in peak_xy:
            markers[tuple(yx)] = next_id
            next_id += 1

        split = watershed(-dist, markers=markers, mask=lbl_mask, connectivity=2, watershed_line=True)
        boundary_mask = (split == 0) & lbl_mask
        if max_watershed_cut_length is not None:
            boundary_mask = _filter_watershed_cuts(boundary_mask, max_watershed_cut_length)

        if not boundary_mask.any():
            continue

        dst_slice[lbl_mask] = 0
        cut_mask = lbl_mask.copy()
        cut_mask[boundary_mask] = 0
        new_labels = label(cut_mask, connectivity=2)
        if new_labels.max() <= 1:
            dst_slice[lbl_mask] = info.label_id
            continue

        for new_lbl in np.unique(new_labels):
            if new_lbl > 0:
                dst_slice[new_labels == new_lbl] = next_id
                next_id += 1

    return mask_out, LabelInfo.from_mask(mask_out)


def process_file(
    input_path: Union[str, Path],
    output_name: Union[str, Path],
    *,
    channels: List[int],
    max_sizes: List[float],
    watershed_large_labels: List[int],
    min_peak_distance: int,
    max_num_peaks: int,
    max_watershed_cut_length: Optional[int],
    yaml_file_extension: str,
) -> Tuple[Optional[Mask], Optional[List[LabelInfo]]]:
    if not channels:
        raise ValueError("'channels' must contain at least one channel index.")

    if len(max_sizes) not in (1, len(channels)):
        raise ValueError("max_sizes must have length 1 or match the number of channels.")
    if len(watershed_large_labels) not in (1, len(channels)):
        raise ValueError("watershed_large_labels must have length 1 or match the number of channels.")

    if len(max_sizes) == 1:
        max_sizes = max_sizes * len(channels)
    if len(watershed_large_labels) == 1:
        watershed_large_labels = watershed_large_labels * len(channels)

    input_path_p = Path(input_path).expanduser().resolve()
    output_path_p = Path(output_name).expanduser().resolve()

    try:
        img = rp.load_tczyx_image(str(input_path_p))
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None

    mask = img.data
    if mask.ndim != 5:
        raise ValueError("Input mask must be 5D (T, C, Z, Y, X)")

    label_info_list = LabelInfo.from_mask(mask)

    enabled_channels = [ch for ch, ws in zip(channels, watershed_large_labels) if ws > 0]
    enabled_max_sizes = [max_sizes[channels.index(ch)] for ch in enabled_channels]

    if enabled_channels:
        mask_out, label_info_out = split_large_labels_with_watershed(
            mask,
            label_info_list,
            channels=enabled_channels,
            max_sizes=enabled_max_sizes,
            min_peak_distance=min_peak_distance,
            max_num_peaks=max_num_peaks,
            max_watershed_cut_length=max_watershed_cut_length,
        )
    else:
        print("Watershed step disabled for all channels; writing input mask unchanged.")
        mask_out = mask
        label_info_out = label_info_list

    try:
        rp.save_tczyx_image(mask_out, str(output_path_p) + ".tif", physical_pixel_sizes=img.physical_pixel_sizes)
        LabelInfo.save(label_info_out, str(output_path_p) + "_labelinfo.yaml")
    except Exception as e:
        print(f"Error saving results: {e}")
        return mask_out, label_info_out

    try:
        yaml_in = _metadata_path(input_path_p, yaml_file_extension)
        if yaml_in.exists():
            with yaml_in.open() as fh:
                metadata: Dict[str, Any] = yaml.safe_load(fh)
            metadata["Watershed split"] = {
                "Channels": channels,
                "Max size": max_sizes,
                "Min peak distance": min_peak_distance,
                "Max num peaks": max_num_peaks,
                "Max watershed cut length": max_watershed_cut_length,
            }
            yaml_out = _metadata_path(output_path_p, yaml_file_extension)
            with yaml_out.open("w") as fh:
                yaml.dump(metadata, fh)
    except Exception as e:
        print(f"Error updating YAML metadata: {e}")

    return mask_out, label_info_out


def process_folder(args: argparse.Namespace) -> None:
    recursive = getattr(args, "search_subfolders", False) or False
    if os.path.isdir(args.input_search_pattern):
        ext = getattr(args, "extension", "") or ".tif"
        pattern = os.path.join(args.input_search_pattern, "**", f"*{ext}") if recursive else os.path.join(args.input_search_pattern, f"*{ext}")
        base_folder = args.input_search_pattern
    else:
        pattern = args.input_search_pattern
        base_folder = os.path.dirname(pattern) or "."

    files_to_process = rp.get_files_to_process2(pattern, recursive)
    if not files_to_process:
        print(f"No files found matching pattern: {pattern}")
        return

    os.makedirs(args.output_folder, exist_ok=True)

    def process_single_file(input_file_path: str) -> None:
        collapsed_name = rp.collapse_filename(input_file_path, base_folder)
        base_noext = os.path.splitext(collapsed_name)[0]
        output_tif_file_name = os.path.join(args.output_folder, base_noext + "_mask_ws")
        try:
            process_file(
                input_path=input_file_path,
                output_name=output_tif_file_name,
                channels=args.channels,
                max_sizes=args.max_sizes,
                watershed_large_labels=args.watershed_large_labels,
                min_peak_distance=args.min_peak_distance,
                max_num_peaks=args.max_num_peaks,
                max_watershed_cut_length=args.max_watershed_cut_length,
                yaml_file_extension=args.yaml_file_extension,
            )
        except Exception as e:
            print(f"Error processing {input_file_path}: {e}")

    if not args.no_parallel:
        from joblib import Parallel, delayed
        Parallel(n_jobs=-1)(delayed(process_single_file)(file) for file in files_to_process)
    else:
        for input_file_path in files_to_process:
            process_single_file(input_file_path)


def main(parsed_args: argparse.Namespace) -> None:
    if os.path.isfile(parsed_args.input_file_or_folder):
        output_file_name = os.path.splitext(os.path.basename(parsed_args.input_file_or_folder))[0] + "_mask_ws"
        output_file_path = os.path.join(parsed_args.output_folder, output_file_name)
        process_file(
            input_path=parsed_args.input_file_or_folder,
            output_name=output_file_path,
            channels=parsed_args.channels,
            max_sizes=parsed_args.max_sizes,
            watershed_large_labels=parsed_args.watershed_large_labels,
            min_peak_distance=parsed_args.min_peak_distance,
            max_num_peaks=parsed_args.max_num_peaks,
            max_watershed_cut_length=parsed_args.max_watershed_cut_length,
            yaml_file_extension=parsed_args.yaml_file_extension,
        )
    elif os.path.isdir(parsed_args.input_file_or_folder):
        process_folder(parsed_args)
    else:
        print("Error: The specified path is neither a file nor a folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Watershed split (default)
  environment: uv@3.11:segment-threshold-watershed-cut
  commands:
  - python
  - '%REPO%/standard_code/python/segment_threshold_watershed_cut.py'
  - --input-search-pattern: '%YAML%/input_data/**/*_mask.tif'
  - --output-folder: '%YAML%/output_data'
  - --channels: 0
  - --max-sizes: 55000
  - --watershed-large-labels: 1

- name: Watershed split (max cut length)
  environment: uv@3.11:segment-threshold-watershed-cut
  commands:
  - python
  - '%REPO%/standard_code/python/segment_threshold_watershed_cut.py'
  - --input-search-pattern: '%YAML%/input_data/**/*_mask.tif'
  - --output-folder: '%YAML%/output_data'
  - --channels: 0
  - --max-sizes: 55000
  - --watershed-large-labels: 1
  - --max-watershed-cut-length: 30
"""
    )
    parser.add_argument("--input-search-pattern", type=str, required=True, help="Glob pattern or folder path for input mask files.")
    parser.add_argument("--extension", type=str, default=".tif", help="File extension to search for when using a folder path.")
    parser.add_argument("--search-subfolders", action="store_true", help="Search recursively in subfolders when using a folder path.")
    parser.add_argument("--output-folder", type=str, required=True, help="Output folder for processed files.")
    parser.add_argument("--channels", type=split_comma_separated_intstring, default=[0], help="Channels to process (comma-separated).")
    parser.add_argument("--max-sizes", type=split_comma_separated_intstring, default=[55000], help="Maximum size per channel for watershed splitting.")
    parser.add_argument("--watershed-large-labels", type=split_comma_separated_intstring, default=[1], help="Enable watershed per channel (1 to enable, 0 to disable).")
    parser.add_argument("--min-peak-distance", type=int, default=10, help="Minimum distance between peaks for marker detection.")
    parser.add_argument("--max-num-peaks", type=int, default=2, help="Maximum number of peaks per label.")
    parser.add_argument("--max-watershed-cut-length", type=int, default=-1, help="Keep only watershed cut segments with length <= this value. Use -1 to disable.")
    parser.add_argument("--no-parallel", action="store_true", help="Do not use parallel processing.")
    parser.add_argument("--yaml-file-extension", type=str, default="_metadata.yaml", help="Extension relative to basename of input image name.")
    parsed_args = parser.parse_args()

    if parsed_args.max_watershed_cut_length is not None and parsed_args.max_watershed_cut_length <= 0:
        parsed_args.max_watershed_cut_length = None

    main(parsed_args)

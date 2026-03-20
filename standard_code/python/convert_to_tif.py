"""
Minimalistic image format converter using BioIO.
Converts various image formats to OME-TIFF with optional Z-projection.
Saves metadata and ROIs as YAML sidecars.

MIT License - BIPHUB, University of Oslo
"""

import os
import argparse
import logging
import re
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import tifffile
from tqdm import tqdm

from bioio import BioImage
import yaml

import bioimage_pipeline_utils as rp
import extract_metadata

# Module-level logger
logger = logging.getLogger(__name__)


def strip_tiff_suffix(path: str) -> str:
    """Return path without a trailing TIFF suffix (.ome.tif/.ome.tiff/.tif/.tiff)."""
    lower = path.lower()
    if lower.endswith(".ome.tif"):
        return path[:-8]
    if lower.endswith(".ome.tiff"):
        return path[:-9]
    if lower.endswith(".tif"):
        return path[:-4]
    if lower.endswith(".tiff"):
        return path[:-5]
    return os.path.splitext(path)[0]



def project_z(data: np.ndarray, method: str) -> np.ndarray:
    """
    Apply Z-projection to image data.
    
    Args:
        data: Input image array
        method: Projection method ('max', 'sum', 'mean', 'median', 'min', 'std')
    
    Returns:
        Projected image array
    """
    if method == "max":
        return np.max(data, axis=0)
    elif method == "sum":
        return np.sum(data, axis=0)
    elif method == "mean":
        return np.mean(data, axis=0)
    elif method == "median":
        return np.median(data, axis=0)
    elif method == "min":
        return np.min(data, axis=0)
    elif method == "std":
        return np.std(data, axis=0)
    else:
        logger.warning(f"Unknown projection method '{method}', using max")
        return np.max(data, axis=0)


def get_scene_dimensions(img: BioImage, scene_id: str) -> tuple[int, int]:
    """
    Get the physical dimensions (Y, X pixel count) of a scene.
    
    Args:
        img: BioImage object
        scene_id: Scene identifier
    
    Returns:
        Tuple of (height, width) in pixels
    """
    img.set_scene(scene_id)
    shape = img.shape  # TCZYX
    return (shape[-2], shape[-1])  # Y, X


def filter_scenes(
    scenes: tuple[str, ...] | list[str],
    img: BioImage,
    scene_filter: str = "largest",
    scene_filter_strings: Optional[list[str]] = None,
) -> list[str]:
    """
    Filter scenes according to the requested strategy.

    Args:
        scenes: All available scene names.
        img: BioImage object (used for dimension-based filters).
        scene_filter: Filtering mode.
            - ``all``      – keep every scene.
            - ``largest``  – keep scenes whose YX pixel count equals the maximum (default).
            - ``smallest`` – keep scenes whose YX pixel count equals the minimum.
            - ``includes`` – keep scenes whose name contains ANY of *scene_filter_strings*.
            - ``excludes`` – keep scenes whose name does NOT contain ANY of *scene_filter_strings*.
        scene_filter_strings: Required for ``includes`` / ``excludes`` modes.

    Returns:
        Filtered list of scene names.
    """
    if scene_filter == "all":
        return list(scenes)

    if scene_filter in ("largest", "smallest"):
        scene_dims = {}
        for scene_id in scenes:
            dims = get_scene_dimensions(img, scene_id)
            scene_dims[scene_id] = dims
            logger.info(f"Scene '{scene_id}': {dims[0]}x{dims[1]} pixels")

        pixel_counts = [dims[0] * dims[1] for dims in scene_dims.values()]
        if scene_filter == "largest":
            target = max(pixel_counts)
            label = "lower resolution pyramid level"
        else:
            target = min(pixel_counts)
            label = "higher resolution level"

        kept = []
        for scene_id, dims in scene_dims.items():
            if dims[0] * dims[1] == target:
                kept.append(scene_id)
            else:
                logger.info(f"Skipping scene '{scene_id}' - {label}")
        return kept

    if scene_filter == "includes":
        if not scene_filter_strings:
            logger.warning("scene_filter='includes' requires --scene-filter-strings; returning all scenes")
            return list(scenes)
        kept = [
            s for s in scenes
            if any(f in s for f in scene_filter_strings)
        ]
        skipped = [s for s in scenes if s not in kept]
        for s in skipped:
            logger.info(f"Skipping scene '{s}' - does not match includes filter")
        return kept

    if scene_filter == "excludes":
        if not scene_filter_strings:
            logger.warning("scene_filter='excludes' requires --scene-filter-strings; returning all scenes")
            return list(scenes)
        kept = [
            s for s in scenes
            if not any(f in s for f in scene_filter_strings)
        ]
        skipped = [s for s in scenes if s not in kept]
        for s in skipped:
            logger.info(f"Skipping scene '{s}' - matches excludes filter")
        return kept

    logger.warning(f"Unknown scene_filter '{scene_filter}', falling back to 'largest'")
    return filter_scenes(scenes, img, scene_filter="largest")


def extract_scene_timestamp(scene_name: str) -> Optional[str]:
    """Extract timestamp in HH:MM:SS format from a scene name."""
    match = re.search(r"\d{2}:\d{2}:\d{2}", scene_name)
    if match:
        return match.group(0)
    return None


def group_scenes_by_timestamp(scenes: list[str]) -> list[tuple[str, list[str]]]:
    """
    Group scene names by timestamp while preserving original scene order.

    Scenes without timestamp are kept as single-item groups.
    """
    groups: dict[str, list[str]] = {}
    ordered_keys: list[str] = []

    for scene_name in scenes:
        timestamp = extract_scene_timestamp(scene_name)
        key = timestamp if timestamp is not None else f"no_timestamp::{scene_name}"
        if key not in groups:
            groups[key] = []
            ordered_keys.append(key)
        groups[key].append(scene_name)

    return [(key, groups[key]) for key in ordered_keys]


def convert_single_file(
    input_path: str,
    output_path: str,
    projection_method: Optional[str] = None,
    save_metadata: bool = True,
    standard_tif: bool = False,
    split: bool = False,
    scene_filter: str = "largest",
    scene_filter_strings: Optional[list[str]] = None,
    scene_merge_channel: bool = False,
) -> bool:
    """
    Convert a single image file to OME-TIFF or standard TIFF.
    Handles multi-scene files by saving each scene separately.

    Args:
        input_path: Path to input image file
        output_path: Path to output TIFF file
        projection_method: Optional Z-projection method
        save_metadata: Whether to save metadata YAML sidecar
        standard_tif: If True, save as standard TIFF instead of OME-TIFF
        split: If True, save each T, C, Z slice as individual file in a folder
        scene_filter: Scene selection strategy ('all', 'largest', 'smallest',
            'includes', 'excludes').
        scene_filter_strings: Filter strings used with 'includes' / 'excludes'.
        scene_merge_channel: If True, group filtered scenes by timestamp in scene name
            and merge each timestamp-group into channel dimension (C).

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Converting: {os.path.basename(input_path)}")
        
        # Load image using BioIO with proper format detection
        img = rp.load_tczyx_image(input_path)
        
        # Check for multiple scenes
        scenes = img.scenes
        logger.info(f"Found {len(scenes)} scene(s)")

        # Filter scenes according to the requested strategy
        if len(scenes) > 1:
            logger.info(f"Applying scene filter: '{scene_filter}' (strings={scene_filter_strings})")
            scenes_to_process = filter_scenes(
                scenes=scenes,
                img=img,
                scene_filter=scene_filter,
                scene_filter_strings=scene_filter_strings,
            )
        else:
            scenes_to_process = list(scenes)

        if not scenes_to_process:
            logger.warning(f"Scene filter removed all scenes from {input_path} - skipping file")
            return False

        if scene_merge_channel and len(scenes_to_process) > 1:
            scene_groups = group_scenes_by_timestamp(scenes_to_process)
            logger.info(
                f"Merging scenes by timestamp: {len(scenes_to_process)} scenes -> "
                f"{len(scene_groups)} output group(s)"
            )
        else:
            scene_groups = [(scene_id, [scene_id]) for scene_id in scenes_to_process]

        logger.info(f"Processing {len(scene_groups)} output group(s)")

        # Process each output group
        for scene_idx, (group_key, scene_ids_in_group) in enumerate(scene_groups):
            merged_data = None
            merged_channel_names = []
            physical_pixel_sizes = None
            representative_scene_id = scene_ids_in_group[0]

            for scene_id in scene_ids_in_group:
                img.set_scene(scene_id)
            
                # Get data for this scene using Dask for better performance
                # Dask provides lazy loading and is ~38% faster for large files
                dask_data = img.dask_data
                data = dask_data.compute()

                logger.info(f"Loaded data shape: {data.shape}, ndim: {data.ndim} for scene '{scene_id}'")

                # Extract metadata for this scene BEFORE any RGB conversion
                # This includes physical pixel sizes, channel names, and other OME metadata
                channel_names = None
                original_channel_count = data.shape[1] if data.ndim >= 2 else 1  # Get C from TCZYX

                try:
                    if physical_pixel_sizes is None and hasattr(img, 'physical_pixel_sizes'):
                        physical_pixel_sizes = img.physical_pixel_sizes
                    if hasattr(img, 'channel_names'):
                        # Convert channel names to regular Python strings to avoid np.str_ issues
                        channel_names = [str(name) for name in img.channel_names]
                    logger.info(f"Extracted metadata - Pixel sizes: {physical_pixel_sizes}, Channels: {channel_names}")
                except Exception as e:
                    logger.warning(f"Could not extract metadata: {e}")

                # Handle RGB images (6D: TCZYXS where S=3 for RGB)
                if data.ndim == 6 and data.shape[-1] == 3:
                    logger.info("Detected RGB image - converting to separate channels")
                    # Reshape from (T, C, Z, Y, X, 3) to (T, C*3, Z, Y, X)
                    T, C, Z, Y, X, S = data.shape
                    # Split RGB into separate channels: R, G, B
                    data = data.transpose(0, 1, 5, 2, 3, 4)  # (T, C, S, Z, Y, X)
                    data = data.reshape(T, C * S, Z, Y, X)  # (T, C*3, Z, Y, X)
                    logger.info(f"Converted RGB to {C * S} channels, new shape: {data.shape}")

                    # Update channel names for RGB split
                    if channel_names and len(channel_names) == C:
                        # Expand channel names: each channel becomes R, G, B variants
                        new_channel_names = []
                        for ch_name in channel_names:
                            new_channel_names.extend([f"{ch_name}_R", f"{ch_name}_G", f"{ch_name}_B"])
                        channel_names = new_channel_names
                        logger.info(f"Updated channel names for RGB: {channel_names}")
                    elif C == 1:
                        # Simple case: single channel RGB becomes R, G, B
                        channel_names = ["Red", "Green", "Blue"]
                        logger.info(f"Set default RGB channel names: {channel_names}")

                if scene_merge_channel and len(scene_ids_in_group) > 1:
                    if merged_data is None:
                        merged_data = data
                    else:
                        if merged_data.shape[0] != data.shape[0] or merged_data.shape[2:] != data.shape[2:]:
                            logger.warning(
                                f"Skipping scene '{scene_id}' due to shape mismatch for merge: "
                                f"{data.shape} vs {merged_data.shape}"
                            )
                            continue
                        merged_data = np.concatenate([merged_data, data], axis=1)

                    base_channel = scene_id.split("/")[-1]
                    if original_channel_count == 1:
                        merged_channel_names.append(base_channel)
                    elif channel_names and len(channel_names) == original_channel_count:
                        merged_channel_names.extend(channel_names)
                    else:
                        merged_channel_names.extend(
                            [f"{base_channel}_C{idx}" for idx in range(original_channel_count)]
                        )
                else:
                    merged_data = data
                    if channel_names:
                        merged_channel_names = channel_names
                    else:
                        merged_channel_names = [f"Channel_{idx}" for idx in range(original_channel_count)]

            if merged_data is None:
                logger.warning(f"No scene data available for group '{group_key}', skipping")
                continue

            # Determine output path for this scene/group
            if len(scene_groups) > 1:
                lower_out = output_path.lower()
                if lower_out.endswith(".ome.tif"):
                    ext = ".ome.tif"
                elif lower_out.endswith(".ome.tiff"):
                    ext = ".ome.tiff"
                elif lower_out.endswith(".tiff"):
                    ext = ".tiff"
                else:
                    ext = ".tif"
                base = strip_tiff_suffix(output_path)
                scene_output_path = f"{base}_{scene_idx + 1}{ext}"
            else:
                scene_output_path = output_path

            logger.info(
                f"Processing group '{group_key}' with {len(scene_ids_in_group)} scene(s) -> "
                f"{os.path.basename(scene_output_path)}"
            )

            # Apply projection if requested
            if projection_method:
                logger.info(f"Applying {projection_method} projection")
                
                # Check if Z dimension exists and is > 1
                if merged_data.ndim >= 3:
                    # Project along Z axis (assuming axis 2 for standard TCZYX)
                    z_axis = None
                    try:
                        # Try to determine Z axis from dims
                        dim_order = img.dims.order
                        z_axis = dim_order.index('Z') if 'Z' in dim_order else 2
                    except:
                        z_axis = 2  # Default to axis 2
                    
                    if merged_data.shape[z_axis] > 1:
                        merged_data = np.apply_along_axis(
                            lambda x: project_z(x, projection_method),
                            z_axis,
                            merged_data
                        )
                        logger.info(f"After projection, data shape: {merged_data.shape}, ndim: {merged_data.ndim}")
            
            logger.info(
                f"Data shape before save: {merged_data.shape}, "
                f"ndim: {merged_data.ndim}, projection_method: {projection_method}"
            )
            
            # Save scene data with metadata preservation
            os.makedirs(os.path.dirname(scene_output_path), exist_ok=True)
            
            # Check if split mode is enabled
            if split:
                # Save each T, C, Z slice as individual file
                # Use same naming scheme as ij_bridge_bioformats.py: basename_Z#_C#.ome.tif
                # This allows NIS-Elements to auto-detect and merge files properly
                split_folder = strip_tiff_suffix(scene_output_path)
                os.makedirs(split_folder, exist_ok=True)
                
                # Get basename for files (without path and extension)
                basename = os.path.basename(strip_tiff_suffix(scene_output_path))
                
                logger.info(f"Split mode: Saving individual slices to {split_folder}")
                logger.info(f"Using basename: {basename}")
                
                # Handle both 5D (TCZYX) and 4D (TCYX after projection) data
                if merged_data.ndim == 5:
                    T, C, Z, Y, X = merged_data.shape
                elif merged_data.ndim == 4:
                    # After Z-projection, data is TCYX
                    T, C, Y, X = merged_data.shape
                    Z = 1
                    # Reshape to 5D for uniform processing
                    merged_data = merged_data[:, :, np.newaxis, :, :]
                else:
                    raise ValueError(f"Unexpected data dimensions: {merged_data.ndim}D (expected 4D or 5D)")
                
                logger.info(f"Scene {scene_idx}, Dimensions: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
                
                # Save files using NIS-Elements compatible naming: basename_Z#_C#.ome.tif
                # This matches what Bio-Formats Exporter creates and what NIS-Elements expects
                for z in range(Z):
                    for c in range(C):
                        # For multi-timepoint, we need T in filename too
                        if T > 1:
                            # Extended format for timepoints: basename_T#_Z#_C#.ome.tif
                            for t in range(T):
                                slice_data = merged_data[t, c, z, :, :]
                                slice_filename = f"{basename}_T{t}_Z{z}_C{c}.ome.tif"
                                slice_path = os.path.join(split_folder, slice_filename)
                                tifffile.imwrite(slice_path, slice_data, photometric='minisblack')
                        else:
                            # Standard format (matches Bio-Formats Exporter): basename_Z#_C#.ome.tif
                            slice_data = merged_data[0, c, z, :, :]
                            slice_filename = f"{basename}_Z{z}_C{c}.ome.tif"
                            slice_path = os.path.join(split_folder, slice_filename)
                            tifffile.imwrite(slice_path, slice_data, photometric='minisblack')
                
                logger.info(f"Saved {Z * C * T} individual slice files for scene {scene_idx}")
                logger.info(f"NIS-Elements should auto-detect and merge when opening: {basename}_Z0_C0.ome.tif")
                
            else:
                # Standard save mode (single file)
                # Build kwargs for saving with metadata
                save_kwargs = {}
                if physical_pixel_sizes is not None:
                    save_kwargs['physical_pixel_sizes'] = physical_pixel_sizes
                if merged_channel_names:
                    save_kwargs['channel_names'] = merged_channel_names
                
                # Add standard_tif flag for NIS-Elements compatibility
                if standard_tif:
                    save_kwargs['ome_tiff'] = False
                    logger.info("Saving as standard TIFF (NIS-Elements compatible)")
                
                # Save with metadata
                rp.save_tczyx_image(merged_data, scene_output_path, **save_kwargs)
                logger.info(f"Saved: {scene_output_path}")
            
            # Save metadata if requested
            if save_metadata:
                metadata_path = strip_tiff_suffix(scene_output_path) + "_metadata.yaml"
                try:
                    metadata = extract_metadata.get_all_metadata(input_path, output_file=None)

                    # Keep metadata channel information aligned with the actual saved output
                    if "Image metadata" in metadata:
                        image_meta = metadata["Image metadata"]
                        image_dims = image_meta.get("Image dimensions")
                        if isinstance(image_dims, dict):
                            if merged_data.ndim >= 2:
                                image_dims["C"] = int(merged_data.shape[1])

                        if merged_channel_names:
                            image_meta["Channels"] = [{"Name": str(name)} for name in merged_channel_names]
                    
                    # Add scene and conversion info
                    metadata["Convert to tif"] = {
                        "Scene_names": [str(s) for s in scene_ids_in_group],
                        "Merged_channel_names": merged_channel_names,
                        "Scene_merge_channel": scene_merge_channel,
                        "Scene_group_key": group_key,
                        "Scene_index": scene_idx + 1,
                        "Total_scenes_processed": len(scene_groups),
                        "Scene_filter": scene_filter,
                        "Scene_filter_strings": scene_filter_strings or [],
                    }
                    if projection_method:
                        metadata["Convert to tif"]["Projection"] = {"Method": projection_method}
                    
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        yaml.safe_dump(metadata, f, sort_keys=False)
                    logger.info(f"Saved metadata: {metadata_path}")
                    
                    # Generate NIS-Elements reassembly macro (only for split mode)
                    if split:
                        try:
                            import generate_nis_reassembly_macro
                            split_folder = strip_tiff_suffix(scene_output_path)
                            output_nd2 = split_folder + ".nd2"
                            macro_path = generate_nis_reassembly_macro.generate_macro(
                                split_folder=split_folder,
                                output_nd2=output_nd2,
                                metadata_yaml=metadata_path
                            )
                            logger.info(f"Generated NIS reassembly macro: {macro_path}")
                        except Exception as e:
                            logger.warning(f"Failed to generate NIS reassembly macro: {e}")
                            
                except Exception as e:
                    logger.warning(f"Failed to save metadata: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert {input_path}: {e}")
        return False


def process_files(
    input_pattern: str,
    output_folder: Optional[str] = None,
    projection_method: Optional[str] = None,
    collapse_delimiter: str = "__",
    no_parallel: bool = False,
    save_metadata: bool = True,
    output_extension: str = "",
    dry_run: bool = False,
    standard_tif: bool = False,
    split: bool = False,
    scene_filter: str = "largest",
    scene_filter_strings: Optional[list[str]] = None,
    scene_merge_channel: bool = False,
) -> None:
    """
    Process multiple files matching a pattern.

    Args:
        input_pattern: File search pattern (supports ** for recursive)
        output_folder: Output directory (default: input_dir + '_tif')
        projection_method: Optional Z-projection method
        collapse_delimiter: Delimiter for collapsing subfolder paths
        no_parallel: Disable parallel processing
        save_metadata: Whether to save metadata YAML sidecars
        output_extension: Additional extension to add before .ome.tif
        dry_run: Only print planned actions without executing
        standard_tif: If True, save as standard TIFF instead of OME-TIFF
        split: If True, save each T, C, Z slice as individual file in a folder
        scene_filter: Scene selection strategy ('all', 'largest', 'smallest',
            'includes', 'excludes').
        scene_filter_strings: Filter strings used with 'includes' / 'excludes'.
        scene_merge_channel: If True, group filtered scenes by timestamp and merge
            each group into channel dimension.
    """
    # Find files
    search_subfolders = '**' in input_pattern
    files = rp.get_files_to_process2(input_pattern, search_subfolders=search_subfolders)
    
    if not files:
        logger.error(f"No files found matching pattern: {input_pattern}")
        return
    
    logger.info(f"Found {len(files)} file(s) to process")
    
    # Determine base folder
    if '**' in input_pattern:
        base_folder = input_pattern.split('**')[0].rstrip('/\\')
        if not base_folder:
            base_folder = os.getcwd()
        base_folder = os.path.abspath(base_folder)
    else:
        base_folder = str(Path(files[0]).parent)
    
    # Determine output folder
    if output_folder is None:
        output_folder = base_folder + "_tif"
    
    logger.info(f"Output folder: {output_folder}")
    
    # Prepare file pairs
    file_pairs = []
    for src in files:
        collapsed = rp.collapse_filename(src, base_folder, collapse_delimiter)
        out_name = os.path.splitext(collapsed)[0] + output_extension + ".ome.tif"
        out_path = os.path.join(output_folder, out_name)
        file_pairs.append((src, out_path))
    
    # Dry run - just print plans
    if dry_run:
        print(f"[DRY RUN] Would process {len(file_pairs)} files")
        print(f"[DRY RUN] Output folder: {output_folder}")
        if projection_method:
            print(f"[DRY RUN] Projection method: {projection_method}")
        print(f"[DRY RUN] Scene filter: {scene_filter} (strings={scene_filter_strings})")
        print(f"[DRY RUN] Scene merge channel: {scene_merge_channel}")
        for src, dst in file_pairs:
            print(f"[DRY RUN] {src} -> {dst}")
        return

    # Process files
    if no_parallel or len(file_pairs) == 1:
        # Sequential processing
        for src, dst in file_pairs:
            convert_single_file(
                src, dst, projection_method, save_metadata, standard_tif, split,
                scene_filter, scene_filter_strings, scene_merge_channel,
            )
    else:
        # Parallel processing
        max_workers = min(os.cpu_count() or 4, len(file_pairs))
        logger.info(f"Processing with {max_workers} workers")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    convert_single_file, src, dst, projection_method, save_metadata,
                    standard_tif, split, scene_filter, scene_filter_strings,
                    scene_merge_channel,
                ): (src, dst)
                for src, dst in file_pairs
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files", unit="file"):
                src, dst = futures[future]
                try:
                    success = future.result()
                    if not success:
                        logger.error(f"Failed: {src}")
                except Exception as e:
                    logger.error(f"Exception processing {src}: {e}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Minimalistic image converter to OME-TIFF with optional Z-projection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Convert ND2 files to OME-TIFF (keep largest scene only)
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/convert_to_tif.py'
  - --input-search-pattern: '%YAML%/input/**/*.nd2'
  - --output-folder: '%YAML%/output'
  - --log-level: INFO

- name: Convert OBF files, keep only MLE scenes
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/convert_to_tif.py'
  - --input-search-pattern: '%YAML%/input/**/*.obf'
  - --output-folder: '%YAML%/output'
  - --scene-filter: includes
  - --scene-filter-strings: /MLE
    - --scene-merge-channel
  - --log-level: INFO

- name: Convert CZI files, exclude overview scenes
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/convert_to_tif.py'
  - --input-search-pattern: '%YAML%/input/**/*.czi'
  - --output-folder: '%YAML%/output'
  - --scene-filter: excludes
  - --scene-filter-strings: Overview
  - --log-level: INFO

- name: Convert LIF files, process all scenes with max projection
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/convert_to_tif.py'
  - --input-search-pattern: '%YAML%/input/**/*.lif'
  - --output-folder: '%YAML%/output'
  - --scene-filter: all
  - --projection-method: max
  - --log-level: INFO
        """
    )
    
    parser.add_argument(
        "--input-search-pattern",
        type=str,
        required=True,
        help="Input file pattern (supports wildcards, use '**' for recursive search)"
    )
    
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder (default: input_folder + '_tif')"
    )
    
    parser.add_argument(
        "--projection-method",
        type=str,
        default=None,
        choices=["max", "sum", "mean", "median", "min", "std"],
        help="Z-projection method (default: no projection)"
    )
    
    parser.add_argument(
        "--collapse-delimiter",
        type=str,
        default="__",
        help="Delimiter for collapsing subfolder paths (default: '__')"
    )
    
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing (process files sequentially)"
    )
    
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip saving metadata YAML sidecars"
    )
    
    parser.add_argument(
        "--output-file-name-extension",
        type=str,
        default="",
        help="Additional extension to add before .ome.tif"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without executing"
    )
    
    parser.add_argument(
        "--standard-tif",
        action="store_true",
        help="Save as standard TIFF instead of OME-TIFF (better NIS-Elements compatibility)"
    )
    
    parser.add_argument(
        "--split",
        action="store_true",
        help="Save each T, C, Z slice as individual file in a folder (maximum compatibility)"
    )
    
    parser.add_argument(
        "--scene-filter",
        type=str,
        default="largest",
        choices=["all", "largest", "smallest", "includes", "excludes"],
        help=(
            "Scene selection strategy for multi-scene files (default: largest). "
            "'all' keeps every scene. "
            "'largest'/'smallest' selects by YX pixel count. "
            "'includes' keeps scenes whose name contains any --scene-filter-strings value. "
            "'excludes' removes scenes whose name contains any --scene-filter-strings value."
        ),
    )

    parser.add_argument(
        "--scene-filter-strings",
        type=str,
        nargs="+",
        default=None,
        metavar="STRING",
        help=(
            "One or more substrings used with --scene-filter includes/excludes. "
            "Example: --scene-filter-strings /MLE  or  --scene-filter-strings Overview Tile"
        ),
    )

    parser.add_argument(
        "--scene-merge-channel",
        action="store_true",
        help=(
            "Group filtered scenes by HH:MM:SS timestamp in scene name and merge "
            "each timestamp-group into the channel dimension"
        ),
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit"
    )

    parser.add_argument('--log-level', type=str, default='WARNING',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                    help='Logging level (default: INFO)')
    


    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    if args.version:
        version_file = Path(__file__).parent.parent.parent / "VERSION"
        try:
            version = version_file.read_text(encoding='utf-8').strip()
        except Exception:
            version = "unknown"
        print(f"convert_to_tif_2.py version: {version}")
        return
    
    
    # Process files
    process_files(
        input_pattern=args.input_search_pattern,
        output_folder=args.output_folder,
        projection_method=args.projection_method,
        collapse_delimiter=args.collapse_delimiter,
        no_parallel=args.no_parallel,
        save_metadata=not args.no_metadata,
        output_extension=args.output_file_name_extension,
        dry_run=args.dry_run,
        standard_tif=args.standard_tif,
        split=args.split,
        scene_filter=args.scene_filter,
        scene_filter_strings=args.scene_filter_strings,
        scene_merge_channel=args.scene_merge_channel,
    )

if __name__ == "__main__":
    main()

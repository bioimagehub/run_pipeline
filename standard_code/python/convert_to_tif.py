"""
Minimalistic image format converter using BioIO.
Converts various image formats to OME-TIFF with optional Z-projection.
Saves metadata and ROIs as YAML sidecars.

MIT License - BIPHUB, University of Oslo
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import tifffile

from bioio import BioImage
import yaml

import bioimage_pipeline_utils as rp
import extract_metadata

# Module-level logger
logger = logging.getLogger(__name__)



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


def convert_single_file(
    input_path: str,
    output_path: str,
    projection_method: Optional[str] = None,
    save_metadata: bool = True,
    standard_tif: bool = False,
    split: bool = False
) -> bool:
    """
    Convert a single image file to OME-TIFF or standard TIFF.
    Handles multi-scene files by saving each scene separately.
    
    Args:
        input_path: Path to input image file
        output_path: Path to output TIFF file
        projection_method: Optional Z-projection method
        save_metadata: Whether to save metadata YAML sidecar
        standard_tif: If True, save as standard TIFF instead of OME-TIFF (better NIS-Elements compatibility)
        split: If True, save each T, C, Z slice as individual file in a folder
    
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
        
        # If multiple scenes, filter by physical dimensions
        scenes_to_process = []
        if len(scenes) > 1:
            # Get dimensions of all scenes
            scene_dims = {}
            for scene_id in scenes:
                dims = get_scene_dimensions(img, scene_id)
                scene_dims[scene_id] = dims
                logger.info(f"Scene '{scene_id}': {dims[0]}x{dims[1]} pixels")
            
            # Find the largest dimension (assuming this is the full resolution)
            max_pixels = max(dims[0] * dims[1] for dims in scene_dims.values())
            
            # Only keep scenes with the same pixel count as the largest
            for scene_id, dims in scene_dims.items():
                if dims[0] * dims[1] == max_pixels:
                    scenes_to_process.append(scene_id)
                else:
                    logger.info(f"Skipping scene '{scene_id}' - lower resolution pyramid level")
        else:
            scenes_to_process = scenes
        
        logger.info(f"Processing {len(scenes_to_process)} scene(s) at full resolution")
        
        # Process each scene
        for scene_idx, scene_id in enumerate(scenes_to_process):
            img.set_scene(scene_id)
            
            # Determine output path for this scene
            if len(scenes_to_process) > 1:
                base, ext = os.path.splitext(output_path)
                scene_output_path = f"{base}_{scene_idx + 1}{ext}"
            else:
                scene_output_path = output_path
            
            logger.info(f"Processing scene '{scene_id}' -> {os.path.basename(scene_output_path)}")
            
            # Get data for this scene using Dask for better performance
            # Dask provides lazy loading and is ~38% faster for large files
            dask_data = img.dask_data
            data = dask_data.compute()
            
            logger.info(f"Loaded data shape: {data.shape}, ndim: {data.ndim}")
            
            # Handle RGB images (6D: TCZYXS where S=3 for RGB)
            if data.ndim == 6 and data.shape[-1] == 3:
                logger.info("Detected RGB image - converting to separate channels")
                # Reshape from (T, C, Z, Y, X, 3) to (T, C*3, Z, Y, X)
                T, C, Z, Y, X, S = data.shape
                # Split RGB into separate channels: R, G, B
                data = data.transpose(0, 1, 5, 2, 3, 4)  # (T, C, S, Z, Y, X)
                data = data.reshape(T, C * S, Z, Y, X)  # (T, C*3, Z, Y, X)
                logger.info(f"Converted RGB to {C * S} channels, new shape: {data.shape}")
            
            # Extract metadata for this scene before projection
            # This includes physical pixel sizes, channel names, and other OME metadata
            physical_pixel_sizes = None
            channel_names = None
            try:
                if hasattr(img, 'physical_pixel_sizes'):
                    physical_pixel_sizes = img.physical_pixel_sizes
                if hasattr(img, 'channel_names'):
                    # Convert channel names to regular Python strings to avoid np.str_ issues
                    channel_names = [str(name) for name in img.channel_names]
                logger.info(f"Extracted metadata - Pixel sizes: {physical_pixel_sizes}, Channels: {channel_names}")
            except Exception as e:
                logger.warning(f"Could not extract metadata: {e}")
            
            # Apply projection if requested
            if projection_method:
                logger.info(f"Applying {projection_method} projection")
                
                # Check if Z dimension exists and is > 1
                if data.ndim >= 3:
                    # Project along Z axis (assuming axis 2 for standard TCZYX)
                    z_axis = None
                    try:
                        # Try to determine Z axis from dims
                        dim_order = img.dims.order
                        z_axis = dim_order.index('Z') if 'Z' in dim_order else 2
                    except:
                        z_axis = 2  # Default to axis 2
                    
                    if data.shape[z_axis] > 1:
                        data = np.apply_along_axis(
                            lambda x: project_z(x, projection_method),
                            z_axis,
                            data
                        )
                        logger.info(f"After projection, data shape: {data.shape}, ndim: {data.ndim}")
            
            logger.info(f"Data shape before save: {data.shape}, ndim: {data.ndim}, projection_method: {projection_method}")
            
            # Save scene data with metadata preservation
            os.makedirs(os.path.dirname(scene_output_path), exist_ok=True)
            
            # Check if split mode is enabled
            if split:
                # Save each T, C, Z slice as individual file
                # Use same naming scheme as ij_bridge_bioformats.py: basename_Z#_C#.ome.tif
                # This allows NIS-Elements to auto-detect and merge files properly
                split_folder = os.path.splitext(scene_output_path)[0]
                os.makedirs(split_folder, exist_ok=True)
                
                # Get basename for files (without path and extension)
                basename = os.path.splitext(os.path.basename(scene_output_path))[0]
                
                logger.info(f"Split mode: Saving individual slices to {split_folder}")
                logger.info(f"Using basename: {basename}")
                
                # Handle both 5D (TCZYX) and 4D (TCYX after projection) data
                if data.ndim == 5:
                    T, C, Z, Y, X = data.shape
                elif data.ndim == 4:
                    # After Z-projection, data is TCYX
                    T, C, Y, X = data.shape
                    Z = 1
                    # Reshape to 5D for uniform processing
                    data = data[:, :, np.newaxis, :, :]
                else:
                    raise ValueError(f"Unexpected data dimensions: {data.ndim}D (expected 4D or 5D)")
                
                logger.info(f"Scene {scene_idx}, Dimensions: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
                
                # Save files using NIS-Elements compatible naming: basename_Z#_C#.ome.tif
                # This matches what Bio-Formats Exporter creates and what NIS-Elements expects
                for z in range(Z):
                    for c in range(C):
                        # For multi-timepoint, we need T in filename too
                        if T > 1:
                            # Extended format for timepoints: basename_T#_Z#_C#.ome.tif
                            for t in range(T):
                                slice_data = data[t, c, z, :, :]
                                slice_filename = f"{basename}_T{t}_Z{z}_C{c}.ome.tif"
                                slice_path = os.path.join(split_folder, slice_filename)
                                tifffile.imwrite(slice_path, slice_data, photometric='minisblack')
                        else:
                            # Standard format (matches Bio-Formats Exporter): basename_Z#_C#.ome.tif
                            slice_data = data[0, c, z, :, :]
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
                if channel_names is not None:
                    save_kwargs['channel_names'] = channel_names
                
                # Add standard_tif flag for NIS-Elements compatibility
                if standard_tif:
                    save_kwargs['ome_tiff'] = False
                    logger.info("Saving as standard TIFF (NIS-Elements compatible)")
                
                # Save with metadata
                rp.save_tczyx_image(data, scene_output_path, **save_kwargs)
                logger.info(f"Saved: {scene_output_path}")
            
            # Save metadata if requested
            if save_metadata:
                metadata_path = os.path.splitext(scene_output_path)[0] + "_metadata.yaml"
                try:
                    metadata = extract_metadata.get_all_metadata(input_path, output_file=None)
                    
                    # Add scene and conversion info
                    metadata["Convert to tif"] = {
                        "Scene": scene_id,
                        "Scene_index": scene_idx + 1,
                        "Total_scenes_processed": len(scenes_to_process)
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
                            split_folder = os.path.splitext(scene_output_path)[0]
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
    split: bool = False
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
        output_extension: Additional extension to add before .tif
        dry_run: Only print planned actions without executing
        standard_tif: If True, save as standard TIFF instead of OME-TIFF
        split: If True, save each T, C, Z slice as individual file in a folder
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
        out_name = os.path.splitext(collapsed)[0] + output_extension + ".tif"
        out_path = os.path.join(output_folder, out_name)
        file_pairs.append((src, out_path))
    
    # Dry run - just print plans
    if dry_run:
        print(f"[DRY RUN] Would process {len(file_pairs)} files")
        print(f"[DRY RUN] Output folder: {output_folder}")
        if projection_method:
            print(f"[DRY RUN] Projection method: {projection_method}")
        for src, dst in file_pairs:
            print(f"[DRY RUN] {src} -> {dst}")
        return
    
    # Process files
    if no_parallel or len(file_pairs) == 1:
        # Sequential processing
        for src, dst in file_pairs:
            convert_single_file(src, dst, projection_method, save_metadata, standard_tif, split)
    else:
        # Parallel processing
        max_workers = min(os.cpu_count() or 4, len(file_pairs))
        logger.info(f"Processing with {max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(convert_single_file, src, dst, projection_method, save_metadata, standard_tif, split): (src, dst)
                for src, dst in file_pairs
            }
            
            for future in as_completed(futures):
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
Examples:
  # Convert all ND2 files in a folder
  python convert_to_tif_2.py --input-search-pattern "data/*.nd2"
  
  # Recursive search with max projection
  python convert_to_tif_2.py --input-search-pattern "data/**/*.czi" --projection-method max
  
  # Sequential processing (no parallel)
  python convert_to_tif_2.py --input-search-pattern "data/*.lif" --no-parallel
  
  # Dry run to preview actions
  python convert_to_tif_2.py --input-search-pattern "data/**/*.nd2" --dry-run
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
        help="Additional extension to add before .tif"
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
        split=args.split
    )

if __name__ == "__main__":
    main()

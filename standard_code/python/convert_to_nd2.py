"""
Minimalistic image format converter to ND2 using nd2 library.
Converts various image formats to ND2 with optional Z-projection.
Saves metadata as YAML sidecars.

MIT License - BIPHUB, University of Oslo
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from bioio import BioImage
import bioio_ome_tiff, bioio_tifffile, bioio_nd2, bioio_lif, bioio_czi, bioio_dv
import nd2
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
    save_metadata: bool = True
) -> bool:
    """
    Convert a single image file to ND2.
    Handles multi-scene files by saving each scene separately.
    
    Args:
        input_path: Path to input image file
        output_path: Path to output ND2 file
        projection_method: Optional Z-projection method
        save_metadata: Whether to save metadata YAML sidecar
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Converting: {os.path.basename(input_path)}")
        
        # Load image using BioIO
        img = BioImage(input_path)
        
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
            
            # Get data for this scene
            data = np.asarray(img.data)  # TCZYX format
            
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
                        # Project each T and C separately
                        projected_slices = []
                        for t in range(data.shape[0]):
                            t_slices = []
                            for c in range(data.shape[1]):
                                z_stack = data[t, c, :, :, :]  # ZYX
                                projected = project_z(z_stack, projection_method)
                                # Add back Z dimension with size 1
                                t_slices.append(projected[np.newaxis, ...])
                            projected_slices.append(np.stack(t_slices, axis=0))
                        data = np.stack(projected_slices, axis=0)  # Back to TCZYX with Z=1
            
            # Save as ND2
            os.makedirs(os.path.dirname(scene_output_path), exist_ok=True)
            
            # Try to extract physical pixel sizes from metadata
            try:
                # Get physical pixel size if available
                physical_pixel_sizes = img.physical_pixel_sizes  # (Z, Y, X) in microns
                
                # Prepare channel names
                try:
                    channel_names = [img.channel_names[i] if i < len(img.channel_names) else f"Channel_{i}" 
                                   for i in range(data.shape[1])]
                except:
                    channel_names = [f"Channel_{i}" for i in range(data.shape[1])]
                
                # Write ND2 file
                nd2.ND2File.write(
                    scene_output_path,
                    data,  # TCZYX array
                    channel_names=channel_names,
                    # Physical sizes (XYZ order for nd2 library)
                    pixel_size_um=(
                        physical_pixel_sizes.X if physical_pixel_sizes.X else 1.0,
                        physical_pixel_sizes.Y if physical_pixel_sizes.Y else 1.0,
                        physical_pixel_sizes.Z if physical_pixel_sizes.Z else 1.0
                    )
                )
                logger.info(f"Saved: {scene_output_path}")
                logger.info(f"  Physical pixel size: X={physical_pixel_sizes.X}, Y={physical_pixel_sizes.Y}, Z={physical_pixel_sizes.Z} µm")
                
            except Exception as e:
                logger.warning(f"Could not extract physical pixel sizes, using defaults: {e}")
                # Write with default pixel sizes
                try:
                    channel_names = [f"Channel_{i}" for i in range(data.shape[1])]
                except:
                    channel_names = ["Channel_0"]
                
                nd2.ND2File.write(
                    scene_output_path,
                    data,
                    channel_names=channel_names,
                    pixel_size_um=(1.0, 1.0, 1.0)  # Default 1µm per pixel
                )
                logger.info(f"Saved: {scene_output_path} (with default pixel sizes)")
            
            # Save metadata if requested
            if save_metadata:
                metadata_path = os.path.splitext(scene_output_path)[0] + "_metadata.yaml"
                try:
                    metadata = extract_metadata.get_all_metadata(input_path, output_file=None)
                    
                    # Add scene and conversion info
                    metadata["Convert to nd2"] = {
                        "Scene": scene_id,
                        "Scene_index": scene_idx + 1,
                        "Total_scenes_processed": len(scenes_to_process)
                    }
                    if projection_method:
                        metadata["Convert to nd2"]["Projection"] = {"Method": projection_method}
                    
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        yaml.safe_dump(metadata, f, sort_keys=False)
                    logger.info(f"Saved metadata: {metadata_path}")
                except Exception as e:
                    logger.warning(f"Failed to save metadata: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert {input_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def process_files(
    input_pattern: str,
    output_folder: Optional[str] = None,
    projection_method: Optional[str] = None,
    collapse_delimiter: str = "__",
    no_parallel: bool = False,
    save_metadata: bool = True,
    output_extension: str = "",
    dry_run: bool = False
) -> None:
    """
    Process multiple files matching a pattern.
    
    Args:
        input_pattern: File search pattern (supports ** for recursive)
        output_folder: Output directory (default: input_dir + '_nd2')
        projection_method: Optional Z-projection method
        collapse_delimiter: Delimiter for collapsing subfolder paths
        no_parallel: Disable parallel processing
        save_metadata: Whether to save metadata YAML sidecars
        output_extension: Additional extension to add before .nd2
        dry_run: Only print planned actions without executing
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
        output_folder = base_folder + "_nd2"
    
    logger.info(f"Output folder: {output_folder}")
    
    # Prepare file pairs
    file_pairs = []
    for src in files:
        collapsed = rp.collapse_filename(src, base_folder, collapse_delimiter)
        out_name = os.path.splitext(collapsed)[0] + output_extension + ".nd2"
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
            convert_single_file(src, dst, projection_method, save_metadata)
    else:
        # Parallel processing
        max_workers = min(os.cpu_count() or 4, len(file_pairs))
        logger.info(f"Processing with {max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(convert_single_file, src, dst, projection_method, save_metadata): (src, dst)
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
        description="Minimalistic image converter to ND2 format with optional Z-projection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all TIFF files to ND2
  python convert_to_nd2.py --input-search-pattern "data/*.tif"
  
  # Recursive search with max projection
  python convert_to_nd2.py --input-search-pattern "data/**/*.czi" --projection-method max
  
  # Sequential processing (no parallel)
  python convert_to_nd2.py --input-search-pattern "data/*.lif" --no-parallel
  
  # Dry run to preview actions
  python convert_to_nd2.py --input-search-pattern "data/**/*.tif" --dry-run
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
        help="Output folder (default: input_folder + '_nd2')"
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
        help="Additional extension to add before .nd2"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without executing"
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit"
    )
    
    args = parser.parse_args()
    
    if args.version:
        version_file = Path(__file__).parent.parent.parent / "VERSION"
        try:
            version = version_file.read_text(encoding='utf-8').strip()
        except Exception:
            version = "unknown"
        print(f"convert_to_nd2.py version: {version}")
        return
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Process files
    process_files(
        input_pattern=args.input_search_pattern,
        output_folder=args.output_folder,
        projection_method=args.projection_method,
        collapse_delimiter=args.collapse_delimiter,
        no_parallel=args.no_parallel,
        save_metadata=not args.no_metadata,
        output_extension=args.output_file_name_extension,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

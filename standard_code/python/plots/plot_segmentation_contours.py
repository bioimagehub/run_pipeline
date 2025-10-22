"""
Visualize segmentation contours from multiple samples overlaid on a single plot.

This module reads the segmentation masks saved by quantify_distance_heatmap.py
and creates overlay plots showing contours from multiple samples, colored by
experimental group.

Author: BIPHUB - Bioimage Informatics Hub, University of Oslo
License: MIT
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
from skimage import measure
import pandas as pd

# Local imports
try:
    from .. import bioimage_pipeline_utils as rp
except ImportError:
    # Fallback for when script is run directly (not as module)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import bioimage_pipeline_utils as rp

import yaml


def get_qc_file_path(image_path: str) -> str:
    """Get the QC YAML file path for an image."""
    return str(Path(image_path).with_suffix('')) + "_QC.yaml"


def load_qc_status(image_path: str, qc_key: str) -> Optional[str]:
    """
    Load QC status for a specific QC key from the QC file.
    
    Args:
        image_path: Path to the image file
        qc_key: QC key to check (e.g., 'nuc_segmentation', 'track_completeness')
        
    Returns:
        Status string ('passed' or 'failed') if found, None otherwise
    """
    qc_file = get_qc_file_path(image_path)
    if not os.path.exists(qc_file):
        return None
    
    try:
        with open(qc_file, 'r') as f:
            qc_data = yaml.safe_load(f)
        
        if qc_data and 'qc' in qc_data and qc_key in qc_data['qc']:
            return qc_data['qc'][qc_key].get('status')
    except Exception as e:
        logging.warning(f"Could not load QC file {qc_file}: {e}")
    
    return None


def filter_by_qc_status(grouped_files: Dict[str, Dict[str, str]], qc_key: str) -> Dict[str, Dict[str, str]]:
    """
    Filter grouped files to exclude those that failed QC.
    
    Args:
        grouped_files: Dict mapping basename to {'input': path, 'mask': path}
        qc_key: QC key to check for filtering
        
    Returns:
        Filtered dict with only passed or unreviewed files
    """
    filtered = {}
    failed_count = 0
    passed_count = 0
    unreviewed_count = 0
    
    for basename, files in grouped_files.items():
        input_path = files.get('input')
        if input_path is None:
            # No input file, skip
            continue
        
        qc_status = load_qc_status(input_path, qc_key)
        
        if qc_status == 'failed':
            failed_count += 1
            logging.info(f"Excluding {basename}: QC status = FAILED")
        else:
            filtered[basename] = files
            if qc_status == 'passed':
                passed_count += 1
            else:
                unreviewed_count += 1
    
    logging.info(f"QC Filtering Results (key: {qc_key}):")
    logging.info(f"  Passed: {passed_count}")
    logging.info(f"  Unreviewed: {unreviewed_count}")
    logging.info(f"  Failed (excluded): {failed_count}")
    logging.info(f"  Total included: {len(filtered)}/{len(grouped_files)}")
    
    return filtered


def extract_experimental_group(filename: str) -> str:
    """
    Extract experimental group from filename.
    
    Algorithm:
    1. Remove extension
    2. Remove trailing 3-digit number if present (e.g., '001', '011')
    3. Remove trailing underscore if present
    4. Split by underscore and take last part
    
    Examples:
        'SP20250625_PC_R3_WT011.tif' -> 'WT'
        'SP20250625_PC_R3_3SA_001.tif' -> '3SA'
        'SP20250625_PC_R3_L58R_010.tif' -> 'L58R'
        'SP20250625_PC_R3_LKO8_.tif' -> 'LKO8'
    
    Args:
        filename: Input filename (can be full path or just filename)
    
    Returns:
        Experimental group identifier
    """
    # Get just the filename without path
    base = Path(filename).stem  # Removes extension
    
    # Remove trailing 3-digit number if present (with or without underscore)
    # Handles: 'name001', 'name_001', 'name011'
    if len(base) >= 3 and base[-3:].isdigit():
        base = base[:-3]
    
    # Remove trailing underscore if present
    if base.endswith('_'):
        base = base[:-1]
    
    # Split by underscore and take last part
    parts = base.split('_')
    group = parts[-1] if parts else base
    
    return group


def find_mask_files(input_directory: str, pattern: str = '*_mask.tif') -> List[str]:
    """
    Find all mask TIFF files in the input directory.
    
    Args:
        input_directory: Directory to search for mask files
        pattern: Glob pattern for mask files (default: '*_mask.tif')
    
    Returns:
        List of absolute paths to mask files
    """
    input_path = Path(input_directory)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_directory}")
    
    mask_files = sorted(input_path.glob(pattern))
    mask_files = [str(f.absolute()) for f in mask_files]
    
    logging.info(f"Found {len(mask_files)} mask files in {input_directory}")
    
    return mask_files


def load_mask_with_metadata(mask_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Load a segmentation mask and extract metadata.
    
    Args:
        mask_path: Path to the mask TIFF file
    
    Returns:
        Tuple of (mask_array, metadata_dict) where:
        - mask_array: 2D binary mask (time, distance)
        - metadata_dict: Dictionary with filename, group, etc.
    """
    logging.info(f"Loading mask: {Path(mask_path).name}")
    
    # Load mask (saved as 5D TCZYX with shape (1, 1, 1, time, distance))
    img = rp.load_tczyx_image(mask_path)
    
    # Extract 2D array (T, C, Z, Y, X) -> (Y, X) where Y=time, X=distance
    # The mask was saved with: final_mask.T (transposed to time, distance) 
    # then np.flip(..., axis=0) which flipped the TIME axis
    # So we need to UN-FLIP the time axis (axis 0) to get correct orientation
    mask_2d = img.data[0, 0, 0, :, :]  # Get Y, X dimensions (flipped time, distance)
    mask_2d = np.flip(mask_2d, axis=0)  # UN-FLIP time axis to restore correct order
    
    # Convert to binary (values should be 0 or 255)
    mask_binary = (mask_2d > 127).astype(bool)
    
    # Extract metadata
    filename = Path(mask_path).stem.replace('_mask', '')
    group = extract_experimental_group(filename)
    
    metadata = {
        'filename': filename,
        'group': group,
        'path': mask_path,
        'shape': mask_binary.shape,
        'n_pixels': np.sum(mask_binary)
    }
    
    logging.info(f"  Group: {group}, Shape: {mask_binary.shape}, Pixels: {metadata['n_pixels']}")
    
    return mask_binary, metadata


def plot_contour_overlay(
    mask_files: List[str],
    output_path: Optional[str] = None,
    linewidth: float = 2.0,
    alpha: float = 0.5,
    colormap: str = 'tab10',
    force_show: bool = False
) -> None:
    """
    Create an overlay plot showing average masks per experimental group.
    
    Each group's average mask is displayed as a semi-transparent filled contour,
    colored by experimental group.
    
    Args:
        mask_files: List of paths to mask TIFF files
        output_path: Optional path to save the plot. If None, displays interactively.
        linewidth: Line width for contours (default: 2.0)
        alpha: Alpha transparency for average masks (default: 0.5)
        colormap: Matplotlib colormap for group colors (default: 'tab10')
        force_show: Display plot even when saving (default: False)
    """
    logging.info(f"Creating average mask overlay plot from {len(mask_files)} mask files")
    
    if len(mask_files) == 0:
        logging.error("No mask files provided")
        return
    
    # Load all masks and metadata
    masks_data = []
    for mask_path in mask_files:
        try:
            mask, metadata = load_mask_with_metadata(mask_path)
            masks_data.append({
                'mask': mask,
                'metadata': metadata
            })
        except Exception as e:
            logging.error(f"Error loading {mask_path}: {e}")
            continue
    
    if len(masks_data) == 0:
        logging.error("No valid masks loaded")
        return
    
    # Extract unique groups and assign colors
    all_groups = sorted(set(data['metadata']['group'] for data in masks_data))
    n_groups = len(all_groups)
    
    logging.info(f"Found {n_groups} experimental groups: {', '.join(all_groups)}")
    
    # Create color map
    if n_groups <= 10:
        cmap = plt.get_cmap(colormap)
        colors = [cmap(i) for i in range(n_groups)]
    else:
        # Use a continuous colormap if too many groups
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i / n_groups) for i in range(n_groups)]
    
    group_colors = dict(zip(all_groups, colors))
    
    # Determine canvas size (max dimensions across all masks)
    max_time = max(data['mask'].shape[0] for data in masks_data)
    max_distance = max(data['mask'].shape[1] for data in masks_data)
    
    logging.info(f"Canvas size: {max_time} timepoints × {max_distance} distance bins")
    
    # Group masks by experimental group
    masks_by_group = {group: [] for group in all_groups}
    for data in masks_data:
        group = data['metadata']['group']
        mask = data['mask']
        # Pad mask to canvas size if needed
        padded_mask = np.zeros((max_time, max_distance), dtype=float)
        padded_mask[:mask.shape[0], :mask.shape[1]] = mask.astype(float)
        masks_by_group[group].append(padded_mask)
    
    # Calculate average mask per group
    average_masks = {}
    for group, masks in masks_by_group.items():
        # Stack all masks and calculate mean
        stacked = np.stack(masks, axis=0)
        avg_mask = np.mean(stacked, axis=0)
        average_masks[group] = avg_mask
        n_samples = len(masks)
        logging.info(f"  Group '{group}': {n_samples} samples, avg mask range: [{np.min(avg_mask):.3f}, {np.max(avg_mask):.3f}]")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot borders/outlines for each group (not filled)
    # This shows the extent of the signal without blurring from averaging
    for group in all_groups:
        avg_mask = average_masks[group]
        color = group_colors[group]
        n_samples = len(masks_by_group[group])
        
        # Create binary mask showing where signal appears
        # Use a threshold to determine "signal present"
        # Lower threshold (e.g., 0.1) = show if ANY sample has signal (union)
        # Higher threshold (e.g., 0.5) = show if MOST samples have signal (consensus)
        threshold = 0.1  # Show if 10% or more of samples have signal at this location
        binary_mask = avg_mask >= threshold
        
        # Create coordinate grids for contour plotting
        x_coords = np.linspace(0, max_time, avg_mask.shape[0])
        y_coords = np.linspace(0, max_distance, avg_mask.shape[1])
        
        # Plot filled contour showing the signal region
        ax.contourf(x_coords, y_coords, binary_mask.T.astype(float), 
                   levels=[0.5, 1.5],  # Binary: 0 or 1
                   colors=[color], 
                   alpha=alpha)
        
        # Plot border/outline around the signal region
        ax.contour(x_coords, y_coords, binary_mask.T.astype(float), 
                  levels=[0.5],
                  colors=[color], 
                  linewidths=linewidth,
                  alpha=1.0)
    
    # Create legend manually with colored patches
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=group_colors[group], alpha=alpha, label=f"{group} (n={len(masks_by_group[group])})") 
                      for group in all_groups]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    # Set axis limits
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, max_distance)
    
    # Labels and title
    ax.set_xlabel('Timepoint', fontweight='bold', fontsize=12)
    ax.set_ylabel('Distance (bins)', fontweight='bold', fontsize=12)
    ax.set_title('Average Segmentation Masks by Experimental Group', 
                fontweight='bold', fontsize=14)
    
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    
    # Add summary text
    summary_text = f"Total samples: {len(masks_data)}\nGroups: {n_groups}\nShowing: Average masks per group\nTransparency: α={alpha:.2f}"
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved contour overlay plot to: {output_path}")
        
        if force_show:
            plt.show()
        else:
            plt.close()
    else:
        plt.show()


def main():
    """
    Command-line interface for plotting segmentation contour overlays.
    """
    parser = argparse.ArgumentParser(
        description='Plot segmentation contours from multiple samples, colored by experimental group. Filters out QC-failed images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create overlay plot with QC filtering
  python plot_segmentation_contours.py --input-search-pattern "./input/*.tif" --mask-search-pattern "./output/*_mask.tif" --qc-key nuc_segmentation -o overlay.png
  
  # Customize appearance
  python plot_segmentation_contours.py --input-search-pattern "./input/*.tif" --mask-search-pattern "./masks/*_mask.tif" --qc-key track_completeness -o overlay.png --linewidth 3.0 --alpha 0.9
  
  # Display interactively instead of saving
  python plot_segmentation_contours.py --input-search-pattern "./input/*.tif" --mask-search-pattern "./masks/*_mask.tif" --qc-key nuc_segmentation --force-show
  
  # Use different mode selection
  python plot_segmentation_contours.py --input-search-pattern "./input/*.tif" --mask-search-pattern "./masks/*_mask.tif" --qc-key nuc_segmentation --mode group1 -o overlay.png
        """
    )
    
    parser.add_argument('--input-search-pattern', type=str, required=True,
                       help='Glob pattern for input images (e.g., "./input/*.tif")')
    parser.add_argument('--mask-search-pattern', type=str, required=True,
                       help='Glob pattern for mask images (e.g., "./output/*_mask.tif")')
    parser.add_argument('--qc-key', type=str, required=True,
                       help='QC key to use for filtering (e.g., "nuc_segmentation", "track_completeness"). Images with failed status will be excluded.')
    parser.add_argument('-o', '--output', default=None,
                       help='Output path for the plot (PNG). If not specified, displays interactively.')
    parser.add_argument('--search-subfolders', action='store_true',
                       help='Enable recursive search for files')
    parser.add_argument('--mode', type=str, default='all',
                       help='File selection mode: "all" = all files, "examples[N]" = first, middle, last N files, '
                            '"first[N]" = first N files, "random[N]" = N random files, '
                            '"group[N]" = N samples per experimental group (default: all)')
    parser.add_argument('--linewidth', type=float, default=2.0,
                       help='Line width for contours (default: 2.0)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Alpha transparency for average masks (default: 0.5)')
    parser.add_argument('--colormap', default='tab10',
                       help='Matplotlib colormap for group colors (default: tab10)')
    parser.add_argument('--force-show', action='store_true',
                       help='Display plot interactively even when saving')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.info("=" * 80)
    logging.info("Segmentation Contour Overlay Plotter with QC Filtering")
    logging.info("=" * 80)
    
    # Use get_grouped_files_to_process to map input files to masks
    search_patterns = {
        'input': args.input_search_pattern,
        'mask': args.mask_search_pattern,
    }
    
    logging.info(f"Searching for files with patterns:")
    logging.info(f"  Input: {args.input_search_pattern}")
    logging.info(f"  Mask:  {args.mask_search_pattern}")
    logging.info(f"  QC Key: {args.qc_key}")
    
    try:
        grouped_files = rp.get_grouped_files_to_process(search_patterns, args.search_subfolders)
    except Exception as e:
        logging.error(f"Error finding files: {e}")
        return 1
    
    if not grouped_files:
        logging.error(f"No files found matching patterns")
        return 1
    
    logging.info(f"Found {len(grouped_files)} input files")
    
    # Count how many have masks
    files_with_masks = sum(1 for files in grouped_files.values() if 'mask' in files)
    files_without_masks = len(grouped_files) - files_with_masks
    logging.info(f"  {files_with_masks} with masks, {files_without_masks} without masks")
    
    # Filter by QC status (exclude failed)
    grouped_files = filter_by_qc_status(grouped_files, args.qc_key)
    
    if not grouped_files:
        logging.error(f"No files remaining after QC filtering")
        return 1
    
    logging.info(f"After QC filtering: {len(grouped_files)} files")
    
    # Extract mask files only (ignore those without masks)
    mask_files = []
    for basename, files in grouped_files.items():
        if 'mask' in files:
            mask_files.append(files['mask'])
    
    if len(mask_files) == 0:
        logging.error(f"No mask files found after filtering")
        return 1
    
    logging.info(f"Found {len(mask_files)} mask files to process")
    
    # Parse mode for file selection
    import re
    mode = str(args.mode).lower()  # Convert to string in case YAML passes boolean
    mode_match = re.match(r'(all|examples|first|random|group)(\d+)?$', mode)
    if not mode_match:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'all', 'examples[N]', 'first[N]', 'random[N]', or 'group[N]'")
    
    mode_type = mode_match.group(1)
    mode_count = int(mode_match.group(2)) if mode_match.group(2) else 1
    
    # Load metadata for all masks to extract groups
    mask_metadata = []
    for mask_file in mask_files:
        try:
            _, metadata = load_mask_with_metadata(mask_file)
            mask_metadata.append({'path': mask_file, 'group': metadata['group']})
        except Exception as e:
            logging.warning(f"Skipping {mask_file}: {e}")
            continue
    
    if len(mask_metadata) == 0:
        logging.error("No valid mask files could be loaded")
        return 1
    
    # Select files based on mode
    selected_files = []
    
    if mode_type == 'group':
        # Group mode: select N samples per experimental group
        from collections import defaultdict
        import random
        
        groups = defaultdict(list)
        for item in mask_metadata:
            groups[item['group']].append(item['path'])
        
        logging.info(f"Found {len(groups)} experimental groups")
        
        for group_name, group_files in sorted(groups.items()):
            n_available = len(group_files)
            n_select = min(mode_count, n_available)
            
            # Select evenly spaced samples from the group
            if n_available <= mode_count:
                selected = group_files
            else:
                # Use 'examples' logic: first, middle(s), last
                indices = []
                if n_select == 1:
                    indices = [n_available // 2]
                elif n_select == 2:
                    indices = [0, n_available - 1]
                else:
                    indices = [0]
                    for i in range(1, n_select - 1):
                        idx = int(i * n_available / (n_select - 1))
                        indices.append(idx)
                    indices.append(n_available - 1)
                selected = [group_files[i] for i in indices]
            
            selected_files.extend(selected)
            logging.info(f"  Group '{group_name}': selected {len(selected)}/{n_available} samples")
    
    else:
        # Standard selection modes (all, examples, first, random)
        def select_files(file_list, mode_type, mode_count):
            n = len(file_list)
            if mode_type == 'all':
                return file_list
            elif mode_type == 'examples':
                if n <= mode_count:
                    return file_list
                indices = []
                if mode_count == 1:
                    indices = [n // 2]
                elif mode_count == 2:
                    indices = [0, n - 1]
                else:
                    indices = [0]
                    for i in range(1, mode_count - 1):
                        idx = int(i * n / (mode_count - 1))
                        indices.append(idx)
                    indices.append(n - 1)
                return [file_list[i] for i in indices]
            elif mode_type == 'first':
                count = min(mode_count, n)
                return file_list[:count]
            elif mode_type == 'random':
                import random
                count = min(mode_count, n)
                return random.sample(file_list, count)
            return file_list
        
        all_files = [item['path'] for item in mask_metadata]
        selected_files = select_files(all_files, mode_type, mode_count)
    
    logging.info(f"Selected {len(selected_files)} mask files for visualization (mode: {args.mode})")
    
    # Create overlay plot
    try:
        plot_contour_overlay(
            mask_files=selected_files,
            output_path=args.output,
            linewidth=args.linewidth,
            alpha=args.alpha,
            colormap=args.colormap,
            force_show=args.force_show
        )
    except Exception as e:
        logging.error(f"Error creating contour overlay plot: {e}", exc_info=True)
        return 1
    
    logging.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

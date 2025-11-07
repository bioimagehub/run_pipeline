"""
QC Tracking - Quality Control for Tracked Masks

Creates a heatmap showing the number of masks per timepoint for all images.
Automatically detects incomplete tracks and marks them as failed in QC files.
Colors failed QC files in red for easy identification.

Author: BIPHUB - Bioimage Informatics Hub, University of Oslo
License: MIT
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
import pandas as pd
import yaml

# Local imports
try:
    from .. import bioimage_pipeline_utils as rp
except ImportError:
    # Fallback for when script is run directly (not as module)
    import sys
    import os
    # Go up to standard_code/python directory to find bioimage_pipeline_utils
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import bioimage_pipeline_utils as rp


def get_qc_file_path(image_path: str) -> str:
    """Get the QC YAML file path for an image."""
    return str(Path(image_path).with_suffix('')) + "_QC.yaml"


def load_existing_qc(image_path: str, qc_key: str) -> Optional[Dict]:
    """
    Load existing QC data if available for the specified qc_key.
    
    Args:
        image_path: Path to the mask file
        qc_key: QC key to check for (e.g., 'track_completeness')
        
    Returns:
        QC data dict if found, None otherwise
    """
    qc_file = get_qc_file_path(image_path)
    if os.path.exists(qc_file):
        try:
            with open(qc_file, 'r') as f:
                qc_data = yaml.safe_load(f)
            
            # Check if this specific qc_key exists
            if qc_data and 'qc' in qc_data and qc_key in qc_data['qc']:
                return qc_data['qc'][qc_key]
        except Exception as e:
            logging.warning(f"Could not load existing QC file {qc_file}: {e}")
    
    return None


def save_qc_result(image_path: str, qc_key: str, status: str, comment: str, 
                   mask_search_pattern: str) -> None:
    """
    Save QC result to a YAML file next to the image.
    Preserves existing QC data for other keys.
    
    Args:
        image_path: Path to the mask file
        qc_key: QC key (e.g., 'track_completeness')
        status: 'passed' or 'failed'
        comment: Detailed comment about the QC result
        mask_search_pattern: Original mask search pattern for documentation
    """
    qc_file = get_qc_file_path(image_path)
    
    # Load existing QC data if it exists
    existing_qc_data = {}
    if os.path.exists(qc_file):
        try:
            with open(qc_file, 'r') as f:
                existing_qc_data = yaml.safe_load(f) or {}
        except Exception as e:
            logging.warning(f"Could not load existing QC file {qc_file}: {e}")
    
    # Initialize qc section if it doesn't exist
    if 'qc' not in existing_qc_data:
        existing_qc_data['qc'] = {}
    
    # Update or add the specific qc_key
    existing_qc_data['qc'][qc_key] = {
        "status": status,
        "comment": comment,
        "mask_search_pattern": mask_search_pattern
    }
    
    try:
        with open(qc_file, "w") as f:
            yaml.dump(existing_qc_data, f, default_flow_style=False, sort_keys=False)
        logging.info(f"QC result saved to {qc_file}: {status.upper()}")
    except Exception as e:
        logging.error(f"Failed to save QC file {qc_file}: {e}")


def check_track_completeness(mask_path: Optional[str], qc_key: str, 
                            mask_search_pattern: str, basename: str) -> Tuple[str, bool, str]:
    """
    Check if all tracked objects appear in every timepoint.
    Also fails if there are zero objects in all timepoints.
    
    Args:
        mask_path: Path to the mask file (None if mask is missing)
        qc_key: QC key to use for saving results
        mask_search_pattern: Search pattern for documentation
        basename: Base filename for identification
        
    Returns:
        Tuple of (filename, is_complete, failure_reason)
    """
    # Handle missing mask file
    if mask_path is None:
        failure_reason = "Mask file not found"
        logging.warning(f"{basename}: FAILED - {failure_reason}")
        return basename, False, failure_reason
    
    filename = Path(mask_path).stem
    
    # Check if already QC'd and status is failed
    existing_qc = load_existing_qc(mask_path, qc_key)
    if existing_qc and existing_qc.get('status') == 'failed':
        logging.info(f"{filename}: Already marked as FAILED in QC")
        return filename, False, existing_qc.get('comment', 'Previously failed')
    
    try:
        mask = rp.load_tczyx_image(mask_path)
        T, C, Z, Y, X = mask.shape
        
        # Collect all labels across all timepoints
        all_labels = set()
        labels_per_timepoint = []
        
        for t in range(T):
            timepoint_data = mask.data[t, 0, :, :, :]
            unique_labels = np.unique(timepoint_data)
            non_zero_labels = set(unique_labels[unique_labels > 0])
            labels_per_timepoint.append(non_zero_labels)
            all_labels.update(non_zero_labels)
        
        # Check if there are zero objects in all timepoints
        if len(all_labels) == 0:
            failure_reason = f"No objects detected in any timepoint (empty mask across all {T} timepoints)"
            save_qc_result(mask_path, qc_key, 'failed', failure_reason, mask_search_pattern)
            logging.warning(f"{filename}: FAILED - {failure_reason}")
            return filename, False, failure_reason
        
        # Check if each label appears in every timepoint
        incomplete_labels = []
        for label in sorted(all_labels):
            missing_timepoints = []
            for t, labels_at_t in enumerate(labels_per_timepoint):
                if label not in labels_at_t:
                    missing_timepoints.append(t)
            
            if missing_timepoints:
                incomplete_labels.append((int(label), missing_timepoints))
        
        # Determine if tracking is complete
        if incomplete_labels:
            # Build detailed failure message
            failure_details = []
            for label, missing_t in incomplete_labels[:5]:  # Limit to first 5 for brevity
                failure_details.append(f"Object {label} missing at T={missing_t}")
            
            if len(incomplete_labels) > 5:
                failure_details.append(f"... and {len(incomplete_labels) - 5} more incomplete tracks")
            
            failure_reason = f"Incomplete tracks detected: {'; '.join(failure_details)}"
            
            # Save as failed if not already QC'd or if status was passed
            if not existing_qc or existing_qc.get('status') == 'passed':
                save_qc_result(mask_path, qc_key, 'failed', failure_reason, mask_search_pattern)
                logging.warning(f"{filename}: FAILED - {failure_reason}")
            
            return filename, False, failure_reason
        
        else:
            # All tracks are complete
            success_msg = f"All {len(all_labels)} tracks complete across {T} timepoints"
            
            # Save as passed if no QC exists yet
            if not existing_qc:
                save_qc_result(mask_path, qc_key, 'passed', success_msg, mask_search_pattern)
                logging.info(f"{filename}: PASSED - {success_msg}")
            
            return filename, True, success_msg
        
    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        logging.error(f"{filename}: {error_msg}")
        save_qc_result(mask_path, qc_key, 'failed', error_msg, mask_search_pattern)
        return filename, False, error_msg


def count_masks_in_image(mask_path: Optional[str], basename: str) -> Tuple[str, List[int], List[int]]:
    """
    Count the number of unique masks (labels) and find the max object size in each timepoint of a mask file.
    
    Args:
        mask_path: Path to the mask image (None if mask is missing)
        basename: Base filename for identification
        
    Returns:
        Tuple of (filename, list of mask counts per timepoint, list of max object sizes per timepoint)
    """
    if mask_path is None:
        logging.warning(f"Mask file missing for: {basename}")
        return basename, [], []
    
    logging.info(f"Processing: {Path(mask_path).name}")
    
    try:
        mask = rp.load_tczyx_image(mask_path)
        T, C, Z, Y, X = mask.shape
        
        mask_counts = []
        max_object_sizes = []
        for t in range(T):
            # Get all unique labels in this timepoint (across all Z slices)
            timepoint_data = mask.data[t, 0, :, :, :]  # Get all Z slices for this timepoint
            unique_labels = np.unique(timepoint_data)
            # Count non-zero labels (excluding background)
            non_zero_labels = unique_labels[unique_labels > 0]
            n_masks = len(non_zero_labels)
            
            # Calculate the size of each object (number of pixels per label)
            object_sizes = [
                np.sum(timepoint_data == label) for label in non_zero_labels
            ]
            max_object_size = max(object_sizes) if object_sizes else 0
            
            mask_counts.append(n_masks)
            max_object_sizes.append(max_object_size)
        
        filename = Path(mask_path).stem
        logging.info(f"  Found {T} timepoints with mask counts: {mask_counts}, max object sizes: {max_object_sizes}")
        return filename, mask_counts, max_object_sizes
        
    except Exception as e:
        logging.error(f"Error processing {mask_path}: {e}")
        return basename, [], []


def plot_mask_count_heatmap(
    grouped_files: Dict[str, Dict[str, str]],
    qc_key: str,
    mask_search_pattern: str,
    output_path: str = None,
    figsize: Tuple[int, int] = None
) -> None:
    """
    Create a heatmap showing mask counts over time for all input images.
    Colors are based on max object size, annotations show object count.
    Y-axis labels are colored red for failed QC, black for passed QC.
    
    Args:
        grouped_files: Dict mapping basename to dict of {'input': path, 'mask': path}
        qc_key: QC key to check (e.g., 'track_completeness')
        mask_search_pattern: Search pattern for documentation
        output_path: Optional path to save the plot. If None, displays interactively.
        figsize: Figure size as (width, height). If None, auto-calculated.
    """
    logging.info(f"Processing {len(grouped_files)} input files...")
    logging.info(f"Checking track completeness and QC status (key: {qc_key})...")
    
    # First, check all files for track completeness and QC status
    qc_status_map = {}  # basename -> (is_complete, reason)
    for basename, files in grouped_files.items():
        mask_path = files.get('mask')
        filename, is_complete, reason = check_track_completeness(
            mask_path, qc_key, mask_search_pattern, basename
        )
        qc_status_map[basename] = (is_complete, reason)
    
    # Collect data from all files
    count_rows = []
    size_rows = []
    for basename, files in grouped_files.items():
        mask_path = files.get('mask')
        filename, counts, max_object_sizes = count_masks_in_image(mask_path, basename)
        if counts:
            count_rows.append([basename] + counts)
            size_rows.append([basename] + max_object_sizes)
        else:
            # Add empty row for missing masks so they still appear in the plot
            count_rows.append([basename])
            size_rows.append([basename])
    
    if not count_rows:
        raise ValueError("No valid data found!")
    
    # Create DataFrames for both counts and max object sizes
    max_timepoints = max((len(row) - 1 for row in count_rows), default=0)
    
    # If all masks are missing, create a single column
    if max_timepoints == 0:
        max_timepoints = 1
        logging.warning("No mask data found for any input file. Creating placeholder heatmap.")
    
    columns = ['Image'] + [f'T{t}' for t in range(max_timepoints)]
    
    # Pad rows that have fewer timepoints
    padded_count_rows = []
    padded_size_rows = []
    for count_row, size_row in zip(count_rows, size_rows):
        if len(count_row) < len(columns):
            count_row = count_row + [0] * (len(columns) - len(count_row))
            size_row = size_row + [0] * (len(columns) - len(size_row))
        padded_count_rows.append(count_row)
        padded_size_rows.append(size_row)
    
    df_counts = pd.DataFrame(padded_count_rows, columns=columns).set_index('Image')
    df_sizes = pd.DataFrame(padded_size_rows, columns=columns).set_index('Image')
    
    logging.info(f"Created data matrix: {df_counts.shape[0]} images × {df_counts.shape[1]} timepoints")
    
    # Count QC results
    passed_count = sum(1 for is_complete, _ in qc_status_map.values() if is_complete)
    failed_count = len(qc_status_map) - passed_count
    logging.info(f"QC Results: {passed_count} PASSED, {failed_count} FAILED")
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        width = max(12, df_counts.shape[1] * 0.3)
        height = max(8, df_counts.shape[0] * 0.3)
        figsize = (width, height)
    
    # Determine if annotations should be shown (disable for >50 files)
    n_files = len(grouped_files)
    show_annotations = n_files <= 50
    
    if not show_annotations:
        logging.info(f"Disabling text annotations due to large number of files ({n_files} > 50)")
    
    # Create heatmap - color by max object size, annotate with count (if enabled)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # Use max object sizes for coloring
    sns.heatmap(
        df_sizes,  # Color based on max object size
        annot=df_counts if show_annotations else False,  # Show counts only if ≤50 files
        fmt='g' if show_annotations else '',
        cmap='YlGnBu',
        cbar_kws={'label': 'Max Object Size (pixels)'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        vmin=0
    )
    
    ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
    ax.set_ylabel('Image File', fontsize=12, fontweight='bold')
    
    # Adjust title based on whether annotations are shown
    title_suffix = "(Color = Max Object Size, Number = Object Count)" if show_annotations else "(Color = Max Object Size)"
    ax.set_title(f'Track Completeness QC - {qc_key}\n'
                 f'{title_suffix}\n'
                 f'PASSED: {passed_count} | FAILED: {failed_count}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Color y-axis labels based on QC status
    yticklabels = ax.get_yticklabels()
    for label in yticklabels:
        basename = label.get_text()
        if basename in qc_status_map:
            is_complete, _ = qc_status_map[basename]
            if not is_complete:
                label.set_color('red')
                label.set_weight('bold')
            else:
                label.set_color('black')
    
    # Rotate y-axis labels for better readability
    plt.yticks(rotation=0, fontsize=8)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved heatmap to: {output_path}")
        plt.close()
    else:
        plt.show()
    
    # Print summary of failed tracks
    if failed_count > 0:
        print("\n" + "="*60)
        print("FAILED TRACKS SUMMARY:")
        print("="*60)
        for basename, (is_complete, reason) in sorted(qc_status_map.items()):
            if not is_complete:
                print(f"❌ {basename}")
                print(f"   Reason: {reason}")
        print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description='QC Tracking - Validate track completeness and create QC heatmap',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check track completeness for all input images (shows all inputs, marks missing masks as failed)
  python qc_tracking.py --input-search-pattern "./input/*.tif" --mask-search-pattern "./masks/*_tracked.tif" --qc-key track_completeness
  
  # Save to file
  python qc_tracking.py --input-search-pattern "./input/*.tif" --mask-search-pattern "./masks/*_tracked.tif" --qc-key track_completeness --output track_qc.png
  
  # Custom figure size
  python qc_tracking.py --input-search-pattern "./input/*.tif" --mask-search-pattern "./masks/*_tracked.tif" --qc-key track_completeness --width 20 --height 15
  
  # Recursive search
  python qc_tracking.py --input-search-pattern "./input/**/*.tif" --mask-search-pattern "./masks/**/*_tracked.tif" --qc-key track_completeness --search-subfolders
  
  # Use different QC key
  python qc_tracking.py --input-search-pattern "./input/*.tif" --mask-search-pattern "./masks/*_tracked.tif" --qc-key tracking_validation
  
  # Process only first 10 files
  python qc_tracking.py --input-search-pattern "./input/*.tif" --mask-search-pattern "./masks/*_tracked.tif" --qc-key track_completeness --mode first10

Note:
  - The script uses rp.get_grouped_files_to_process to match input files to masks
  - All input files are shown in the heatmap, even if masks are missing
  - Missing masks are marked as FAILED with reason "Mask file not found"
  - Empty masks (zero objects in all timepoints) are marked as FAILED
  - The * wildcard in patterns is used for matching (e.g., input/image*.tif matches masks/image*_tracked.tif)
        """
    )
    
    parser.add_argument('--input-search-pattern', type=str, required=True,
                       help='Glob pattern for input images (e.g., "./input/*.tif")')
    parser.add_argument('--mask-search-pattern', type=str, required=True,
                       help='Glob pattern for mask images (e.g., "./masks/*_tracked.tif")')
    parser.add_argument('--qc-key', type=str, required=True,
                       help='QC key to use for checking results (e.g., "track_completeness")')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path. If not specified, displays interactively.')
    parser.add_argument('--remove-qc-failed', action='store_true',
                       help='Remove files that have failed QC from the plot (only show passed files)')
    parser.add_argument('--mode', type=str, default='all',
                       help='File selection mode: "all" = all files (default), "examples[N]" = first, middle, last N files, '
                            '"first[N]" = first N files, "random[N]" = N random files, '
                            '"group[N]" = N samples per experimental group')
    parser.add_argument('--search-subfolders', action='store_true',
                       help='Enable recursive search for files')
    parser.add_argument('--width', type=float, default=None,
                       help='Figure width in inches (auto-calculated if not specified)')
    parser.add_argument('--height', type=float, default=None,
                       help='Figure height in inches (auto-calculated if not specified)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Use get_grouped_files_to_process to map input files to masks and QC files
    search_patterns = {
        'input': args.input_search_pattern,
        'mask': args.mask_search_pattern,
    }
    
    logging.info(f"Searching for files with patterns:")
    logging.info(f"  Input: {args.input_search_pattern}")
    logging.info(f"  Mask:  {args.mask_search_pattern}")
    
    grouped_files = rp.get_grouped_files_to_process(search_patterns, args.search_subfolders)
    
    if not grouped_files:
        raise FileNotFoundError(f"No files found matching patterns")
    
    logging.info(f"Found {len(grouped_files)} input files")
    
    # Count how many have masks
    files_with_masks = sum(1 for files in grouped_files.values() if 'mask' in files)
    files_without_masks = len(grouped_files) - files_with_masks
    logging.info(f"  {files_with_masks} with masks, {files_without_masks} without masks")
    
    # Parse mode for file selection
    import re
    mode = str(args.mode).lower()  # Convert to string in case YAML passes boolean
    mode_match = re.match(r'(all|examples|first|random|group)(\d+)?$', mode)
    if not mode_match:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'all', 'examples[N]', 'first[N]', 'random[N]', or 'group[N]'")
    
    mode_type = mode_match.group(1)
    mode_count = int(mode_match.group(2)) if mode_match.group(2) else (1 if mode_type == 'group' else 3)
    
    # Select files based on mode
    if mode_type == 'group':
        # Group mode: select N samples per experimental group
        from collections import defaultdict
        from pathlib import Path
        
        # Import the extract_experimental_group function
        import sys
        parent_dir = Path(__file__).parent
        sys.path.insert(0, str(parent_dir))
        from quantify_distance_heatmap import extract_experimental_group
        
        groups = defaultdict(list)
        for basename, files in grouped_files.items():
            input_path = files.get('input', '')
            group = extract_experimental_group(input_path) if input_path else basename
            groups[group].append(basename)
        
        logging.info(f"Found {len(groups)} experimental groups")
        
        selected_basenames = []
        for group_name, basenames in sorted(groups.items()):
            n_available = len(basenames)
            n_select = min(mode_count, n_available)
            
            if n_available <= mode_count:
                selected = basenames
            else:
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
                selected = [basenames[i] for i in indices]
            
            selected_basenames.extend(selected)
            logging.info(f"  Group '{group_name}': selected {len(selected)}/{n_available} samples")
        
        # Filter grouped_files to only include selected basenames
        grouped_files = {basename: files for basename, files in grouped_files.items() if basename in selected_basenames}
        
    elif mode_type != 'all':
        # Standard selection modes
        basenames = list(grouped_files.keys())
        n = len(basenames)
        if mode_type == 'examples':
            if n > mode_count:
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
                basenames = [basenames[i] for i in indices]
        elif mode_type == 'first':
            basenames = basenames[:min(mode_count, n)]
        elif mode_type == 'random':
            import random
            basenames = random.sample(basenames, min(mode_count, n))
        
        # Filter grouped_files
        grouped_files = {basename: files for basename, files in grouped_files.items() if basename in basenames}
    
    logging.info(f"Processing {len(grouped_files)} files after mode selection (mode: {args.mode})")
    
    # Determine figure size
    figsize = None
    if args.width is not None and args.height is not None:
        figsize = (args.width, args.height)
    elif args.width is not None or args.height is not None:
        logging.warning("Both --width and --height must be specified. Using auto size.")
    
    # Create heatmap with QC validation
    plot_mask_count_heatmap(
        grouped_files=grouped_files,
        qc_key=args.qc_key,
        mask_search_pattern=args.mask_search_pattern,
        output_path=args.output,
        figsize=figsize
    )
    
    logging.info("Processing complete!")


if __name__ == "__main__":
    main()

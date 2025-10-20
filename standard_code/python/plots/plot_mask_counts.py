"""
Plot Mask Count Heatmap

Creates a heatmap showing the number of masks per timepoint for all images.
Useful for quality control and identifying tracking issues.

Author: BIPHUB - Bioimage Informatics Hub, University of Oslo
License: MIT
"""

''' TODO
This is interesing and I am happy that I ran this. since there are some missing frames. can you write a function that closes the segmentation gaps if there are less missing frames than --max-gap-size. can you make some "interpolation" that if --max-gap-size = 1 makes one intermediate between frame-1 and frame +1. if the gap is two then two steps should be created. also we need to think carefully what happens if we have more than one object per cell. since when there are missing things the values are not reliable. there should not be overlap between them though, but you will figure this out. 
'''



import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple
import logging
import pandas as pd

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

def count_masks_in_image(mask_path: str) -> Tuple[str, List[int], List[int]]:
    """
    Count the number of unique masks (labels) and find max label ID in each timepoint of a mask file.
    
    Args:
        mask_path: Path to the mask image
        
    Returns:
        Tuple of (filename, list of mask counts per timepoint, list of max label IDs per timepoint)
    """
    logging.info(f"Processing: {Path(mask_path).name}")
    
    try:
        mask = rp.load_tczyx_image(mask_path)
        T, C, Z, Y, X = mask.shape
        
        mask_counts = []
        max_labels = []
        for t in range(T):
            # Get all unique labels in this timepoint (across all Z slices)
            timepoint_data = mask.data[t, 0, :, :, :]  # Get all Z slices for this timepoint
            unique_labels = np.unique(timepoint_data)
            # Count non-zero labels (excluding background)
            non_zero_labels = unique_labels[unique_labels > 0]
            n_masks = len(non_zero_labels)
            max_label = int(non_zero_labels.max()) if len(non_zero_labels) > 0 else 0
            
            mask_counts.append(n_masks)
            max_labels.append(max_label)
        
        filename = Path(mask_path).stem
        logging.info(f"  Found {T} timepoints with mask counts: {mask_counts}, max labels: {max_labels}")
        return filename, mask_counts, max_labels
        
    except Exception as e:
        logging.error(f"Error processing {mask_path}: {e}")
        return Path(mask_path).stem, [], []


def plot_mask_count_heatmap(
    mask_files: List[str],
    output_path: str = None,
    figsize: Tuple[int, int] = None
) -> None:
    """
    Create a heatmap showing mask counts over time for all input masks.
    Colors are based on max label ID, annotations show object count.
    
    Args:
        mask_files: List of paths to mask files
        output_path: Optional path to save the plot. If None, displays interactively.
        figsize: Figure size as (width, height). If None, auto-calculated.
    """
    logging.info(f"Processing {len(mask_files)} mask files...")
    
    # Collect data from all mask files
    count_rows = []
    label_rows = []
    for mask_path in mask_files:
        filename, counts, max_labels = count_masks_in_image(mask_path)
        if counts:
            count_rows.append([filename] + counts)
            label_rows.append([filename] + max_labels)
    
    if not count_rows:
        raise ValueError("No valid mask data found!")
    
    # Create DataFrames for both counts and max labels
    max_timepoints = max(len(row) - 1 for row in count_rows)
    columns = ['Image'] + [f'T{t}' for t in range(max_timepoints)]
    
    # Pad rows that have fewer timepoints
    padded_count_rows = []
    padded_label_rows = []
    for count_row, label_row in zip(count_rows, label_rows):
        if len(count_row) < len(columns):
            count_row = count_row + [0] * (len(columns) - len(count_row))
            label_row = label_row + [0] * (len(columns) - len(label_row))
        padded_count_rows.append(count_row)
        padded_label_rows.append(label_row)
    
    df_counts = pd.DataFrame(padded_count_rows, columns=columns).set_index('Image')
    df_labels = pd.DataFrame(padded_label_rows, columns=columns).set_index('Image')
    
    logging.info(f"Created data matrix: {df_counts.shape[0]} images × {df_counts.shape[1]} timepoints")
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        # Get the size of a single monitor to constrain figure size
        # This is important for multi-monitor setups
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            root.update_idletasks()  # Process pending events
            
            # Get the size of the primary monitor (not combined desktop)
            monitor_width_px = root.winfo_screenwidth()
            monitor_height_px = root.winfo_screenheight()
            
            # Properly destroy the tkinter root to avoid event errors
            try:
                root.update()  # Final update
            except:
                pass
            root.quit()
            try:
                root.destroy()
            except:
                pass
            
            # Convert to inches (assuming 96 DPI, standard on Windows)
            dpi = 96
            monitor_width_in = monitor_width_px / dpi
            monitor_height_in = monitor_height_px / dpi
            
            # Use 85% of monitor to leave room for window decorations
            max_width = monitor_width_in * 0.85
            max_height = monitor_height_in * 0.80  # Leave more room for taskbar
            
            # Calculate figure size based on data dimensions
            width = min(max_width, max(12, df_counts.shape[1] * 0.3))
            height = min(max_height, max(8, df_counts.shape[0] * 0.3))
            figsize = (width, height)
            
            logging.info(f"Monitor: {monitor_width_px}×{monitor_height_px} px ({monitor_width_in:.1f}×{monitor_height_in:.1f} in)")
            logging.info(f"Creating figure: {figsize[0]:.1f}×{figsize[1]:.1f} in")
        except Exception as e:
            # Fallback to default calculation
            width = max(12, df_counts.shape[1] * 0.3)
            height = max(8, df_counts.shape[0] * 0.3)
            figsize = (width, height)
            logging.debug(f"Monitor detection failed: {e}, using fallback size")
    
    # Create heatmap - color by max label ID, annotate with count
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # Use max label IDs for coloring
    sns.heatmap(
        df_labels,  # Color based on max label ID
        annot=df_counts,  # Show counts as annotations
        fmt='g',
        cmap='YlOrRd',
        cbar_kws={'label': 'Max Object ID'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        vmin=0
    )
    
    ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
    ax.set_ylabel('Image File', fontsize=12, fontweight='bold')
    ax.set_title('Mask Count per Timepoint - Quality Control Heatmap\n(Color = Max Object ID, Number = Object Count)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Rotate y-axis labels for better readability
    plt.yticks(rotation=0, fontsize=8)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    
    # Save or show
    if output_path:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Created output directory: {output_dir}")
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved heatmap to: {output_path}")
        plt.close()
    else:
        # Try to maximize, but don't error if it fails
        try:
            manager = plt.get_current_fig_manager()
            if hasattr(manager, 'window'):
                if hasattr(manager.window, 'state'):
                    manager.window.state('zoomed')  # TkAgg
                elif hasattr(manager.window, 'showMaximized'):
                    manager.window.showMaximized()  # Qt
        except:
            pass  # Silently fail if maximization doesn't work
        
        plt.show()
    
    # Print summary statistics
    logging.info("\n" + "="*60)
    logging.info("SUMMARY STATISTICS")
    logging.info("="*60)
    
    # Find files with tracking issues (changing mask counts)
    issues = []
    for idx, row in df_counts.iterrows():
        counts = row.values
        if len(set(counts[counts > 0])) > 1:  # More than one unique non-zero count
            issues.append(idx)
    
    if issues:
        logging.warning(f"Found {len(issues)} images with varying mask counts (potential tracking issues):")
        for img in issues:
            counts = df_counts.loc[img].values
            logging.warning(f"  {img}: counts vary from {counts.min()} to {counts.max()}")
    else:
        logging.info("✓ All images have consistent mask counts across timepoints")
    
    # Files with zero masks at any timepoint
    zero_issues = []
    for idx, row in df_counts.iterrows():
        if 0 in row.values:
            zero_issues.append(idx)
    
    if zero_issues:
        logging.warning(f"\nFound {len(zero_issues)} images with zero masks in some timepoints:")
        for img in zero_issues:
            zero_timepoints = [i for i, val in enumerate(df_counts.loc[img].values) if val == 0]
            logging.warning(f"  {img}: zero masks at T={zero_timepoints}")
    
    logging.info("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Create heatmap of mask counts per timepoint for quality control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create heatmap from all masks
  python plot_mask_counts.py --mask-search-pattern "./output_masks/*_segmentation.tif"
  
  # Save to file
  python plot_mask_counts.py --mask-search-pattern "./output_masks/*_segmentation.tif" --output mask_qc.png
  
  # Custom figure size
  python plot_mask_counts.py --mask-search-pattern "./output_masks/*_segmentation.tif" --width 20 --height 15
  
  # Recursive search
  python plot_mask_counts.py --mask-search-pattern "./experiments/**/*_mask.tif" --search-subfolders
        """
    )
    
    parser.add_argument('--mask-search-pattern', type=str, required=True,
                       help='Glob pattern for mask images')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path. If not specified, displays interactively.')
    parser.add_argument('--search-subfolders', action='store_true',
                       help='Enable recursive search for files')
    parser.add_argument('--width', type=float, default=None,
                       help='Figure width in inches (auto-calculated if not specified)')
    parser.add_argument('--height', type=float, default=None,
                       help='Figure height in inches (auto-calculated if not specified)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Find mask files
    mask_files = rp.get_files_to_process2(args.mask_search_pattern, args.search_subfolders)
    
    if not mask_files:
        raise FileNotFoundError(f"No files found matching pattern: {args.mask_search_pattern}")
    
    logging.info(f"Found {len(mask_files)} mask files")
    
    # Sort files for consistent ordering
    mask_files.sort()
    
    # Determine figure size
    figsize = None
    if args.width is not None and args.height is not None:
        figsize = (args.width, args.height)
    elif args.width is not None or args.height is not None:
        logging.warning("Both --width and --height must be specified. Using auto size.")
    
    # Create heatmap
    plot_mask_count_heatmap(mask_files, args.output, figsize)
    
    logging.info("Processing complete!")


if __name__ == "__main__":
    main()

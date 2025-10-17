"""
Visualize distance matrices with colored overlays and intensity heatmaps.

This module creates visualizations showing distance information from masks,
with distance-based coloring and aggregate intensity plots.

Author: BIPHUB - Bioimage Informatics Hub, University of Oslo
License: MIT
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Tuple, Optional, List
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


def plot_distance_overlay(
    image_path: str,
    distance_path: str,
    output_path: Optional[str] = None,
    channel: int = 0,
    z_slice: Optional[int] = None,
    colormap: str = 'viridis',
    num_distance_bins: int = 50,
    overlay_timepoints: str = 'first_middle_last',
    show_bin_contours: bool = False
) -> None:
    """
    Create a plot showing image with distance-based colored overlay.
    
    The image grayscale values control alpha (transparency), while distance
    values control the color. Low intensity = transparent, high intensity = solid.
    Distances are binned to match the heatmap visualization.
    
    Args:
        image_path: Path to the input image
        distance_path: Path to the distance matrix image
        output_path: Optional path to save the plot. If None, displays interactively.
        channel: Which channel to display (default: 0)
        z_slice: Which Z slice to display. If None, uses maximum intensity projection.
        colormap: Matplotlib colormap name (default: 'viridis')
        num_distance_bins: Number of bins for distance values (default: 50)
        overlay_timepoints: Which timepoints to show. Options:
            'all' = all timepoints
            'first' = first timepoint only
            'last' = last timepoint only
            'first_middle_last' = first, middle, and last (default)
            'first[N]' = first N timepoints (e.g., 'first2', 'first5')
            'last[N]' = last N timepoints (e.g., 'last2', 'last3')
            'middle[N]' = N evenly spaced timepoints (e.g., 'middle3')
        show_bin_contours: If True, overlay contour lines showing distance bin boundaries (default: False)
    """
    logging.info(f"Loading image: {image_path}")
    img = rp.load_tczyx_image(image_path)
    
    logging.info(f"Loading distance matrix: {distance_path}")
    dist = rp.load_tczyx_image(distance_path)
    
    T, C, Z, Y, X = img.shape
    dT, dC, dZ, dY, dX = dist.shape
    
    logging.info(f"Image shape: {img.shape}")
    logging.info(f"Distance shape: {dist.shape}")
    
    # Validate dimensions
    if (Y, X) != (dY, dX):
        raise ValueError(f"Image and distance XY dimensions must match. Got image: ({Y}, {X}), distance: ({dY}, {dX})")
    
    # Find global min/max distance across all timepoints (excluding zeros) for consistent binning
    all_nonzero = []
    for t in range(dT):
        data_t = dist.data[t, 0, :, :, :]
        nonzero_vals = data_t[data_t > 0]
        if len(nonzero_vals) > 0:
            all_nonzero.append(nonzero_vals)
    
    if len(all_nonzero) == 0:
        logging.warning(f"No non-zero distance values found in {Path(distance_path).name}")
        global_dist_min = 0
        global_dist_max = 1
        distance_bins = None
    else:
        all_nonzero = np.concatenate(all_nonzero)
        global_dist_min = float(np.min(all_nonzero))
        global_dist_max = float(np.max(all_nonzero))
        # Create distance bins matching the heatmap
        distance_bins = np.linspace(global_dist_min, global_dist_max, num_distance_bins + 1)
        logging.info(f"Distance range: {global_dist_min:.2f} to {global_dist_max:.2f}, bins: {num_distance_bins}")
    
    # Determine which timepoints to show based on overlay_timepoints argument
    import re
    overlay_mode = overlay_timepoints.lower()
    
    if overlay_mode == 'all':
        timepoints = list(range(T))
    elif overlay_mode == 'first':
        timepoints = [0]
    elif overlay_mode == 'last':
        timepoints = [T - 1]
    elif overlay_mode == 'first_middle_last':
        if T == 1:
            timepoints = [0]
        elif T == 2:
            timepoints = [0, 1]
        else:
            middle_t = T // 2
            timepoints = [0, middle_t, T - 1]
    elif overlay_mode.startswith('first'):
        match = re.match(r'first(\d+)$', overlay_mode)
        if match:
            count = min(int(match.group(1)), T)
            timepoints = list(range(count))
        else:
            raise ValueError(f"Invalid overlay_timepoints format: {overlay_timepoints}")
    elif overlay_mode.startswith('last'):
        match = re.match(r'last(\d+)$', overlay_mode)
        if match:
            count = min(int(match.group(1)), T)
            timepoints = list(range(T - count, T))
        else:
            raise ValueError(f"Invalid overlay_timepoints format: {overlay_timepoints}")
    elif overlay_mode.startswith('middle'):
        match = re.match(r'middle(\d+)$', overlay_mode)
        if match:
            count = min(int(match.group(1)), T)
            if count == 1:
                timepoints = [T // 2]
            elif count == 2:
                timepoints = [0, T - 1]
            else:
                indices = [0]
                for i in range(1, count - 1):
                    idx = int(i * T / (count - 1))
                    indices.append(idx)
                indices.append(T - 1)
                timepoints = indices
        else:
            raise ValueError(f"Invalid overlay_timepoints format: {overlay_timepoints}")
    else:
        raise ValueError(f"Invalid overlay_timepoints: {overlay_timepoints}. Must be 'all', 'first', 'last', 'first_middle_last', 'first[N]', 'last[N]', or 'middle[N]'")
    
    # Handle distance timepoint mapping
    if dT == 1 and T > 1:
        dist_t_map = {t: 0 for t in timepoints}
    elif dT == T:
        dist_t_map = {t: t for t in timepoints}
    else:
        raise ValueError(f"Incompatible timepoints: image has {T}, distance has {dT}")
    
    n_cols = len(timepoints)
    
    # Get monitor size for figure scaling
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        
        monitor_width_px = root.winfo_screenwidth()
        monitor_height_px = root.winfo_screenheight()
        
        try:
            root.update()
        except:
            pass
        root.quit()
        try:
            root.destroy()
        except:
            pass
        
        dpi = 96
        monitor_width_in = monitor_width_px / dpi
        monitor_height_in = monitor_height_px / dpi
        
        max_width = monitor_width_in * 0.85
        max_height = monitor_height_in * 0.85
        
        width_per_col = max_width / n_cols
        height = max_height * 0.8
        
        figsize = (width_per_col * n_cols, height)
    except:
        figsize = (6 * n_cols, 5)
    
    fig, axes = plt.subplots(1, n_cols, figsize=figsize, constrained_layout=True)
    
    if n_cols == 1:
        axes = [axes]
    
    for col_idx, t in enumerate(timepoints):
        ax = axes[col_idx]
        dt = dist_t_map[t]
        
        # Get image and distance slices
        if z_slice is not None:
            img_2d = img.data[t, channel, z_slice, :, :]
            dist_2d = dist.data[dt, 0, z_slice, :, :]
        else:
            # Maximum intensity projection
            img_2d = np.max(img.data[t, channel, :, :, :], axis=0)
            dist_2d = np.max(dist.data[dt, 0, :, :, :], axis=0)
        
        # Normalize image intensity to [0, 1] for alpha values
        img_norm = img_2d.astype(float)
        img_min, img_max = np.percentile(img_norm[img_norm > 0], [1, 99]) if np.any(img_norm > 0) else (0, 1)
        img_norm = np.clip((img_norm - img_min) / (img_max - img_min + 1e-10), 0, 1)
        
        # Create RGBA image where distance controls color and intensity controls alpha
        # Distance values (exclude zeros)
        dist_2d_raw = dist_2d.copy()
        dist_mask = dist_2d_raw > 0
        
        if np.any(dist_mask) and distance_bins is not None:
            # Bin the distance values to match heatmap
            dist_binned = np.zeros_like(dist_2d_raw)
            
            for i in range(len(distance_bins) - 1):
                bin_min = distance_bins[i]
                bin_max = distance_bins[i + 1]
                bin_center = (bin_min + bin_max) / 2
                
                if i == len(distance_bins) - 2:  # Last bin includes upper edge
                    bin_mask = (dist_2d_raw >= bin_min) & (dist_2d_raw <= bin_max)
                else:
                    bin_mask = (dist_2d_raw >= bin_min) & (dist_2d_raw < bin_max)
                
                dist_binned[bin_mask] = bin_center
            
            # Create colormap
            cmap = plt.get_cmap(colormap)
            norm = Normalize(vmin=global_dist_min, vmax=global_dist_max)
            
            # Convert binned distance to colors (only for non-zero distances)
            colored = np.zeros((dist_2d.shape[0], dist_2d.shape[1], 4))
            colored[dist_mask] = cmap(norm(dist_binned[dist_mask]))
            
            # Set alpha based on image intensity (only where distance > 0)
            colored[dist_mask, 3] = img_norm[dist_mask]
            
            # Display
            ax.imshow(colored, interpolation='nearest')
            
            # Add bin contours if requested
            if show_bin_contours and distance_bins is not None:
                # Draw contour lines at bin boundaries
                # Use fewer contours for readability - show every 5th bin or so
                contour_interval = max(1, len(distance_bins) // 10)
                contour_levels = distance_bins[::contour_interval]
                
                # Make sure we have some contours to draw
                if len(contour_levels) > 0:
                    try:
                        contours = ax.contour(
                            dist_2d_raw,
                            levels=contour_levels,
                            colors='cyan',  # Changed to cyan for better visibility
                            linewidths=1.5,  # Thicker lines
                            alpha=0.8,  # More opaque
                            linestyles='solid'
                        )
                        
                        # Add labels to contours
                        ax.clabel(contours, inline=True, fontsize=9, fmt='%.1f', 
                                 colors='cyan', inline_spacing=10)
                        
                        logging.info(f"Drew {len(contour_levels)} contour levels")
                    except Exception as e:
                        logging.warning(f"Could not draw contours: {e}")
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Distance (binned)', rotation=270, labelpad=15)
        else:
            # No distance data, show grayscale image
            ax.imshow(img_2d, cmap='gray', interpolation='nearest')
            ax.text(0.5, 0.5, 'No distance data', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, color='red')
        
        z_info = f"Z={z_slice}" if z_slice is not None else "MIP"
        ax.set_title(f'T={t} ({z_info})', fontsize=11)
        ax.axis('off')
    
    # Add overall title
    img_name = Path(image_path).stem
    dist_name = Path(distance_path).stem
    fig.suptitle(f'{img_name}\nDistance: {dist_name}', 
                fontsize=14, fontweight='bold')
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved plot to: {output_path}")
        plt.close()
    else:
        fig.canvas.manager.set_window_title(f'{Path(image_path).stem} - Distance Overlay')
        
        try:
            manager = plt.get_current_fig_manager()
            if hasattr(manager, 'window'):
                if hasattr(manager.window, 'state'):
                    manager.window.state('zoomed')
                elif hasattr(manager.window, 'showMaximized'):
                    manager.window.showMaximized()
        except:
            pass
        
        plt.show()


def plot_distance_time_heatmap(
    image_path: str,
    distance_path: str,
    output_path: str = None,
    channel: int = 0,
    colormap: str = 'viridis',
    normalize_to_t0: bool = False,
    force_show: bool = False
) -> None:
    """
    Create a heatmap showing image intensity distribution over time and distance.
    
    X-axis: Time (timepoints)
    Y-axis: Distance values (already binned in the distance matrix)
    Color: Sum of image intensity at each (time, distance) bin
    
    This function uses the distance values directly as bins (assumes distance matrix
    is already binned) and sums the corresponding image intensities for each bin.
    
    Args:
        image_path: Path to the input image file
        distance_path: Path to distance matrix file (already binned)
        output_path: Optional path to save the plot. If None, displays interactively.
        channel: Which channel to use from the image (default: 0)
        colormap: Matplotlib colormap name (default: 'viridis')
        normalize_to_t0: If True, normalize each distance bin by its T0 value to show relative changes (default: False)
        force_show: If True, display plot interactively even when output_path is specified (default: False)
    """
    logging.info(f"Processing: {Path(image_path).name}")
    
    # Load image and distance data
    img = rp.load_tczyx_image(image_path)
    dist = rp.load_tczyx_image(distance_path)
    
    T, C, Z, Y, X = img.shape
    dT, dC, dZ, dY, dX = dist.shape
    
    logging.info(f"Image shape: {img.shape}")
    logging.info(f"Distance shape: {dist.shape}")
    
    # Validate dimensions
    if (Y, X) != (dY, dX):
        raise ValueError(f"Image and distance XY dimensions must match. Got image: ({Y}, {X}), distance: ({dY}, {dX})")
    
    if T != dT:
        raise ValueError(f"Image and distance must have same number of timepoints. Got image: {T}, distance: {dT}")
    
    # Get unique distance values (which are already binned)
    # across all timepoints to determine the bins
    all_distance_values = []
    for t in range(T):
        data_t = dist.data[t, 0, :, :, :]
        unique_vals = np.unique(data_t[data_t > 0])  # Exclude zeros
        if len(unique_vals) > 0:
            all_distance_values.extend(unique_vals)
    
    if len(all_distance_values) == 0:
        logging.error(f"No non-zero distance values found in {Path(distance_path).name}")
        return
    
    # Get sorted unique distance bins
    distance_bins = np.sort(np.unique(all_distance_values))
    num_distance_bins = len(distance_bins)
    
    logging.info(f"Found {num_distance_bins} unique distance bins")
    logging.info(f"Distance range: {distance_bins[0]:.2f} to {distance_bins[-1]:.2f}")
    
    # Create 2D histogram: rows = distance bins, cols = timepoints
    # Sum IMAGE INTENSITY (not distance values) within each bin
    heatmap_data = np.zeros((num_distance_bins, T))
    
    for t in range(T):
        # Get distance map and corresponding image data for this timepoint
        dist_t = dist.data[t, 0, :, :, :]
        img_t = img.data[t, channel, :, :, :]
        
        # For each distance bin value, sum the intensities where distance equals that bin
        for i, bin_value in enumerate(distance_bins):
            bin_mask = dist_t == bin_value
            
            # Sum the IMAGE INTENSITIES (not distances) for this bin
            heatmap_data[i, t] = np.sum(img_t[bin_mask])
    
    # Normalize to T0 if requested (subtract T0 to make it zero baseline)
    if normalize_to_t0:
        t0_values = heatmap_data[:, 0].copy()
        # Subtract T0 from each timepoint (T0 becomes 0)
        heatmap_data = heatmap_data - t0_values[:, np.newaxis]
        
        # Verify T0 is now zero (within floating point precision)
        t0_check = np.max(np.abs(heatmap_data[:, 0]))
        logging.info(f"T0 normalization complete. Max absolute T0 value: {t0_check:.2e} (should be ~0)")
        
        colorbar_label = 'Intensity Change from T0'
        title_suffix = ' (T0 = 0 Baseline)'
    else:
        colorbar_label = 'Sum Image Intensity'
        title_suffix = ''
    
    # Create DataFrame for seaborn
    df = pd.DataFrame(
        heatmap_data,
        index=[f'{bin_val:.1f}' for bin_val in distance_bins],
        columns=[f'T{t}' for t in range(T)]
    )
    
    # Save CSV if output_path is specified
    if output_path:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename from image path
        base_filename = Path(image_path).stem
        
        # Create TSV with metadata
        tsv_path = output_path.replace('.png', '.tsv')
        
        # Create a copy of the dataframe with metadata columns
        csv_df = df.copy()
        
        # Add metadata columns at the beginning
        csv_df.insert(0, 'Base_Filename', base_filename)
        csv_df.insert(1, 'Channel', channel)
        csv_df.insert(2, 'Normalized_to_T0', normalize_to_t0)
        if normalize_to_t0:
            csv_df.insert(3, 'Units', 'Intensity_Change_from_T0')
        else:
            csv_df.insert(3, 'Units', 'Sum_Image_Intensity')
        
        # Set index name for distance bins
        csv_df.index.name = 'Distance_Bin'
        
        # Save as TSV (tab-separated, better for Excel)
        csv_df.to_csv(tsv_path, sep='\t')
        logging.info(f"Saved TSV data to: {tsv_path}")
    
    
    # Auto-calculate figure size
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        
        monitor_width_px = root.winfo_screenwidth()
        monitor_height_px = root.winfo_screenheight()
        
        try:
            root.update()
        except:
            pass
        root.quit()
        try:
            root.destroy()
        except:
            pass
        
        dpi = 96
        monitor_width_in = monitor_width_px / dpi
        monitor_height_in = monitor_height_px / dpi
        
        max_width = monitor_width_in * 0.85
        max_height = monitor_height_in * 0.85
        
        width = min(max_width, max(10, T * 0.4))
        height = min(max_height, max(8, num_distance_bins * 0.15))
        figsize = (width, height)
    except:
        width = max(10, T * 0.4)
        height = max(8, num_distance_bins * 0.15)
        figsize = (width, height)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    sns.heatmap(
        df,
        annot=False,
        cmap=colormap,
        cbar_kws={'label': colorbar_label},
        linewidths=0,
        ax=ax,
        vmin=0 if not normalize_to_t0 else None  # Auto-scale if normalized
    )
    
    # Invert y-axis so shortest distance is at the bottom
    ax.invert_yaxis()
    
    ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance', fontsize=12, fontweight='bold')
    
    img_name = Path(image_path).stem
    ax.set_title(f'Image Intensity by Distance Over Time{title_suffix}\n{img_name}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Adjust tick labels
    # Show every Nth y-tick to avoid crowding
    y_tick_interval = max(1, num_distance_bins // 10)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks[::y_tick_interval])
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    # Save or show
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved heatmap to: {output_path}")
        
        # Show interactively if force_show is enabled
        if force_show:
            fig.canvas.manager.set_window_title(f'{Path(image_path).stem} - Distance-Time Heatmap')
            
            try:
                manager = plt.get_current_fig_manager()
                if hasattr(manager, 'window'):
                    if hasattr(manager.window, 'state'):
                        manager.window.state('zoomed')
                    elif hasattr(manager.window, 'showMaximized'):
                        manager.window.showMaximized()
            except:
                pass
            
            plt.show()
        else:
            plt.close()
    else:
        fig.canvas.manager.set_window_title(f'{Path(image_path).stem} - Distance-Time Heatmap')
        
        try:
            manager = plt.get_current_fig_manager()
            if hasattr(manager, 'window'):
                if hasattr(manager.window, 'state'):
                    manager.window.state('zoomed')
                elif hasattr(manager.window, 'showMaximized'):
                    manager.window.showMaximized()
        except:
            pass
        
        plt.show()
    
    # Print summary statistics
    total_intensity = np.sum(heatmap_data)
    logging.info(f"Total image intensity in heatmap: {total_intensity:.2e}")
    logging.info(f"Distance range: {distance_bins[0]:.2f} - {distance_bins[-1]:.2f}")
    logging.info(f"Number of distance bins: {num_distance_bins}")
    logging.info(f"Timepoints: {T}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize distance matrices with colored overlays and intensity heatmaps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create distance-time heatmaps for 3 evenly-spaced examples (default)
  python plot_distance.py --input-search-pattern "./images/*.tif" --distance-search-pattern "./distances/*_geodesic.tif"
  
  # Save outputs instead of displaying
  python plot_distance.py --input-search-pattern "./images/*.tif" --distance-search-pattern "./distances/*_geodesic.tif" --output-folder ./plots
  
  # Save outputs AND display interactively
  python plot_distance.py --input-search-pattern "./images/*.tif" --distance-search-pattern "./distances/*_geodesic.tif" --output-folder ./plots --force-show
  
  # Show first 5 files
  python plot_distance.py --input-search-pattern "./images/*.tif" --distance-search-pattern "./distances/*_geodesic.tif" --mode first5
  
  # Create heatmaps normalized to T0 (first timepoint) to show relative changes
  python plot_distance.py --input-search-pattern "./images/*.tif" --distance-search-pattern "./distances/*_geodesic.tif" --normalize-to-t0
  
  # Show random 10 heatmaps
  python plot_distance.py --input-search-pattern "./images/*.tif" --distance-search-pattern "./distances/*_geodesic.tif" --mode random10
  
  # Process all files
  python plot_distance.py --input-search-pattern "./images/*.tif" --distance-search-pattern "./distances/*_geodesic.tif" --mode all
  
  # Use different colormap
  python plot_distance.py --input-search-pattern "./images/*.tif" --distance-search-pattern "./distances/*_geodesic.tif" --colormap plasma
  
  # Show specific channel in heatmaps
  python plot_distance.py --input-search-pattern "./images/*.tif" --distance-search-pattern "./distances/*_geodesic.tif" --channel 1
        """
    )
    
    parser.add_argument('--input-search-pattern', type=str, required=True,
                       help='Glob pattern for input images (REQUIRED)')
    parser.add_argument('--distance-search-pattern', type=str, required=True,
                       help='Glob pattern for distance matrix images (REQUIRED)')
    parser.add_argument('--output-folder', type=str, default=None,
                       help='Folder to save plots. If not specified, displays interactively.')
    parser.add_argument('--mode', type=str, default='examples3',
                       help='File selection mode for heatmaps: "all" = all files, "examples[N]" = first, middle, last N files (default: 3), '
                            '"first[N]" = first N files (default: 3), "random[N]" = N random files (default: 3)')
    parser.add_argument('--normalize-to-t0', action='store_true',
                       help='Use T0 (first timepoint) as zero baseline - subtract T0 from all timepoints to show changes (T0 = 0)')
    parser.add_argument('--force-show', action='store_true',
                       help='Display plots interactively even when saving to output folder')
    parser.add_argument('--search-subfolders', action='store_true',
                       help='Enable recursive search for files')
    parser.add_argument('--channel', type=int, default=0,
                       help='Which channel to display for heatmaps (default: 0)')
    parser.add_argument('--colormap', type=str, default='viridis',
                       help='Matplotlib colormap for heatmaps (default: viridis)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Both patterns are required - build search patterns dict
    patterns = {
        'image': args.input_search_pattern,
        'distance': args.distance_search_pattern
    }
    
    # Get grouped files - this is our primary file discovery
    grouped_files = rp.get_grouped_files_to_process(patterns, args.search_subfolders)
    
    if not grouped_files:
        raise FileNotFoundError(f"No files found matching patterns: {patterns}")
    
    logging.info(f"Found {len(grouped_files)} grouped file set(s)")
    
    # Filter to only complete pairs (both image and distance present)
    complete_pairs = {
        basename: files 
        for basename, files in grouped_files.items() 
        if 'image' in files and 'distance' in files
    }
    
    if not complete_pairs:
        raise ValueError("No complete image-distance pairs found!")
    
    logging.info(f"Found {len(complete_pairs)} complete image-distance pair(s)")
    
    # Parse mode for file selection
    mode = args.mode.lower()
    import re
    mode_match = re.match(r'(all|examples|first|random)(\d+)?$', mode)
    if not mode_match:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'all', 'examples[N]', 'first[N]', or 'random[N]'")
    
    mode_type = mode_match.group(1)
    mode_count = int(mode_match.group(2)) if mode_match.group(2) else 3
    
    # Helper function to select basenames based on mode
    def select_basenames(basenames_list, mode_type, mode_count):
        n = len(basenames_list)
        if mode_type == 'all':
            return basenames_list
        elif mode_type == 'examples':
            if n <= mode_count:
                return basenames_list
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
            return [basenames_list[i] for i in indices]
        elif mode_type == 'first':
            count = min(mode_count, n)
            return basenames_list[:count]
        elif mode_type == 'random':
            import random
            count = min(mode_count, n)
            return random.sample(basenames_list, count)
        return basenames_list
    
    # Create heatmaps
    basenames = sorted(complete_pairs.keys())
    selected_basenames = select_basenames(basenames, mode_type, mode_count)
    
    logging.info(f"Creating {len(selected_basenames)} distance-time heatmap(s)")
    
    for basename in selected_basenames:
        img_path = complete_pairs[basename]['image']
        dist_path = complete_pairs[basename]['distance']
        
        if args.output_folder:
            output_name = f"{Path(img_path).stem}_distance_time_heatmap.png"
            heatmap_output = os.path.join(args.output_folder, output_name)
        else:
            heatmap_output = None
        
        try:
            plot_distance_time_heatmap(
                img_path,
                dist_path,
                output_path=heatmap_output,
                channel=args.channel,
                colormap=args.colormap,
                normalize_to_t0=args.normalize_to_t0,
                force_show=args.force_show
            )
        except Exception as e:
            logging.error(f"Error creating heatmap for {Path(img_path).name}: {e}")
            continue
    
    logging.info("Processing complete!")


if __name__ == "__main__":
    main()

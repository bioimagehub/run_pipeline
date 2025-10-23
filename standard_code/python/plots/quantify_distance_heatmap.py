"""
Quantify signal spread and decay from distance matrices.

This module performs quantification analysis on distance matrices paired with images,
measuring signal spread (distance) and decay (time) using region growing segmentation.

Author: BIPHUB - Bioimage Informatics Hub, University of Oslo
License: MIT
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import logging
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import tkinter as tk
import re
from multiprocessing import Pool, cpu_count
from functools import partial

# Local imports
try:
    from .. import bioimage_pipeline_utils as rp
except ImportError:
    # Fallback for when script is run directly (not as module)
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
        logging.info(f"No non-zero distance values found in {Path(distance_path).name}")
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
    force_show: bool = False,
    remove_first_n_bins: int = 5,
    max_measure_pixels: int = 20,
    random_seed: int = 42
) -> None:
    """
    Create a heatmap showing image intensity distribution over time and distance.
    
    X-axis: Time (timepoints)
    Y-axis: Distance values (already binned in the distance matrix)
    Color: Sum of image intensity at each (time, distance) bin
    
    This function uses the distance values directly as bins (assumes distance matrix
    is already binned) and sums the corresponding image intensities for each bin.
    Random sampling is used to avoid bias from pixel count variations.
    
    Args:
        image_path: Path to the input image file
        distance_path: Path to distance matrix file (already binned)
        output_path: Optional path to save the plot. If None, displays interactively.
        channel: Which channel to use from the image (default: 0)
        colormap: Matplotlib colormap name (default: 'viridis')
        normalize_to_t0: If True, normalize each distance bin by its T0 value to show relative changes (default: False)
        force_show: If True, display plot interactively even when output_path is specified (default: False)
        remove_first_n_bins: Number of first (closest) distance bins to remove from visualization (default: 5)
        max_measure_pixels: Maximum pixels to randomly sample per distance bin. 
                            Set to -1 to disable sampling and use all pixels (default: 20)
        random_seed: Seed for random sampling for reproducibility (default: 42, ignored if max_measure_pixels=-1)
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
        logging.info(f"No non-zero distance values found in {Path(distance_path).name}")
        return
    
    # Get sorted unique distance bins
    distance_bins = np.sort(np.unique(all_distance_values))
    num_distance_bins = len(distance_bins)
    
    # Apply maximum bin limit
    max_bins = 200
    if num_distance_bins > max_bins:
        logging.info(f"Found {num_distance_bins} distance bins, limiting to {max_bins} for visualization")
        # Keep only the first max_bins bins
        distance_bins = distance_bins[:max_bins]
        num_distance_bins = max_bins
    
    # Remove first N bins if requested
    if remove_first_n_bins > 0 and num_distance_bins > remove_first_n_bins:
        logging.info(f"Removing first {remove_first_n_bins} distance bins (closest to reference)")
        distance_bins = distance_bins[remove_first_n_bins:]
        num_distance_bins = len(distance_bins)
    
    logging.info(f"Using {num_distance_bins} distance bins for visualization")
    logging.info(f"Distance range: {distance_bins[0]:.2f} to {distance_bins[-1]:.2f}")
    
    # Check if sampling is enabled
    use_sampling = max_measure_pixels > 0
    if use_sampling:
        logging.info(f"Random sampling: max {max_measure_pixels} pixels per bin (seed={random_seed})")
    else:
        logging.info(f"Random sampling DISABLED (max_measure_pixels={max_measure_pixels}): using ALL pixels per bin")
    
    # Create 2D histogram: rows = distance bins, cols = timepoints
    # Use random sampling to avoid bias from pixel count variations (if enabled)
    # Also track pixel counts per bin
    heatmap_data = np.zeros((num_distance_bins, T))
    pixel_counts = np.zeros((num_distance_bins, T), dtype=int)
    sampled_pixel_counts = np.zeros((num_distance_bins, T), dtype=int)
    
    # Initialize random number generator with fixed seed for reproducibility (only if sampling enabled)
    rng = np.random.RandomState(random_seed) if use_sampling else None
    
    for t in range(T):
        # Get distance map and corresponding image data for this timepoint
        dist_t = dist.data[t, 0, :, :, :]
        img_t = img.data[t, channel, :, :, :]
        
        # For each distance bin value, randomly sample pixels and sum their intensities
        for i, bin_value in enumerate(distance_bins):
            bin_mask = dist_t == bin_value
            
            # Count total pixels in this bin
            pixel_counts[i, t] = np.sum(bin_mask)
            
            if pixel_counts[i, t] > 0:
                # Get intensities of all pixels in this bin
                intensities = img_t[bin_mask]
                
                if use_sampling:
                    # Randomly sample up to max_measure_pixels
                    n_pixels_to_sample = min(max_measure_pixels, len(intensities))
                    
                    # Random sampling without replacement
                    sampled_indices = rng.choice(len(intensities), size=n_pixels_to_sample, replace=False)
                    sampled_intensities = intensities[sampled_indices]
                    
                    # Sum the sampled intensities
                    heatmap_data[i, t] = np.sum(sampled_intensities)
                    sampled_pixel_counts[i, t] = n_pixels_to_sample
                else:
                    # Use all pixels (no sampling)
                    heatmap_data[i, t] = np.sum(intensities)
                    sampled_pixel_counts[i, t] = len(intensities)
            else:
                heatmap_data[i, t] = 0
                sampled_pixel_counts[i, t] = 0
    
    # Calculate statistics
    total_pixels_per_timepoint = np.sum(pixel_counts, axis=0)
    total_sampled_per_timepoint = np.sum(sampled_pixel_counts, axis=0)
    avg_pixels_per_bin = np.mean(pixel_counts, axis=1)
    avg_sampled_per_bin = np.mean(sampled_pixel_counts, axis=1)
    
    # Log pixel count statistics
    logging.info(f"Total pixels per timepoint: min={np.min(total_pixels_per_timepoint)}, "
                f"max={np.max(total_pixels_per_timepoint)}, "
                f"mean={np.mean(total_pixels_per_timepoint):.0f}")
    logging.info(f"Sampled pixels per timepoint: min={np.min(total_sampled_per_timepoint)}, "
                f"max={np.max(total_sampled_per_timepoint)}, "
                f"mean={np.mean(total_sampled_per_timepoint):.0f}")
    if use_sampling:
        logging.info(f"Average sampled pixels per bin: {np.mean(avg_sampled_per_bin):.1f} (max allowed: {max_measure_pixels})")
    else:
        logging.info(f"Average pixels per bin: {np.mean(avg_sampled_per_bin):.1f} (all pixels used, no sampling)")
    
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
    
    # Create DataFrame for seaborn with pixel count as additional column
    df = pd.DataFrame(
        heatmap_data,
        index=[f'{bin_val:.1f}' for bin_val in distance_bins],
        columns=[f'T{t}' for t in range(T)]
    )
    
    # Add average pixel count column for reference (will show in TSV but not in heatmap)
    df_with_counts = df.copy()
    df_with_counts.insert(0, 'Avg_N_Pixels', avg_pixels_per_bin.astype(int))
    
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
        
        # Create a copy of the dataframe with metadata and pixel counts
        csv_df = df_with_counts.copy()
        
        # Add metadata columns at the beginning
        csv_df.insert(0, 'Base_Filename', base_filename)
        csv_df.insert(1, 'Channel', channel)
        csv_df.insert(2, 'Normalized_to_T0', normalize_to_t0)
        if normalize_to_t0:
            csv_df.insert(3, 'Units', 'Intensity_Change_from_T0')
        else:
            csv_df.insert(3, 'Units', 'Sum_Image_Intensity')
        csv_df.insert(4, 'QC_Status', 'passed')  # Only QC-passed files are processed
        
        # Set index name for distance bins
        csv_df.index.name = 'Distance_Bin'
        
        # Save as TSV (tab-separated, better for Excel)
        csv_df.to_csv(tsv_path, sep='\t')
        logging.info(f"Saved TSV data to: {tsv_path}")
        
        # Also save pixel counts as separate TSV
        pixel_count_path = output_path.replace('.png', '_pixel_counts.tsv')
        pixel_df = pd.DataFrame(
            pixel_counts,
            index=[f'{bin_val:.1f}' for bin_val in distance_bins],
            columns=[f'T{t}' for t in range(T)]
        )
        pixel_df.insert(0, 'Base_Filename', base_filename)
        pixel_df.insert(1, 'QC_Status', 'passed')  # Only QC-passed files are processed
        pixel_df.index.name = 'Distance_Bin'
        pixel_df.to_csv(pixel_count_path, sep='\t')
        logging.info(f"Saved pixel count data to: {pixel_count_path}")
    
    
    # Auto-calculate figure size - add extra height for pixel count text
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
        height = min(max_height, max(8, num_distance_bins * 0.15)) + 1.5  # Extra space for pixel count text
        figsize = (width, height)
    except:
        width = max(10, T * 0.4)
        height = max(8, num_distance_bins * 0.15) + 1.5  # Extra space for pixel count text
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
    ax.set_ylabel('Distance (n̄ pixels per bin)', fontsize=12, fontweight='bold')
    
    img_name = Path(image_path).stem
    title_text = f'Image Intensity by Distance Over Time{title_suffix}\n{img_name}'
    title_text += f'\nAvg pixels/bin: {np.mean(avg_pixels_per_bin):.0f} (range: {np.min(avg_pixels_per_bin):.0f}-{np.max(avg_pixels_per_bin):.0f})'
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
    
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


    # TODO
    # The next goal is to quantify the signal. I am not sure how to proceed from here, 
    # but To me this heatmap looks like an image even if is not pixels directly. 
    # My instincts would be to 
    # 1) make some blurring/ smoothing of the data on distance vs time axis as if it was an image
    # 2) find the spread of the signal per frame. E.g., for each timepoint, find the distance range that contains 80% of the signal
    # 3) plot the upper cutoff and the lower cutoff on top of the heatmap.
    # 4) At some point in time, the signal should disapear, can we fisrt do the same as above just in time dimension instead of distance?
    # 5) plot the timepoint where 80% of the signal is lost as
    # OR we can do a segmentation of the heatmap, and calcualte the first and the second eigenvalue of the segmented overall. the height should tell us the distance spread and the width the time spread.
    # 6) plot the eigenvalues as function of time.

    # Print summary statistics
    total_intensity = np.sum(heatmap_data)
    logging.info(f"Total image intensity in heatmap: {total_intensity:.2e}")
    logging.info(f"Distance range: {distance_bins[0]:.2f} - {distance_bins[-1]:.2f}")
    logging.info(f"Number of distance bins: {num_distance_bins}")
    logging.info(f"Timepoints: {T}")


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


def quantify_signal_spread_and_decay(
    image_path: str,
    distance_path: str,
    output_path: str = None,
    channel: int = 0,
    colormap: str = 'viridis',
    remove_first_n_bins: int = 5,
    smooth_sigma_distance: float = 1.5,
    max_measure_pixels: int = 20,
    random_seed: int = 42,
    force_show: bool = False,
    mask_alpha: float = 0.5,
    region_grow_threshold: float = 0.5
) -> Dict:
    """
    Quantify signal spread (distance) and decay (time) using segmentation.
    
    This function:
    1. Uses robust random sampling to create unbiased heatmap data (10 runs, selects closest to mean)
    2. Smooths heatmap in DISTANCE DIMENSION ONLY (each timepoint processed independently)
    3. Applies region growing segmentation starting from peak signal (avoids over-segmentation from distant bright pixels)
    4. Grows region by including neighbors >= threshold % of running maximum intensity
    5. Measures center of mass, distance spread, and signal area per timepoint
    6. Plots signal trajectory and spread bounds over time
    7. Returns quantitative metrics
    
    Args:
        image_path: Path to the input image file
        distance_path: Path to distance matrix file
        output_path: Optional path to save plots. If None, displays interactively.
        channel: Which channel to use from the image (default: 0)
        colormap: Matplotlib colormap name (default: 'viridis')
        remove_first_n_bins: Number of first distance bins to remove (default: 5)
        smooth_sigma_distance: Gaussian smoothing sigma applied ONLY in distance dimension.
                               Each timepoint is smoothed independently. (default: 1.5)
        max_measure_pixels: Maximum pixels to randomly sample per distance bin. 
                            Set to -1 to disable sampling and use all pixels (default: 20)
        random_seed: Base seed for random sampling. Uses seeds random_seed to random_seed+9 
                     for 10 sampling runs, then selects the value closest to mean. 
                     (default: 42, ignored if max_measure_pixels=-1)
        force_show: Display plots even when saving (default: False)
        mask_alpha: Alpha transparency for mask overlay on heatmap (0=transparent, 1=opaque) (default: 0.5)
        region_grow_threshold: Fraction of running maximum for region growing (0.0-1.0).
                               Higher = more restrictive (e.g., 0.75 = 75% of max).
                               Lower = more permissive (e.g., 0.25 = 25% of max). (default: 0.5)
    
    Returns:
        Dictionary containing:
            - 'distance_peak': Center of mass per timepoint (intensity-weighted)
            - 'distance_spread_lower': Lower distance extent per timepoint
            - 'distance_spread_upper': Upper distance extent per timepoint
            - 'distance_spread': Total distance spread per timepoint
            - 'signal_area': Number of distance bins with signal per timepoint
            - 'temporal_intensity': Total intensity per timepoint
            - 'heatmap_smoothed': Smoothed heatmap data
            - 'segmentation_mask': Binary mask from segmentation
            - 'distance_bins': Distance bin values
            - 'timepoints': Timepoint indices
            - 'smooth_sigma_distance': Smoothing parameter used
            - 'signal_duration': Total number of timepoints with detectable signal
            - 'time_to_50_percent': Timepoint where intensity drops to 50% of T1
            - 'time_to_disappearance': Last timepoint with detectable signal
            - 'decay_rate': Exponential decay rate (1/timepoint)
            - 'intensity_T1': Intensity at T=1
            - 'intensity_T10': Intensity at T=10 (NaN if unavailable)
            - 'intensity_T20': Intensity at T=20 (NaN if unavailable)
    """
    logging.info(f"Quantifying signal spread and decay: {Path(image_path).name}")
    
    # Load image and distance data (reuse logic from plot_distance_time_heatmap)
    img = rp.load_tczyx_image(image_path)
    dist = rp.load_tczyx_image(distance_path)
    
    T, C, Z, Y, X = img.shape
    dT, dC, dZ, dY, dX = dist.shape
    
    # Validate dimensions
    if (Y, X) != (dY, dX):
        raise ValueError(f"Image and distance XY dimensions must match. Got image: ({Y}, {X}), distance: ({dY}, {dX})")
    if T != dT:
        raise ValueError(f"Image and distance must have same number of timepoints. Got image: {T}, distance: {dT}")
    
    # Get distance bins
    all_distance_values = []
    for t in range(T):
        data_t = dist.data[t, 0, :, :, :]
        unique_vals = np.unique(data_t[data_t > 0])
        if len(unique_vals) > 0:
            all_distance_values.extend(unique_vals)
    
    if len(all_distance_values) == 0:
        logging.info(f"No non-zero distance values found")
        return None
    
    distance_bins = np.sort(np.unique(all_distance_values))
    num_distance_bins = len(distance_bins)
    
    # Apply bin limits
    max_bins = 200
    if num_distance_bins > max_bins:
        distance_bins = distance_bins[:max_bins]
        num_distance_bins = max_bins
    
    if remove_first_n_bins > 0 and num_distance_bins > remove_first_n_bins:
        distance_bins = distance_bins[remove_first_n_bins:]
        num_distance_bins = len(distance_bins)
    
    logging.info(f"Using {num_distance_bins} distance bins")
    
    # Check if sampling is enabled
    use_sampling = max_measure_pixels > 0
    if use_sampling:
        logging.info(f"Robust random sampling: max {max_measure_pixels} pixels per bin")
        logging.info(f"Using 10 sampling runs (seeds {random_seed} to {random_seed+9}), selecting value closest to mean")
    else:
        logging.info(f"Random sampling DISABLED (max_measure_pixels={max_measure_pixels}): using ALL pixels per bin")
    
    # Create heatmap data with robust random sampling
    # Sample with 10 different seeds and use the value closest to mean
    # This makes measurements more robust to outliers in random sampling
    
    # Skip T=0 for quantification - only analyze T=1 onwards
    logging.info(f"Skipping T=0, analyzing timepoints T=1 to T={T-1}")
    T_quantify = T - 1  # Number of timepoints to quantify (excluding T0)
    
    n_sampling_runs = 10  # Number of different random seeds to try (only if sampling enabled)
    if use_sampling:
        logging.info(f"Using robust sampling: {n_sampling_runs} runs per bin, selecting value closest to mean")
    
    heatmap_data = np.zeros((num_distance_bins, T_quantify))
    
    for t in range(1, T):  # Start from T=1 instead of T=0
        dist_t = dist.data[t, 0, :, :, :]
        img_t = img.data[t, channel, :, :, :]
        
        for i, bin_value in enumerate(distance_bins):
            bin_mask = dist_t == bin_value
            
            if np.sum(bin_mask) > 0:
                # Get intensities of all pixels in this bin
                intensities = img_t[bin_mask]
                
                if use_sampling:
                    # Randomly sample with multiple seeds
                    n_pixels_to_sample = min(max_measure_pixels, len(intensities))
                    
                    # If we have very few pixels, just use all of them
                    if len(intensities) <= max_measure_pixels:
                        heatmap_data[i, t-1] = np.sum(intensities)
                    else:
                        # Sample with 10 different seeds
                        sampled_sums = []
                        for seed_offset in range(n_sampling_runs):
                            rng = np.random.RandomState(random_seed + seed_offset)
                            sampled_indices = rng.choice(len(intensities), size=n_pixels_to_sample, replace=False)
                            sampled_intensities = intensities[sampled_indices]
                            sampled_sums.append(np.sum(sampled_intensities))
                        
                        # Calculate mean of all samples
                        mean_sum = np.mean(sampled_sums)
                        
                        # Find the sample closest to the mean
                        closest_idx = np.argmin(np.abs(np.array(sampled_sums) - mean_sum))
                        heatmap_data[i, t-1] = sampled_sums[closest_idx]
                else:
                    # Use all pixels (no sampling)
                    heatmap_data[i, t-1] = np.sum(intensities)
            else:
                heatmap_data[i, t-1] = 0
    
    # === 1. SMOOTH THE HEATMAP IN DISTANCE DIMENSION ONLY ===
    # Smooth only in distance dimension (axis 0), not time dimension (axis 1)
    # This ensures each timepoint is processed completely independently
    # sigma parameter: [distance_sigma, time_sigma] where time_sigma=0 means NO temporal smoothing
    heatmap_smoothed = gaussian_filter(heatmap_data, sigma=[smooth_sigma_distance, 0])
    logging.info(f"Applied Gaussian smoothing in distance dimension only (σ_distance={smooth_sigma_distance}, σ_time=0)")
    logging.info("Each timepoint is processed independently - no temporal cross-contamination")
    
    # === 2. SEGMENTATION-BASED QUANTIFICATION ===
    # Use region growing from the peak signal instead of global thresholding
    # This avoids over-segmentation from bright pixels far from the initial signal
    from skimage.morphology import binary_dilation, binary_erosion, label
    from skimage.measure import regionprops
    from scipy.ndimage import binary_fill_holes
    from collections import deque
    
    logging.info("Using region growing segmentation from peak signal")
    logging.info(f"Heatmap shape: {heatmap_smoothed.shape} (distance × time)")
    logging.info(f"Heatmap range: [{np.min(heatmap_smoothed):.2f}, {np.max(heatmap_smoothed):.2f}]")
    
    # Find the starting point: maximum intensity in the heatmap
    # This should be at the first distance bin and early timepoint (where signal starts)
    max_idx = np.unravel_index(np.argmax(heatmap_smoothed), heatmap_smoothed.shape)
    seed_distance_idx, seed_time_idx = max_idx
    seed_value = heatmap_smoothed[seed_distance_idx, seed_time_idx]
    
    logging.info(f"Seed point: distance_idx={seed_distance_idx} (distance={distance_bins[seed_distance_idx]:.1f}), "
                f"time_idx={seed_time_idx} (T={seed_time_idx+1}), intensity={seed_value:.2f}")
    
    # Region growing algorithm with adaptive threshold AND connectivity constraint
    # Strategy:
    # 1. Allow threshold to update as we find brighter regions (adaptive to signal strength)
    # 2. Grow region using >= threshold% of running maximum
    # 3. After growing, use connected components to keep only pixels connected to seed
    # This prevents jumping to disconnected bright regions while being adaptive to signal intensity
    
    mask = np.zeros_like(heatmap_smoothed, dtype=bool)
    visited = np.zeros_like(heatmap_smoothed, dtype=bool)
    
    # Queue for region growing: stores (distance_idx, time_idx)
    queue = deque()
    queue.append((seed_distance_idx, seed_time_idx))
    visited[seed_distance_idx, seed_time_idx] = True
    mask[seed_distance_idx, seed_time_idx] = True
    
    # Track running maximum (updates as we explore)
    current_max = seed_value
    logging.info(f"Starting with threshold: {current_max * region_grow_threshold:.2f} ({region_grow_threshold*100:.0f}% of seed value {seed_value:.2f})")
    
    n_distance, n_time = heatmap_smoothed.shape
    
    # 4-connectivity neighbors (up, down, left, right in distance and time)
    neighbors_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    pixels_added = 0
    max_iterations = n_distance * n_time  # Safety limit
    iteration = 0
    
    while queue and iteration < max_iterations:
        d_idx, t_idx = queue.popleft()
        iteration += 1
        
        # Check all 4-connected neighbors
        for d_offset, t_offset in neighbors_offsets:
            new_d = d_idx + d_offset
            new_t = t_idx + t_offset
            
            # Check bounds
            if 0 <= new_d < n_distance and 0 <= new_t < n_time:
                # Skip if already visited
                if visited[new_d, new_t]:
                    continue
                
                visited[new_d, new_t] = True
                neighbor_value = heatmap_smoothed[new_d, new_t]
                
                # Update running maximum if this pixel is brighter
                if neighbor_value > current_max:
                    current_max = neighbor_value
                    logging.info(f"  Updated max to {current_max:.2f} at distance_idx={new_d}, time_idx={new_t} (T={new_t+1})")
                
                # Add to mask if >= threshold % of CURRENT running maximum
                threshold_value = current_max * region_grow_threshold
                if neighbor_value >= threshold_value:
                    mask[new_d, new_t] = True
                    queue.append((new_d, new_t))
                    pixels_added += 1
    
    logging.info(f"Region growing complete: {np.sum(mask)} pixels (seed + {pixels_added})")
    logging.info(f"Final running maximum: {current_max:.2f}")
    logging.info(f"Final threshold ({region_grow_threshold*100:.0f}% of max): {current_max * region_grow_threshold:.2f}")
    
    # === CONNECTIVITY FILTERING ===
    # Keep only pixels that are connected to the seed point
    # This prevents including disconnected bright regions
    from skimage.measure import label
    
    logging.info("Applying connected component filtering to ensure connectivity to seed...")
    labeled_mask = label(mask, connectivity=1)  # 4-connectivity (not diagonal)
    seed_label = labeled_mask[seed_distance_idx, seed_time_idx]
    
    if seed_label == 0:
        logging.warning("Seed point not labeled - this shouldn't happen!")
        connected_mask = mask
    else:
        # Keep only the component containing the seed
        connected_mask = (labeled_mask == seed_label)
        n_components = np.max(labeled_mask)
        pixels_removed = np.sum(mask) - np.sum(connected_mask)
        
        logging.info(f"  Found {n_components} connected components in initial mask")
        logging.info(f"  Kept component {seed_label} (connected to seed)")
        logging.info(f"  Removed {pixels_removed} disconnected pixels")
        logging.info(f"  Final mask: {np.sum(connected_mask)} pixels")
    
    # Replace mask with connectivity-filtered version
    mask = connected_mask
    
    # Optional: Apply light morphological cleanup
    # Fill small holes but don't dilate/erode to preserve the region-grown boundaries
    mask = binary_fill_holes(mask)
    
    logging.info(f"Final mask: {np.sum(mask)} pixels")
    
    # Check if we have any signal
    if np.sum(mask) == 0:
        logging.info("No signal found in region growing!")
        # Return empty results
        distance_peak = np.full(T_quantify, np.nan)
        distance_spread_lower = np.full(T_quantify, np.nan)
        distance_spread_upper = np.full(T_quantify, np.nan)
        distance_spread = np.full(T_quantify, np.nan)
        signal_area = np.zeros(T_quantify)
        temporal_intensity = np.sum(heatmap_smoothed, axis=0)
        
        # Empty decay metrics
        signal_duration = 0
        time_to_50_percent = 0
        time_to_disappearance = 0
        decay_rate = 0
        intensity_T1 = 0
        intensity_T10 = np.nan
        intensity_T20 = np.nan
        
        final_mask = mask
    else:
        # === 3. EXTRACT METRICS FROM SEGMENTATION PER TIMEPOINT ===
        # Arrays sized for T_quantify (excluding T0)
        distance_peak = np.zeros(T_quantify)          # Center of mass in distance dimension
        distance_spread_lower = np.zeros(T_quantify)  # Minimum distance extent
        distance_spread_upper = np.zeros(T_quantify)  # Maximum distance extent
        distance_spread = np.zeros(T_quantify)        # Total distance spread
        
        # Use the region-grown mask directly
        final_mask = mask
        signal_area = np.zeros(T_quantify)            # Number of pixels in mask per timepoint
        temporal_intensity = np.zeros(T_quantify)     # Total intensity per timepoint
        
        for t in range(T_quantify):
            # Get mask column for this timepoint
            mask_column = final_mask[:, t]
            
            if np.sum(mask_column) > 0:
                # Find indices of True values (signal present)
                signal_indices = np.where(mask_column)[0]
                
                # Distance extent
                distance_spread_lower[t] = distance_bins[signal_indices[0]]
                distance_spread_upper[t] = distance_bins[signal_indices[-1]]
                distance_spread[t] = distance_spread_upper[t] - distance_spread_lower[t]
                
                # Center of mass (weighted by intensity)
                intensities_in_mask = heatmap_smoothed[mask_column, t]
                if np.sum(intensities_in_mask) > 0:
                    distances_in_mask = distance_bins[signal_indices]
                    distance_peak[t] = np.average(distances_in_mask, weights=intensities_in_mask)
                else:
                    # If no intensity, use geometric center
                    distance_peak[t] = np.mean(distance_bins[signal_indices])
                
                # Signal area (number of distance bins with signal)
                signal_area[t] = len(signal_indices)
                
                # Total intensity in masked region
                temporal_intensity[t] = np.sum(heatmap_smoothed[mask_column, t])
                
                logging.info(f"T={t+1}: peak={distance_peak[t]:.2f}, "
                           f"spread={distance_spread[t]:.2f} "
                           f"({distance_spread_lower[t]:.2f} - {distance_spread_upper[t]:.2f}), "
                           f"area={signal_area[t]:.0f} bins")
            else:
                # No signal at this timepoint
                distance_peak[t] = np.nan
                distance_spread_lower[t] = np.nan
                distance_spread_upper[t] = np.nan
                distance_spread[t] = np.nan
                signal_area[t] = 0
                temporal_intensity[t] = 0
        
        logging.info(f"Segmentation-based quantification complete for {T_quantify} timepoints (T=1 to T={T-1})")
    
    # === CALCULATE TEMPORAL DECAY METRICS ===
    # Calculate signal duration and decay characteristics
    
    # 1. Signal duration (number of timepoints with detectable signal)
    signal_present = signal_area > 0  # Boolean array
    signal_duration = np.sum(signal_present)
    
    # 2. Time to 50% intensity (half-life)
    if len(temporal_intensity) > 0 and np.max(temporal_intensity) > 0:
        initial_intensity = temporal_intensity[0]  # T=1 intensity
        half_intensity = initial_intensity * 0.5
        
        # Find first timepoint where intensity drops below 50%
        below_half = temporal_intensity < half_intensity
        if np.any(below_half):
            time_to_50_percent = np.argmax(below_half) + 1  # +1 because T=1 is index 0
        else:
            time_to_50_percent = T_quantify  # Signal never drops below 50%
    else:
        time_to_50_percent = 0
    
    # 3. Time to disappearance (last timepoint with signal)
    if signal_duration > 0:
        # Find last timepoint where signal is present
        last_signal_idx = np.max(np.where(signal_present)[0])
        time_to_disappearance = last_signal_idx + 1  # +1 because T=1 is index 0
    else:
        time_to_disappearance = 0
    
    # 4. Decay rate (linear fit of log(intensity) over time)
    # Define timepoints array for quantification (T=1 to T=T_quantify)
    timepoints_arr = np.arange(1, T)  # Start from T=1 (T=0 is excluded)
    
    if len(temporal_intensity) > 1 and np.sum(temporal_intensity > 0) > 1:
        # Use only timepoints with signal for decay rate calculation
        valid_times = timepoints_arr[temporal_intensity > 0]
        valid_intensities = temporal_intensity[temporal_intensity > 0]
        
        if len(valid_times) > 1:
            # Linear fit: log(I) = -k*t + log(I0), where k is decay rate
            log_intensities = np.log(valid_intensities + 1e-10)  # Add small value to avoid log(0)
            
            # Fit line using polyfit
            coeffs = np.polyfit(valid_times, log_intensities, 1)
            decay_rate = -coeffs[0]  # Negative slope gives decay rate
        else:
            decay_rate = 0
    else:
        decay_rate = 0
    
    # 5. Intensity at specific timepoints (T=1, T=10, T=20 if available)
    intensity_T1 = temporal_intensity[0] if len(temporal_intensity) > 0 else 0
    intensity_T10 = temporal_intensity[9] if len(temporal_intensity) >= 10 else np.nan
    intensity_T20 = temporal_intensity[19] if len(temporal_intensity) >= 20 else np.nan
    
    logging.info(f"Temporal decay metrics:")
    logging.info(f"  Signal duration: {signal_duration} timepoints")
    logging.info(f"  Time to 50% intensity: T={time_to_50_percent}")
    logging.info(f"  Time to disappearance: T={time_to_disappearance}")
    logging.info(f"  Decay rate: {decay_rate:.4f} (1/timepoint)")
    logging.info(f"  Initial intensity (T=1): {intensity_T1:.2e}")
    
    # final_mask is already set above (either from region growing or empty)
    
    # === 4. CREATE VISUALIZATION ===
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Smoothed heatmap data with segmentation mask overlay
    ax1 = fig.add_subplot(gs[0, :])
    
    # Show the SMOOTHED heatmap data (what segmentation was performed on)
    df_smooth = pd.DataFrame(
        heatmap_smoothed,
        index=[f'{b:.1f}' for b in distance_bins],
        columns=[f'T{t+1}' for t in range(T_quantify)]  # T1, T2, T3, ... (no T0)
    )
    
    # Create the heatmap (use the colormap argument passed to function)
    sns.heatmap(df_smooth, annot=False, cmap=colormap, ax=ax1, cbar_kws={'label': 'Intensity (smoothed)'})
    ax1.invert_yaxis()
    
    # Overlay the segmentation mask with semi-transparent white color
    # Create a mask overlay image
    mask_overlay = np.zeros((*final_mask.shape, 4))  # RGBA
    mask_overlay[final_mask] = [1, 1, 1, mask_alpha]  # White with alpha transparency
    
    # Display mask overlay on top of heatmap
    ax1.imshow(mask_overlay, aspect='auto', extent=[0, T_quantify, 0, num_distance_bins], 
              origin='upper', interpolation='nearest')
    
    # Overlay segmentation metrics (center and spread lines)
    # Convert distance values to row indices for plotting
    distance_to_idx_func = lambda d: np.argmin(np.abs(distance_bins - d))
    
    valid_timepoints = [t for t in range(T_quantify) if not np.isnan(distance_peak[t])]
    
    if len(valid_timepoints) > 0:
        peak_indices = [distance_to_idx_func(distance_peak[t]) for t in valid_timepoints]
        lower_indices = [distance_to_idx_func(distance_spread_lower[t]) for t in valid_timepoints]
        upper_indices = [distance_to_idx_func(distance_spread_upper[t]) for t in valid_timepoints]
        
        # Plot peak trajectory (center of mass)
        ax1.plot(np.array(valid_timepoints) + 0.5, 
                np.array(peak_indices) + 0.5, 
                'r-', linewidth=3, label='Signal Center', marker='o', markersize=5)
        
        # Plot spread bounds (from segmentation)
        ax1.plot(np.array(valid_timepoints) + 0.5, 
                np.array(lower_indices) + 0.5, 
                'cyan', linestyle='--', linewidth=2, label='Signal extent')
        ax1.plot(np.array(valid_timepoints) + 0.5, 
                np.array(upper_indices) + 0.5, 
                'cyan', linestyle='--', linewidth=2)
        
        ax1.legend(loc='upper right', fontsize=10)
    
    ax1.set_title(f'Smoothed Heatmap with Segmentation Mask Overlay (α={mask_alpha}, σ_distance={smooth_sigma_distance})', 
                 fontweight='bold', fontsize=14)
    ax1.set_xlabel('Timepoint', fontweight='bold')
    ax1.set_ylabel('Distance', fontweight='bold')
    
    # Subplot 2: Peak position and spread bounds over time
    ax2 = fig.add_subplot(gs[1, :])
    # timepoints_arr already defined earlier for decay rate calculation
    
    # Fill between spread bounds
    ax2.fill_between(timepoints_arr, distance_spread_lower, distance_spread_upper, 
                     alpha=0.2, color='blue', label='Signal extent')
    
    # Plot peak trajectory (center of mass)
    ax2.plot(timepoints_arr, distance_peak, 'r-', linewidth=3, 
            label='Signal Center (CoM)', marker='o', markersize=6)
    
    # Plot spread bounds
    ax2.plot(timepoints_arr, distance_spread_lower, 'b--', linewidth=2, 
            label='Lower bound')
    ax2.plot(timepoints_arr, distance_spread_upper, 'g--', linewidth=2, 
            label='Upper bound')
    
    ax2.set_xlabel('Timepoint', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Distance', fontweight='bold', fontsize=12)
    ax2.set_title('Signal Center and Spread Bounds Over Time (T0 excluded)', fontweight='bold', fontsize=14)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Distance spread width over time
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(timepoints_arr, distance_spread, 'purple', linewidth=2, marker='s', markersize=5)
    ax3.set_xlabel('Timepoint', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Distance Spread', fontweight='bold', fontsize=12)
    ax3.set_title('Signal Distance Spread Over Time (T0 excluded)', fontweight='bold', fontsize=13)
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Temporal intensity and signal area
    ax4 = fig.add_subplot(gs[2, 1])
    ax4_twin = ax4.twinx()
    
    # Plot total intensity
    line1 = ax4.plot(timepoints_arr, temporal_intensity, 'b-', linewidth=2, 
                     label='Total Intensity', marker='o', markersize=4)
    ax4.set_xlabel('Timepoint', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Total Intensity', color='b', fontweight='bold', fontsize=12)
    ax4.tick_params(axis='y', labelcolor='b')
    
    # Plot signal area (number of distance bins)
    line2 = ax4_twin.plot(timepoints_arr, signal_area, 'r--', linewidth=2, 
                          label='Signal Area (bins)', marker='s', markersize=4)
    ax4_twin.set_ylabel('Signal Area (bins)', color='r', fontweight='bold', fontsize=12)
    ax4_twin.tick_params(axis='y', labelcolor='r')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='best', fontsize=9)
    
    ax4.set_title('Temporal Intensity and Signal Area (T0 excluded)', fontweight='bold', fontsize=13)
    ax4.grid(True, alpha=0.3)
    
    # Calculate summary stats
    mean_spread = np.nanmean(distance_spread)
    mean_peak = np.nanmean(distance_peak)
    mean_area = np.nanmean(signal_area)
    
    summary_text = f"""
    Summary Statistics (Segmentation-Based):
    
    Peak Position (Center of Mass):
    • Mean: {mean_peak:.2f}
    • Range: {np.nanmin(distance_peak):.2f} - {np.nanmax(distance_peak):.2f}
    
    Distance Spread:
    • Mean: {mean_spread:.2f}
    • Max: {np.nanmax(distance_spread):.2f}
    • Min: {np.nanmin(distance_spread):.2f}
    
    Signal Area:
    • Mean: {mean_area:.1f} bins
    • Max: {np.nanmax(signal_area):.0f} bins
    
    Processing:
    • Segmentation: Region growing from peak
    • Threshold: {region_grow_threshold*100:.0f}% of running maximum
    • Morphology: Fill holes only
    • Distance smoothing σ: {smooth_sigma_distance} (time σ: 0)
    • Distance bins: {num_distance_bins}
    • Removed first N bins: {remove_first_n_bins}
    • Total timepoints analyzed: {T_quantify} (T=1 to T={T-1})
    • T0 excluded from analysis
    • Timepoints processed independently
    """
    
    # Add summary text annotation to figure
    fig.text(0.02, 0.02, summary_text, fontsize=9, family='monospace',
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Overall title
    img_name = Path(image_path).stem
    fig.suptitle(f'Segmentation-Based Analysis: Signal Spread and Center Tracking (T0 Excluded)\n{img_name}', 
                fontsize=15, fontweight='bold')
    
    # Save or show
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved quantification plot to: {output_path}")
        
        # Extract experimental group from filename
        experimental_group = extract_experimental_group(image_path)
        logging.info(f"Experimental group: {experimental_group}")
        
        # Save metrics as TSV
        metrics_path = output_path.replace('.png', '_metrics.tsv')
        metrics_df = pd.DataFrame({
            'Experimental_Group': [experimental_group] * len(timepoints_arr),
            'Filename': [Path(image_path).stem] * len(timepoints_arr),
            'Timepoint': timepoints_arr,
            'QC_Status': ['passed'] * len(timepoints_arr),  # Only QC-passed files are processed
            'Center_Distance': distance_peak,
            'Spread_Lower': distance_spread_lower,
            'Spread_Upper': distance_spread_upper,
            'Distance_Spread': distance_spread,
            'Signal_Area_Bins': signal_area,
            'Total_Intensity': temporal_intensity
        })
        metrics_df.to_csv(metrics_path, sep='\t', index=False)
        logging.info(f"Saved metrics to: {metrics_path}")
        
        # Save temporal decay summary (single row per file)
        decay_summary_path = output_path.replace('.png', '_decay_summary.tsv')
        decay_df = pd.DataFrame({
            'Experimental_Group': [experimental_group],
            'Filename': [Path(image_path).stem],
            'QC_Status': ['passed'],  # Only QC-passed files are processed
            'Signal_Duration_Timepoints': [signal_duration],
            'Time_To_50_Percent': [time_to_50_percent],
            'Time_To_Disappearance': [time_to_disappearance],
            'Decay_Rate_Per_Timepoint': [decay_rate],
            'Initial_Intensity_T1': [intensity_T1],
            'Intensity_T10': [intensity_T10],
            'Intensity_T20': [intensity_T20]
        })
        decay_df.to_csv(decay_summary_path, sep='\t', index=False)
        logging.info(f"Saved decay summary to: {decay_summary_path}")
        
        # Save smoothed heatmap as TIFF in TCZYX order
        heatmap_tif_path = output_path.replace('_quantification.png', '_heatmap.tif')
        # heatmap_smoothed is 2D (distance, time), transpose to (time, distance) and expand to 5D TCZYX: (1, 1, 1, time, distance)
        # Flip Y axis to match the heatmap visualization (which uses invert_yaxis)
        # This makes Y=timepoints (flipped) and X=distance bins
        heatmap_5d = np.flip(heatmap_smoothed.T, axis=0)[np.newaxis, np.newaxis, np.newaxis, :, :]
        rp.save_tczyx_image(heatmap_5d, heatmap_tif_path)
        logging.info(f"Saved heatmap TIFF to: {heatmap_tif_path}")
        
        # Save segmentation mask as TIFF
        mask_tif_path = output_path.replace('_quantification.png', '_mask.tif')
        mask_5d = np.flip(final_mask.T.astype(np.uint8) * 255, axis=0)[np.newaxis, np.newaxis, np.newaxis, :, :]
        rp.save_tczyx_image(mask_5d, mask_tif_path)
        logging.info(f"Saved segmentation mask TIFF to: {mask_tif_path}")
        
        if force_show:
            plt.show()
        else:
            plt.close()
    else:
        plt.show()
    
    # Return quantitative results
    return {
        'distance_peak': distance_peak,
        'distance_spread_lower': distance_spread_lower,
        'distance_spread_upper': distance_spread_upper,
        'distance_spread': distance_spread,
        'signal_area': signal_area,
        'temporal_intensity': temporal_intensity,
        'heatmap_smoothed': heatmap_smoothed,
        'segmentation_mask': final_mask,
        'distance_bins': distance_bins,
        'timepoints': timepoints_arr,
        'smooth_sigma_distance': smooth_sigma_distance,
        # Temporal decay metrics
        'signal_duration': signal_duration,
        'time_to_50_percent': time_to_50_percent,
        'time_to_disappearance': time_to_disappearance,
        'decay_rate': decay_rate,
        'intensity_T1': intensity_T1,
        'intensity_T10': intensity_T10,
        'intensity_T20': intensity_T20
    }


def process_single_file(
    basename: str,
    complete_pairs: Dict,
    args: argparse.Namespace
) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single file for quantification (used in parallel processing).
    
    This function is designed to be called in parallel for multiple files.
    Always runs quantification analysis on the image-distance pair.
    
    Args:
        basename: Base filename key
        complete_pairs: Dictionary mapping basenames to file paths (uses 'input' and 'distance' keys)
        args: Command-line arguments namespace
    
    Returns:
        Tuple of (basename, success, error_message)
    """
    img_path = complete_pairs[basename]['input']
    dist_path = complete_pairs[basename]['distance']
    
    if args.output_folder:
        quant_output_name = f"{Path(img_path).stem}_quantification.png"
        quant_output = os.path.join(args.output_folder, quant_output_name)
    else:
        quant_output = None
    
    try:
        # In parallel mode, we always save and never show
        logging.info(f"Quantifying {Path(img_path).name}")
        quantify_signal_spread_and_decay(
            img_path,
            dist_path,
            output_path=quant_output,
            channel=args.channel,
            colormap=args.colormap,
            remove_first_n_bins=args.remove_first_n_bins,
            smooth_sigma_distance=args.smooth_sigma_distance,
            max_measure_pixels=args.max_measure_pixels,
            force_show=False,  # Never show in parallel mode
            mask_alpha=args.mask_alpha,
            region_grow_threshold=args.region_grow_threshold
        )
        
        return (basename, True, None)
    
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error processing {basename}: {error_msg}")
        return (basename, False, error_msg)
        return (basename, False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description='Quantify signal spread (distance) and decay (time) from distance matrices. Always performs quantification analysis.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic quantification with QC filtering
  python quantify_distance_heatmap.py --input-search-pattern "./input/*.tif" --distance-search-pattern "./distances/*_geodesic.tif" --qc-key nuc_segmentation --output-folder ./quantification
  
  # Display plots interactively even when saving
  python quantify_distance_heatmap.py --input-search-pattern "./input/*.tif" --distance-search-pattern "./distances/*_geodesic.tif" --qc-key track_completeness --output-folder ./quantification --force-show
  
  # Process all files in parallel (DEFAULT)
  python quantify_distance_heatmap.py --input-search-pattern "./input/*.tif" --distance-search-pattern "./distances/*_geodesic.tif" --qc-key nuc_segmentation --output-folder ./quantification --mode all
  
  # Process specific groups (e.g., 2 samples per experimental group)
  python quantify_distance_heatmap.py --input-search-pattern "./input/*.tif" --distance-search-pattern "./distances/*_geodesic.tif" --qc-key nuc_segmentation --output-folder ./quantification --mode group2
  
  # Recursive search with folder structure collapsing
  python quantify_distance_heatmap.py --input-search-pattern "./data/**/*.tif" --distance-search-pattern "./distances/**/*_geodesic.tif" --qc-key nuc_segmentation --output-folder ./quantification
  
  # Customize smoothing and segmentation thresholds
  python quantify_distance_heatmap.py --input-search-pattern "./input/*.tif" --distance-search-pattern "./distances/*_geodesic.tif" --qc-key nuc_segmentation --smooth-sigma-distance 2.0 --region-grow-threshold 0.75 --output-folder ./quantification
  
  # Sequential processing (disable parallel)
  python quantify_distance_heatmap.py --input-search-pattern "./input/*.tif" --distance-search-pattern "./distances/*_geodesic.tif" --qc-key nuc_segmentation --output-folder ./quantification --no-parallel
        """
    )
    
    parser.add_argument('--input-search-pattern', type=str, required=True,
                       help='Glob pattern for input images. Use "**" for recursive search (e.g., "./data/**/*.tif")')
    parser.add_argument('--distance-search-pattern', type=str, required=True,
                       help='Glob pattern for distance matrix images. Use "**" for recursive search (e.g., "./distances/**/*_geodesic.tif")')
    parser.add_argument('--qc-key', type=str, required=True,
                       help='QC key to use for filtering (e.g., "nuc_segmentation", "track_completeness"). Images with failed status will be excluded.')
    parser.add_argument('--output-folder', type=str, default=None,
                       help='Folder to save plots and quantification results. If not specified, displays interactively.')
    parser.add_argument('--mode', type=str, default='all',
                       help='File selection mode: "all" = all files, "examples[N]" = first, middle, last N files, '
                            '"first[N]" = first N files, "random[N]" = N random files, '
                            '"group[N]" = N samples per experimental group (default: all)')
    parser.add_argument('--force-show', action='store_true',
                       help='Display plots interactively even when saving to output folder')
    parser.add_argument('--channel', type=int, default=0,
                       help='Which channel to use for quantification (default: 0)')
    parser.add_argument('--colormap', type=str, default='viridis',
                       help='Matplotlib colormap for heatmaps (default: viridis)')
    parser.add_argument('--remove-first-n-bins', type=int, default=5,
                       help='Remove first N distance bins (closest to reference) from analysis (default: 5)')
    parser.add_argument('--smooth-sigma-distance', type=float, default=1.5,
                       help='Gaussian smoothing sigma for distance dimension ONLY. '
                            'Each timepoint is processed independently (no temporal smoothing). (default: 1.5)')
    parser.add_argument('--max-measure-pixels', type=int, default=20,
                       help='Maximum number of pixels to randomly sample per distance bin (default: 20). '
                            'Makes measurement less sensitive to pixel count variations. Use -1 for all pixels.')
    parser.add_argument('--mask-alpha', type=float, default=0.5,
                       help='Alpha transparency for mask overlay on heatmap (0=transparent, 1=opaque) (default: 0.5)')
    parser.add_argument('--region-grow-threshold', type=float, default=0.5,
                       help='Threshold for region growing segmentation (0.0-1.0). Grows region from peak by including '
                            'neighbors >= threshold × running maximum intensity. Higher = more restrictive (e.g., 0.75), '
                            'lower = more permissive (e.g., 0.25). (default: 0.5)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing. By default, files are processed in parallel when saving to output folder.')
    parser.add_argument('--n-jobs', type=int, default=None,
                       help='Number of parallel jobs to run. Default: use all available CPU cores. Ignored if --no-parallel is set.')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("="*80)
    logging.info("Distance Matrix Quantification with QC Filtering")
    logging.info("="*80)
    
    # Determine if recursive search is requested
    search_subfolders = '**' in args.input_search_pattern or '**' in args.distance_search_pattern
    
    # Both patterns are required - build search patterns dict
    patterns = {
        'input': args.input_search_pattern,
        'distance': args.distance_search_pattern
    }
    
    logging.info(f"Searching for files with patterns:")
    logging.info(f"  Input:    {args.input_search_pattern}")
    logging.info(f"  Distance: {args.distance_search_pattern}")
    logging.info(f"  QC Key:   {args.qc_key}")
    
    # Get grouped files - this is our primary file discovery
    try:
        grouped_files = rp.get_grouped_files_to_process(patterns, search_subfolders)
    except Exception as e:
        logging.error(f"Error finding files: {e}")
        return 1
    
    if not grouped_files:
        logging.error(f"No files found matching patterns")
        return 1
    
    logging.info(f"Found {len(grouped_files)} file pairs")
    
    # Filter to only complete pairs (both image and distance present)
    complete_pairs = {
        basename: files 
        for basename, files in grouped_files.items() 
        if 'input' in files and 'distance' in files
    }
    
    if not complete_pairs:
        logging.error("No complete image-distance pairs found!")
        return 1
    
    files_with_both = len(complete_pairs)
    files_missing = len(grouped_files) - files_with_both
    logging.info(f"  {files_with_both} with both input and distance, {files_missing} incomplete")
    
    # ============================================================================
    # QC FILTERING - CRITICAL STEP
    # ============================================================================
    # Filter out files that failed QC checks BEFORE any processing or plotting.
    # This ensures that failed QC files are NEVER used in quantification or plots.
    # All TSV outputs will have QC_Status='passed' since only passed files proceed.
    # ============================================================================
    
    # Import QC filtering function from plot_segmentation_contours
    from plot_segmentation_contours import filter_by_qc_status
    
    # Filter by QC status (exclude failed) - need to adapt keys to match
    # Convert 'input' key to what filter expects
    qc_grouped_files = {
        basename: {'input': files['input'], 'mask': files.get('distance')}
        for basename, files in complete_pairs.items()
    }
    
    filtered_files = filter_by_qc_status(qc_grouped_files, args.qc_key)
    
    if not filtered_files:
        logging.error(f"No files remaining after QC filtering")
        return 1
    
    # Convert back to distance key naming
    complete_pairs = {
        basename: {'input': files['input'], 'distance': files['mask']}
        for basename, files in filtered_files.items()
    }
    
    logging.info(f"After QC filtering: {len(complete_pairs)} pairs")
    
    # Parse mode for file selection
    mode = str(args.mode).lower()  # Convert to string in case YAML passes boolean
    import re
    mode_match = re.match(r'(all|examples|first|random|group)(\d+)?$', mode)
    if not mode_match:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'all', 'examples[N]', 'first[N]', 'random[N]', or 'group[N]'")
    
    mode_type = mode_match.group(1)
    mode_count = int(mode_match.group(2)) if mode_match.group(2) else (1 if mode_type == 'group' else 3)
    
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
    
    # Select basenames to process
    basenames = sorted(complete_pairs.keys())
    
    # If group mode, select N samples per experimental group
    if mode_type == 'group':
        from collections import defaultdict
        
        # Group basenames by experimental group
        groups = defaultdict(list)
        for basename in basenames:
            # Extract group from the input file path
            img_path = complete_pairs[basename]['input']
            group = extract_experimental_group(img_path)
            groups[group].append(basename)
        
        logging.info(f"Found {len(groups)} experimental groups")
        
        selected_basenames = []
        for group_name, group_basenames in sorted(groups.items()):
            n_available = len(group_basenames)
            n_select = min(mode_count, n_available)
            
            # Select evenly spaced samples from the group (using 'examples' logic)
            if n_available <= mode_count:
                selected = group_basenames
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
                selected = [group_basenames[i] for i in indices]
            
            selected_basenames.extend(selected)
            logging.info(f"  Group '{group_name}': selected {len(selected)}/{n_available} samples")
    else:
        # Standard selection modes
        selected_basenames = select_basenames(basenames, mode_type, mode_count)
    
    logging.info(f"Quantifying {len(selected_basenames)} file pair(s) (mode: {args.mode})")
    
    # Determine if we should use parallel processing
    # Parallel is DEFAULT when: saving to folder AND not displaying plots
    # Disabled when: --no-parallel flag OR no output folder OR displaying plots
    use_parallel = (
        not args.no_parallel and 
        args.output_folder and 
        not args.force_show
    )
    
    if use_parallel:
        # Parallel processing (DEFAULT for batch saves)
        n_jobs = args.n_jobs if args.n_jobs is not None else cpu_count()
        n_jobs = min(n_jobs, len(selected_basenames))  # Don't use more cores than files
        
        logging.info(f"Processing {len(selected_basenames)} files in PARALLEL using {n_jobs} CPU cores")
        logging.info("Parallel mode: plots will be saved but not displayed")
        
        # Create partial function with fixed arguments
        worker_func = partial(process_single_file, complete_pairs=complete_pairs, args=args)
        
        # Process files in parallel
        with Pool(processes=n_jobs) as pool:
            results = pool.map(worker_func, selected_basenames)
        
        # Report results
        successes = sum(1 for _, success, _ in results if success)
        failures = sum(1 for _, success, _ in results if not success)
        
        logging.info(f"Parallel processing complete: {successes} succeeded, {failures} failed")
        
        if failures > 0:
            logging.warning("Failed files:")
            for basename, success, error in results:
                if not success:
                    logging.warning(f"  {basename}: {error}")
    
    else:
        # Sequential processing
        if not args.output_folder:
            logging.info(f"Processing {len(selected_basenames)} files sequentially (no output folder, displaying plots)")
        elif args.force_show:
            logging.info(f"Processing {len(selected_basenames)} files sequentially (displaying plots)")
        elif args.no_parallel:
            logging.info(f"Processing {len(selected_basenames)} files sequentially (--no-parallel flag set)")
        else:
            logging.info(f"Processing {len(selected_basenames)} files sequentially")
        
        for basename in selected_basenames:
            img_path = complete_pairs[basename]['input']
            dist_path = complete_pairs[basename]['distance']
            
            if args.output_folder:
                quant_output_name = f"{Path(img_path).stem}_quantification.png"
                quant_output = os.path.join(args.output_folder, quant_output_name)
            else:
                quant_output = None
            
            try:
                logging.info(f"Quantifying {Path(img_path).name}")
                quantify_signal_spread_and_decay(
                    img_path,
                    dist_path,
                    output_path=quant_output,
                    channel=args.channel,
                    colormap=args.colormap,
                    remove_first_n_bins=args.remove_first_n_bins,
                    smooth_sigma_distance=args.smooth_sigma_distance,
                    max_measure_pixels=args.max_measure_pixels,
                    force_show=args.force_show,
                    mask_alpha=args.mask_alpha,
                    region_grow_threshold=args.region_grow_threshold
                )
            except Exception as e:
                logging.error(f"Error processing {Path(img_path).name}: {e}")
                continue
    
    logging.info("="*80)
    logging.info("Processing complete!")
    logging.info("="*80)
    return 0


if __name__ == "__main__":
    sys.exit(main())

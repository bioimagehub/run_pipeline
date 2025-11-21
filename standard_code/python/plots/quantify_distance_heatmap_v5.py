"""
Quantify signal spread and decay from distance heatmap TSV files (v5).

Global thresholding version: uses a single percentile threshold computed from all images after T0 subtraction.

This module:
1. Loads pre-computed heatmap data from TSV files
2. Smooths heatmap in distance dimension
3. Subtracts T0 background to remove static artifacts
4. Collects all positive pixels across all images
5. Computes global percentile threshold
6. Applies DBSCAN clustering to segment signal from noise using global threshold
7. Calculates simple, robust metrics:
   - Maximum distance reached per timepoint
   - Time to signal disappearance
8. Saves metrics to TSV for downstream analysis

Author: BIPHUB - Bioimage Informatics Hub, University of Oslo
License: MIT
"""
import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
import glob
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# Local imports
try:
    from .. import bioimage_pipeline_utils as rp
except ImportError:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import bioimage_pipeline_utils as rp


def get_experiment_info_from_filename(filename: str, split: str = "__", colname: list[str] = ["Group"]) -> Dict[str, str]:
    """Extract experimental metadata from filename using split character."""
    base = Path(filename).stem
    parts = base.split(split)
    
    result = {}
    for i, name in enumerate(colname):
        if i < len(parts):
            result[name] = parts[i]
        else:
            result[name] = ""
    
    return result


def subtract_t0_background(heatmap_data: np.ndarray, quiet: bool = False) -> np.ndarray:
    """
    Subtract T0 background from all timepoints.
    
    Args:
        heatmap_data: Heatmap (distance × timepoints) INCLUDING T0
        quiet: Suppress logging
    
    Returns:
        Background-subtracted heatmap (T0 values set to 0, rest subtracted)
    """
    def log_info(msg):
        if not quiet:
            logging.info(msg)
    
    n_distance, n_time = heatmap_data.shape
    
    if n_time < 2:
        logging.warning("Only one timepoint - cannot perform T0 subtraction")
        return heatmap_data.copy()
    
    # Ensure float64 dtype
    heatmap_data = heatmap_data.astype(np.float64)
    
    log_info("=" * 60)
    log_info("T0 BACKGROUND SUBTRACTION")
    log_info("=" * 60)
    
    t0_values = heatmap_data[:, 0:1]  # Keep as 2D
    
    log_info(f"T0 intensity range: [{np.min(t0_values):.2f}, {np.max(t0_values):.2f}]")
    
    heatmap_subtracted = heatmap_data - t0_values
    heatmap_subtracted = np.maximum(heatmap_subtracted, 0)  # Clip negatives
    
    original_sum = np.sum(heatmap_data[:, 1:])
    subtracted_sum = np.sum(heatmap_subtracted[:, 1:])
    removed_pct = ((original_sum - subtracted_sum) / original_sum * 100) if original_sum > 0 else 0
    
    log_info(f"Background removed: {removed_pct:.2f}% of original signal")
    log_info("=" * 60)
    
    return heatmap_subtracted


def quantify_signal_from_mask(
    signal_mask: np.ndarray,
    distance_bins: np.ndarray,
    exclude_t0: bool = True
) -> pd.DataFrame:
    """
    Calculate simple, robust quantification metrics from segmentation mask.
    
    For each timepoint, calculates:
    - Maximum distance reached by signal
    - Whether signal is present (binary)
    
    Args:
        signal_mask: Boolean mask (distance × timepoints) INCLUDING T0
        distance_bins: Distance bin values
        exclude_t0: Skip T0 in output (default: True)
    
    Returns:
        DataFrame with columns: Timepoint, Max_Distance, Signal_Present, Time_Since_Disappearance
    """
    n_distance, n_time = signal_mask.shape
    
    # Determine timepoint range
    if exclude_t0 and n_time > 1:
        start_t = 1
        timepoint_labels = list(range(1, n_time))
    else:
        start_t = 0
        timepoint_labels = list(range(0, n_time))
    
    metrics = []
    
    for t_idx in range(start_t, n_time):
        mask_t = signal_mask[:, t_idx]
        
        if np.sum(mask_t) > 0:
            # Signal present - find max distance
            distances_with_signal = distance_bins[mask_t]
            max_distance = np.max(distances_with_signal)
            signal_present = True
        else:
            # No signal
            max_distance = 0.0
            signal_present = False
        
        metrics.append({
            'Timepoint': t_idx if not exclude_t0 else t_idx,  # Keep original indexing
            'Max_Distance': max_distance,
            'Signal_Present': signal_present
        })
    
    df = pd.DataFrame(metrics)
    
    # Calculate time since disappearance
    df['Time_Since_Disappearance'] = 0
    
    last_signal_idx = None
    for i in range(len(df) - 1, -1, -1):
        if df.loc[i, 'Signal_Present']:
            last_signal_idx = i
            break
    
    if last_signal_idx is not None:
        for i in range(last_signal_idx + 1, len(df)):
            df.loc[i, 'Time_Since_Disappearance'] = i - last_signal_idx
    
    return df


def calculate_summary_statistics(
    signal_mask: np.ndarray,
    distance_bins: np.ndarray,
    exclude_t0: bool = True
) -> dict:
    """
    Calculate single-row summary statistics across all timepoints.
    
    For spatial spread:
    - Finds max distance with signal for each timepoint
    - Reports max, mean, and median of these distances
    
    For temporal extent:
    - Finds last timepoint where signal is present
    
    Args:
        signal_mask: Boolean mask (distance × timepoints) INCLUDING T0
        distance_bins: Distance bin values
        exclude_t0: Skip T0 in calculations (default: True)
    
    Returns:
        Dictionary with summary statistics:
        - Max_Spread_Max: Maximum of all max distances across timepoints
        - Max_Spread_Mean: Mean of max distances across timepoints
        - Max_Spread_Median: Median of max distances across timepoints
        - Last_Timepoint_With_Signal: Last timepoint where signal is present
    """
    n_distance, n_time = signal_mask.shape
    
    # Determine timepoint range
    start_t = 1 if exclude_t0 and n_time > 1 else 0
    
    # Collect max distance for each timepoint
    max_distances = []
    
    for t_idx in range(start_t, n_time):
        mask_t = signal_mask[:, t_idx]
        
        if np.sum(mask_t) > 0:
            # Signal present - find max distance
            distances_with_signal = distance_bins[mask_t]
            max_distance = np.max(distances_with_signal)
            max_distances.append(max_distance)
    
    # Calculate spread statistics
    if len(max_distances) > 0:
        max_spread_max = float(np.max(max_distances))
        max_spread_mean = float(np.mean(max_distances))
        max_spread_median = float(np.median(max_distances))
    else:
        max_spread_max = 0.0
        max_spread_mean = 0.0
        max_spread_median = 0.0
    
    # Find last timepoint with signal
    last_timepoint_with_signal = 0
    for t_idx in range(n_time - 1, start_t - 1, -1):
        if np.sum(signal_mask[:, t_idx]) > 0:
            last_timepoint_with_signal = t_idx
            break
    
    return {
        'Max_Spread_Max': max_spread_max,
        'Max_Spread_Mean': max_spread_mean,
        'Max_Spread_Median': max_spread_median,
        'Last_Timepoint_With_Signal': int(last_timepoint_with_signal)
    }


def create_summary_plot(
    heatmap_smoothed: np.ndarray,
    signal_mask: np.ndarray,
    metrics_df: pd.DataFrame,
    distance_bins: np.ndarray,
    output_path: str,
    colormap: str = 'viridis',
    mask_alpha: float = 0.3
) -> None:
    """
    Create summary visualization with heatmap, mask overlay, and max distance plot.
    
    Args:
        heatmap_smoothed: Smoothed heatmap after T0 subtraction (distance × time)
        signal_mask: Boolean signal mask (distance × time)
        metrics_df: Metrics dataframe with Max_Distance column
        distance_bins: Distance bin values
        output_path: Path to save PNG
        colormap: Matplotlib colormap (default: 'viridis')
        mask_alpha: Alpha for mask overlay (default: 0.3)
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # === PLOT 1: Heatmap with mask overlay ===
    ax1 = axes[0]
    
    n_distance, n_time = heatmap_smoothed.shape
    
    # Display heatmap
    vmin, vmax = np.percentile(heatmap_smoothed, [1, 99])
    
    im1 = ax1.imshow(heatmap_smoothed, aspect='auto', cmap=colormap,
                     extent=[0, n_time, distance_bins[-1], distance_bins[0]],
                     vmin=vmin, vmax=vmax, interpolation='nearest')
    
    # Overlay mask with transparency
    mask_overlay = np.zeros((*signal_mask.shape, 4))
    mask_overlay[signal_mask] = [1, 1, 1, mask_alpha]
    
    ax1.imshow(mask_overlay, aspect='auto', extent=[0, n_time, distance_bins[-1], distance_bins[0]],
              origin='upper', interpolation='nearest')
    
    ax1.set_xlabel('Timepoint', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Distance', fontweight='bold', fontsize=12)
    ax1.set_title('T0-Subtracted Heatmap with DBSCAN Signal Mask', fontweight='bold', fontsize=14)
    plt.colorbar(im1, ax=ax1, label='Intensity (T0-subtracted)')
    
    # === PLOT 2: Max distance over time ===
    ax2 = axes[1]
    
    # Filter to only timepoints with signal
    signal_present = metrics_df[metrics_df['Signal_Present'] == True]
    
    if len(signal_present) > 0:
        ax2.plot(signal_present['Timepoint'], signal_present['Max_Distance'], 
                'b-', linewidth=2, marker='o', markersize=6, label='Max Distance')
        
        # Mark disappearance point
        last_signal_t = signal_present['Timepoint'].max()
        ax2.axvline(x=last_signal_t, color='red', linestyle='--', linewidth=2,
                   label=f'Signal Disappears (T={last_signal_t})')
    
    ax2.set_xlabel('Timepoint', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Maximum Distance Reached', fontweight='bold', fontsize=12)
    ax2.set_title('Signal Spread Over Time', fontweight='bold', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Overall figure title with key metrics
    if len(signal_present) > 0:
        max_dist_overall = signal_present['Max_Distance'].max()
        last_t = signal_present['Timepoint'].max()
        fig.suptitle(f'Signal Quantification Summary\n'
                    f'Max Distance: {max_dist_overall:.1f} | Signal Duration: {last_t} timepoints',
                    fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved summary plot: {output_path}")


# --- DBSCAN segmentation with global threshold ---
def segment_by_clustering_global(
    heatmap_smoothed: np.ndarray,
    distance_bins: np.ndarray,
    global_threshold: float,
    eps: float = 0.20,
    min_samples: int = 30,
    quiet: bool = False
) -> np.ndarray:
    """
    Segment signal using DBSCAN clustering with a global threshold.
    """
    def log_info(msg):
        if not quiet:
            logging.info(msg)
    n_distance, n_time = heatmap_smoothed.shape
    log_info("=" * 60)
    log_info(f"DBSCAN CLUSTERING (global threshold={global_threshold:.2f})")
    log_info("=" * 60)
    # Pre-filter candidate pixels
    candidate_mask = heatmap_smoothed > global_threshold
    n_candidates = np.sum(candidate_mask)
    log_info(f"Candidate pixels: {n_candidates} (threshold={global_threshold:.2f})")
    if n_candidates == 0:
        logging.warning("No candidates above global threshold")
        return np.zeros_like(heatmap_smoothed, dtype=bool)
    # Extract and normalize features
    features = []
    pixel_coords = []
    distance_scale = np.max(distance_bins) - np.min(distance_bins)
    time_scale = float(n_time)
    intensity_scale = np.max(heatmap_smoothed)
    if distance_scale == 0:
        distance_scale = 1.0
    if intensity_scale == 0:
        intensity_scale = 1.0
    for d_idx in range(n_distance):
        for t_idx in range(n_time):
            if candidate_mask[d_idx, t_idx]:
                features.append([
                    distance_bins[d_idx] / distance_scale,
                    t_idx / time_scale,
                    heatmap_smoothed[d_idx, t_idx] / intensity_scale
                ])
                pixel_coords.append((d_idx, t_idx))
    features = np.array(features)
    pixel_coords = np.array(pixel_coords)
    # DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)
    labels = dbscan.fit_predict(features)
    unique_labels = set(labels) - {-1}
    n_clusters = len(unique_labels)
    n_noise = np.sum(labels == -1)
    log_info(f"Found {n_clusters} clusters, {n_noise} noise pixels")
    if n_clusters == 0:
        logging.warning("No clusters found - all noise")
        return np.zeros_like(heatmap_smoothed, dtype=bool)
    # Calculate cluster properties and select closest to origin
    cluster_info = {}
    for label in unique_labels:
        cluster_mask = (labels == label)
        cluster_features = features[cluster_mask]
        center_dist_norm = np.mean(cluster_features[:, 0])
        center_time_norm = np.mean(cluster_features[:, 1])
        origin_norm = np.array([0.0, 1.0 / time_scale])
        cluster_center_norm = np.array([center_dist_norm, center_time_norm])
        dist_to_origin = np.linalg.norm(cluster_center_norm - origin_norm)
        cluster_info[label] = {
            'size': np.sum(cluster_mask),
            'dist_to_origin': dist_to_origin,
            'pixels': pixel_coords[cluster_mask]
        }
    signal_cluster_label = min(cluster_info.keys(), key=lambda k: cluster_info[k]['dist_to_origin'])
    signal_info = cluster_info[signal_cluster_label]
    log_info(f"Selected cluster {signal_cluster_label}: {signal_info['size']} pixels")
    signal_mask = np.zeros_like(heatmap_smoothed, dtype=bool)
    for d_idx, t_idx in signal_info['pixels']:
        signal_mask[d_idx, t_idx] = True
    log_info("=" * 60)
    return signal_mask

# --- Main two-pass pipeline ---
def main():
    parser = argparse.ArgumentParser(
        description='Quantify signal spread and decay from distance heatmap TSV files (v5 - global threshold)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Quantify with DBSCAN (v5 - global threshold)
  environment: uv@3.11:segmentation
  commands:
  - python
  - '%REPO%/standard_code/python/plots/quantify_distance_heatmap_v5.py'
  - --input-search-pattern: '%YAML%/quantification_bulk/*bulk_per_distance.tsv'
  - --output-folder: '%YAML%/quantification_bulk'
  - --clustering-eps: 0.20
  - --clustering-min-samples: 30
  - --smooth-sigma-distance: 3.0
  - --threshold-percentile: 50.0
  - --split-char: '__'
  - --metadata-columns: expID,Group,Replicate
  - --log-level: INFO
"""
    )
    parser.add_argument('--input-search-pattern', type=str, required=True,
                       help='Glob pattern for input TSV files')
    parser.add_argument('--output-folder', type=str, required=True,
                       help='Output folder for metrics TSV files')
    parser.add_argument('--smooth-sigma-distance', type=float, default=3.0,
                       help='Gaussian smoothing sigma in distance (default: 3.0)')
    parser.add_argument('--clustering-eps', type=float, default=0.20,
                       help='DBSCAN epsilon (default: 0.20)')
    parser.add_argument('--clustering-min-samples', type=int, default=30,
                       help='DBSCAN min samples (default: 30)')
    parser.add_argument('--threshold-percentile', type=float, default=50.0,
                       help='Global percentile threshold for clustering (default: 50.0)')
    parser.add_argument('--split-char', type=str, default='__',
                       help='Filename split character (default: "__")')
    parser.add_argument('--metadata-columns', type=str, default='Group',
                       help='Comma-separated metadata column names (default: "Group")')
    parser.add_argument('--colormap', type=str, default='viridis',
                       help='Matplotlib colormap for visualization (default: "viridis")')
    parser.add_argument('--mask-alpha', type=float, default=0.3,
                       help='Alpha transparency for mask overlay (default: 0.3)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    tsv_files = glob.glob(args.input_search_pattern, recursive=True)
    if len(tsv_files) == 0:
        logging.error(f"No TSV files found: {args.input_search_pattern}")
        sys.exit(1)
    logging.info(f"Found {len(tsv_files)} TSV files")
    os.makedirs(args.output_folder, exist_ok=True)
    # --- First pass: collect all positive pixels ---
    all_positive_pixels = []
    for tsv_file in tqdm(tsv_files, desc="Collecting positive pixels", unit="file"):
        df = pd.read_csv(tsv_file, sep='\t')
        sum_cols = [col for col in df.columns if col.startswith('Sum_') and '_Random' in col]
        if len(sum_cols) == 0:
            logging.warning(f"No Sum_*_Random column found in {tsv_file}")
            continue
        intensity_col = sum_cols[0]
        pivot_df = df.pivot(index='Mask_Index', columns='Timepoint', values=intensity_col)
        heatmap_data = pivot_df.values.astype(np.float64)
        heatmap_data_clean = np.nan_to_num(heatmap_data, nan=0.0)
        heatmap_smoothed = gaussian_filter(heatmap_data_clean, sigma=[args.smooth_sigma_distance, 0])
        heatmap_smoothed = subtract_t0_background(heatmap_smoothed, quiet=True)
        positive_pixels = heatmap_smoothed[heatmap_smoothed > 0]
        all_positive_pixels.append(positive_pixels)
    if len(all_positive_pixels) == 0:
        logging.error("No positive pixels found in any file.")
        sys.exit(1)
    all_positive_pixels = np.concatenate(all_positive_pixels)
    global_threshold = np.percentile(all_positive_pixels, args.threshold_percentile)
    logging.info(f"Global threshold ({args.threshold_percentile}th percentile): {global_threshold:.2f}")
    # --- Second pass: process files with global threshold ---
    def process_single_tsv_v5(tsv_path: str, output_dir: str, args: argparse.Namespace, global_threshold: float, quiet: bool = False) -> tuple:
        try:
            tsv_name = Path(tsv_path).stem
            df = pd.read_csv(tsv_path, sep='\t')
            sum_cols = [col for col in df.columns if col.startswith('Sum_') and '_Random' in col]
            if len(sum_cols) == 0:
                raise ValueError(f"No Sum_*_Random column found")
            intensity_col = sum_cols[0]
            pivot_df = df.pivot(index='Mask_Index', columns='Timepoint', values=intensity_col)
            distance_bins = pivot_df.index.values.astype(float)
            heatmap_data = pivot_df.values.astype(np.float64)
            heatmap_data_clean = np.nan_to_num(heatmap_data, nan=0.0)
            heatmap_smoothed = gaussian_filter(heatmap_data_clean, sigma=[args.smooth_sigma_distance, 0])
            heatmap_smoothed = subtract_t0_background(heatmap_smoothed, quiet=quiet)
            # Save raw heatmap TIF (before T0 subtraction)
            raw_heatmap_smoothed = gaussian_filter(heatmap_data_clean, sigma=[args.smooth_sigma_distance, 0])
            raw_heatmap_path = os.path.join(output_dir, f"{tsv_name}_raw_heatmap.tif")
            heatmap_min = np.nanmin(raw_heatmap_smoothed)
            heatmap_max = np.nanmax(raw_heatmap_smoothed)
            
            if heatmap_max > heatmap_min:
                heatmap_scaled = ((raw_heatmap_smoothed - heatmap_min) / (heatmap_max - heatmap_min) * 65535).astype(np.uint16)
            else:
                heatmap_scaled = np.zeros_like(raw_heatmap_smoothed, dtype=np.uint16)
            
            heatmap_flipped = np.flipud(heatmap_scaled)
            heatmap_img = heatmap_flipped[np.newaxis, np.newaxis, np.newaxis, :, :]
            rp.save_tczyx_image(heatmap_img, raw_heatmap_path)
            
            if not quiet:
                logging.info(f"Saved raw heatmap: {raw_heatmap_path}")
            
            # Save T0-subtracted heatmap TIF
            t0_subtracted_path = os.path.join(output_dir, f"{tsv_name}_t0_subtracted.tif")
            t0sub_min = np.nanmin(heatmap_smoothed)
            t0sub_max = np.nanmax(heatmap_smoothed)
            
            if t0sub_max > t0sub_min:
                t0sub_scaled = ((heatmap_smoothed - t0sub_min) / (t0sub_max - t0sub_min) * 65535).astype(np.uint16)
            else:
                t0sub_scaled = np.zeros_like(heatmap_smoothed, dtype=np.uint16)
            
            t0sub_flipped = np.flipud(t0sub_scaled)
            t0sub_img = t0sub_flipped[np.newaxis, np.newaxis, np.newaxis, :, :]
            rp.save_tczyx_image(t0sub_img, t0_subtracted_path)
            
            if not quiet:
                logging.info(f"Saved T0-subtracted heatmap: {t0_subtracted_path}")
            
            # DBSCAN segmentation with global threshold
            signal_mask = segment_by_clustering_global(
                heatmap_smoothed=heatmap_smoothed,
                distance_bins=distance_bins,
                global_threshold=global_threshold,
                eps=args.clustering_eps,
                min_samples=args.clustering_min_samples,
                quiet=quiet
            )
            
            # Save mask TIF
            mask_path = os.path.join(output_dir, f"{tsv_name}_signal_mask.tif")
            mask_uint8 = (signal_mask * 255).astype(np.uint8)
            mask_flipped = np.flipud(mask_uint8)
            mask_img = mask_flipped[np.newaxis, np.newaxis, np.newaxis, :, :]
            rp.save_tczyx_image(mask_img, mask_path)
            
            if not quiet:
                logging.info(f"Saved signal mask: {mask_path}")
            
            # Quantify
            metrics_df = quantify_signal_from_mask(
                signal_mask=signal_mask,
                distance_bins=distance_bins,
                exclude_t0=True
            )
            
            # Add metadata from filename
            experiment_info = get_experiment_info_from_filename(
                tsv_name,
                split=args.split_char,
                colname=args.metadata_columns.split(',')
            )
            
            for key, value in experiment_info.items():
                metrics_df[key] = value
            
            metrics_df['Source_File'] = tsv_name
            
            # Save metrics TSV
            output_tsv = os.path.join(output_dir, f"{tsv_name}_metrics.tsv")
            metrics_df.to_csv(output_tsv, sep='\t', index=False)
            
            if not quiet:
                logging.info(f"Saved metrics: {output_tsv}")
            
            # Calculate and save summary statistics (single row)
            summary_stats = calculate_summary_statistics(
                signal_mask=signal_mask,
                distance_bins=distance_bins,
                exclude_t0=True
            )
            
            # Add metadata to summary
            for key, value in experiment_info.items():
                summary_stats[key] = value
            summary_stats['Source_File'] = tsv_name
            
            # Save summary TSV
            summary_df = pd.DataFrame([summary_stats])
            output_summary_tsv = os.path.join(output_dir, f"{tsv_name}_summary.tsv")
            summary_df.to_csv(output_summary_tsv, sep='\t', index=False)
            
            if not quiet:
                logging.info(f"Saved summary: {output_summary_tsv}")
            
            # Create summary plot
            output_plot = os.path.join(output_dir, f"{tsv_name}_summary.png")
            create_summary_plot(
                heatmap_smoothed=heatmap_smoothed,
                signal_mask=signal_mask,
                metrics_df=metrics_df,
                distance_bins=distance_bins,
                output_path=output_plot,
                colormap=args.colormap,
                mask_alpha=args.mask_alpha
            )
            
            return (tsv_path, True, None)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return (tsv_path, False, error_msg)
    # Parallel or sequential processing
    if not args.no_parallel and len(tsv_files) > 1:
        n_workers = min(cpu_count(), len(tsv_files))
        logging.info(f"Using {n_workers} parallel workers")
        process_func = partial(process_single_tsv_v5, output_dir=args.output_folder, args=args, global_threshold=global_threshold, quiet=True)
        with Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, tsv_files),
                total=len(tsv_files),
                desc="Processing TSV files",
                unit="file"
            ))
    else:
        results = []
        for tsv_file in tqdm(tsv_files, desc="Processing TSV files", unit="file"):
            result = process_single_tsv_v5(tsv_file, args.output_folder, args, global_threshold, quiet=False)
            results.append(result)
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    logging.info(f"Complete: {successful} successful, {failed} failed")
    if failed > 0:
        logging.info("Failed files:")
        for tsv_path, success, error_msg in results:
            if not success:
                logging.info(f"  {tsv_path}: {error_msg}")

if __name__ == "__main__":
    main()

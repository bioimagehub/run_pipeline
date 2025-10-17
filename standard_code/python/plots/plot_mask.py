"""
Visualize grayscale images with colored mask overlays.

This module creates a visualization showing the first, middle, and last timepoints
of an image with colored mask overlays. Each labeled region in the mask gets a
unique color for easy identification.

Author: BIPHUB - Bioimage Informatics Hub, University of Oslo
License: MIT
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import Tuple, Optional
import logging

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

def create_label_colormap(num_labels: int, background_color: Tuple[float, float, float, float] = (0, 0, 0, 0), 
                         colormap: str = 'hsv', alpha: float = 0.5) -> ListedColormap:
    """
    Create a colormap with distinct colors for each label.
    
    Args:
        num_labels: Number of unique labels (excluding background)
        background_color: RGBA color for background (label 0)
        colormap: Color generation strategy:
            'hsv' - evenly spaced hues (default, adjacent labels have similar colors)
            'random' - shuffled hues (adjacent labels have different colors)
            'tab20', 'tab20b', 'tab20c' - matplotlib categorical colormaps
            or any matplotlib colormap name
        alpha: Alpha transparency value (0-1, default: 0.5)
        
    Returns:
        ListedColormap with unique colors for each label
    """
    colors = [background_color]  # Background is transparent
    
    if num_labels > 0:
        if colormap in ['hsv', 'random']:
            # Generate evenly spaced hues
            hues = np.linspace(0, 1, num_labels, endpoint=False)
            
            if colormap == 'random':
                # Shuffle hues so adjacent labels get dissimilar colors
                # Use a deterministic shuffle for reproducibility
                np.random.seed(42)
                np.random.shuffle(hues)
            
            for hue in hues:
                # Convert HSV to RGB (H=hue, S=1, V=1)
                h = hue * 6
                x = 1 - abs((h % 2) - 1)
                
                if h < 1:
                    r, g, b = 1, x, 0
                elif h < 2:
                    r, g, b = x, 1, 0
                elif h < 3:
                    r, g, b = 0, 1, x
                elif h < 4:
                    r, g, b = 0, x, 1
                elif h < 5:
                    r, g, b = x, 0, 1
                else:
                    r, g, b = 1, 0, x
                
                colors.append((r, g, b, alpha))
        else:
            # Use matplotlib colormap
            try:
                cmap = plt.get_cmap(colormap)
                # Sample colors evenly across the colormap
                for i in range(num_labels):
                    rgba = cmap(i / max(num_labels - 1, 1))
                    colors.append((rgba[0], rgba[1], rgba[2], alpha))
            except ValueError:
                logging.warning(f"Unknown colormap '{colormap}', falling back to 'hsv'")
                # Fallback to HSV
                for i in range(num_labels):
                    hue = i / max(num_labels, 1)
                    h = hue * 6
                    x = 1 - abs((h % 2) - 1)
                    
                    if h < 1:
                        r, g, b = 1, x, 0
                    elif h < 2:
                        r, g, b = x, 1, 0
                    elif h < 3:
                        r, g, b = 0, 1, x
                    elif h < 4:
                        r, g, b = 0, x, 1
                    elif h < 5:
                        r, g, b = x, 0, 1
                    else:
                        r, g, b = 1, 0, x
                    
                    colors.append((r, g, b, alpha))
    
    return ListedColormap(colors)


def plot_image_with_mask(
    image_path: str,
    mask_path: str,
    output_path: Optional[str] = None,
    alpha: float = 0.5,
    channel: int = None,
    z_slice: Optional[int] = None,
    colormap: str = 'random'
) -> None:
    """
    Create a plot showing first, middle, and last timepoints with mask overlays.
    Shows all channels vertically with the same mask overlay on each.
    
    Args:
        image_path: Path to the input image
        mask_path: Path to the mask image
        output_path: Optional path to save the plot. If None, displays interactively.
        alpha: Transparency of mask overlay (0-1)
        channel: Which channel to display. If None, displays all channels (default: None)
        z_slice: Which Z slice to display. If None, uses maximum intensity projection.
        colormap: Color generation strategy for mask labels (default: 'random')
    """
    logging.info(f"Loading image: {image_path}")
    img = rp.load_tczyx_image(image_path)
    
    logging.info(f"Loading mask: {mask_path}")
    mask = rp.load_tczyx_image(mask_path)
    
    T, C, Z, Y, X = img.shape
    mT, mC, mZ, mY, mX = mask.shape
    
    logging.info(f"Image shape: {img.shape}")
    logging.info(f"Mask shape: {mask.shape}")
    
    # Validate dimensions
    if (Y, X) != (mY, mX):
        raise ValueError(f"Image and mask XY dimensions must match. Got image: ({Y}, {X}), mask: ({mY}, {mX})")
    
    # Determine which channels to show
    if channel is not None:
        if channel >= C:
            raise ValueError(f"Channel {channel} not available. Image has {C} channels (0-{C-1})")
        channels_to_show = [channel]
        logging.info(f"Displaying channel {channel}")
    else:
        channels_to_show = list(range(C))
        logging.info(f"Displaying all {C} channels")
    
    # Determine which timepoints to show
    if T == 1:
        timepoints = [0]
        logging.info("Single timepoint image, showing only T=0")
    elif T == 2:
        timepoints = [0, 1]
        logging.info("Two timepoint image, showing T=0 and T=1")
    else:
        middle_t = T // 2
        timepoints = [0, middle_t, T - 1]
        logging.info(f"Showing timepoints: T=0, T={middle_t}, T={T-1}")
    
    # Handle mask timepoint mapping
    if mT == 1 and T > 1:
        mask_t_map = {t: 0 for t in timepoints}
        logging.info("Mask has single timepoint, using for all image timepoints")
    elif mT == T:
        mask_t_map = {t: t for t in timepoints}
    else:
        raise ValueError(f"Incompatible timepoints: image has {T}, mask has {mT}")
    
    # Create figure with channels as rows and timepoints as columns
    n_rows = len(channels_to_show)
    n_cols = len(timepoints)
    
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
        max_height = monitor_height_in * 0.85
        
        # Calculate figure size to fit in single monitor
        width_per_col = max_width / n_cols
        height_per_row = max_height / n_rows
        
        figsize = (width_per_col * n_cols, height_per_row * n_rows)
        logging.info(f"Monitor: {monitor_width_px}×{monitor_height_px} px ({monitor_width_in:.1f}×{monitor_height_in:.1f} in)")
        logging.info(f"Creating figure: {figsize[0]:.1f}×{figsize[1]:.1f} in ({n_rows} rows × {n_cols} cols)")
    except:
        # Fallback to smaller base size if monitor detection fails
        base_width_per_col = 3
        base_height_per_row = 2.5
        figsize = (base_width_per_col * n_cols, base_height_per_row * n_rows)
        logging.info(f"Creating figure with size: {figsize[0]:.1f} × {figsize[1]:.1f} inches ({n_rows} rows × {n_cols} cols)")
    
    # Use constrained_layout for automatic tight spacing
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    
    # Handle single row/column cases
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Get global min/max for consistent grayscale scaling per channel
    channel_ranges = {}
    for c in channels_to_show:
        img_min = np.inf
        img_max = -np.inf
        for t in timepoints:
            if z_slice is not None:
                img_2d = img.data[t, c, z_slice, :, :]
            else:
                # Maximum intensity projection
                img_2d = np.max(img.data[t, c, :, :, :], axis=0)
            img_min = min(img_min, np.percentile(img_2d, 1))
            img_max = max(img_max, np.percentile(img_2d, 99))
        channel_ranges[c] = (img_min, img_max)
        logging.info(f"Channel {c} intensity range (1-99 percentile): [{img_min:.2f}, {img_max:.2f}]")
    
    # Get number of unique labels for colormap
    max_label = 0
    for t in timepoints:
        mt = mask_t_map[t]
        if z_slice is not None:
            mask_2d = mask.data[mt, 0, z_slice, :, :]
        else:
            # For masks, take max projection
            mask_2d = np.max(mask.data[mt, 0, :, :, :], axis=0)
        max_label = max(max_label, int(mask_2d.max()))
    
    logging.info(f"Maximum mask label: {max_label}")
    
    # Create colormap for masks
    mask_cmap = create_label_colormap(max_label, colormap=colormap, alpha=alpha)
    
    # Plot each channel (row) and timepoint (column)
    for row_idx, c in enumerate(channels_to_show):
        img_min, img_max = channel_ranges[c]
        
        for col_idx, t in enumerate(timepoints):
            ax = axes[row_idx, col_idx]
            mt = mask_t_map[t]
            
            # Get image slice
            if z_slice is not None:
                img_2d = img.data[t, c, z_slice, :, :]
                z_info = f"Z={z_slice}"
            else:
                img_2d = np.max(img.data[t, c, :, :, :], axis=0)
                z_info = "MIP"
            
            # Get mask slice
            if z_slice is not None:
                mask_2d = mask.data[mt, 0, z_slice, :, :]
            else:
                mask_2d = np.max(mask.data[mt, 0, :, :, :], axis=0)
            
            # Display grayscale image
            ax.imshow(img_2d, cmap='gray', vmin=img_min, vmax=img_max, interpolation='nearest')
            
            # Overlay mask with colors
            # Create a masked array where background (0) is transparent
            mask_overlay = np.ma.masked_where(mask_2d == 0, mask_2d)
            # Alpha is already baked into the colormap, so use alpha=1.0 here
            ax.imshow(mask_overlay, cmap=mask_cmap, alpha=1.0, interpolation='nearest', 
                     vmin=0, vmax=max_label)
            
            # Count objects in this timepoint
            n_objects = len(np.unique(mask_2d)) - 1  # Subtract background
            
            # Title shows timepoint and channel info
            if row_idx == 0:
                ax.set_title(f'T={t} ({z_info})\n{n_objects} objects', fontsize=11)
            else:
                ax.set_title(f'{n_objects} objects', fontsize=10)
            
            # Add channel label on the left
            if col_idx == 0:
                ax.set_ylabel(f'Ch {c}', fontsize=12, fontweight='bold')
            
            ax.axis('off')
    
    # Add overall title
    img_name = Path(image_path).stem
    mask_name = Path(mask_path).stem
    if channel is not None:
        fig.suptitle(f'{img_name}\nMask: {mask_name} (Channel {channel})', 
                    fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f'{img_name}\nMask: {mask_name} (All {C} channels)', 
                    fontsize=14, fontweight='bold')
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved plot to: {output_path}")
        plt.close()
    else:
        # Set window title
        fig.canvas.manager.set_window_title(f'{Path(image_path).stem} - Mask Overlay')
        
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


def main():
    parser = argparse.ArgumentParser(
        description='Visualize images with colored mask overlays',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show 3 example plots (first, middle, last) with ALL channels - DEFAULT
  python plot_mask.py --input-search-pattern "./images/*.tif" --mask-search-pattern "./masks/*_mask.tif"
  
  # Show 5 evenly spaced examples
  python plot_mask.py --input-search-pattern "./images/*.tif" --mask-search-pattern "./masks/*_mask.tif" --mode examples5
  
  # Show first 3 images
  python plot_mask.py --input-search-pattern "./images/*.tif" --mask-search-pattern "./masks/*_mask.tif" --mode first3
  
  # Show first 10 images
  python plot_mask.py --input-search-pattern "./images/*.tif" --mask-search-pattern "./masks/*_mask.tif" --mode first10
  
  # Show 5 random images
  python plot_mask.py --input-search-pattern "./images/*.tif" --mask-search-pattern "./masks/*_mask.tif" --mode random5
  
  # Show only specific channel
  python plot_mask.py --input-search-pattern "./images/*.tif" --mask-search-pattern "./masks/*_mask.tif" --channel 1
  
  # Process all files
  python plot_mask.py --input-search-pattern "./images/*.tif" --mask-search-pattern "./masks/*_mask.tif" --mode all
  
  # Save plots instead of displaying (examples mode)
  python plot_mask.py --input-search-pattern "./images/*.tif" --mask-search-pattern "./masks/*_mask.tif" --output-folder ./plots
  
  # Show specific Z slice
  python plot_mask.py --input-search-pattern "./images/*.tif" --mask-search-pattern "./masks/*_mask.tif" --z-slice 5
  
  # Adjust mask transparency
  python plot_mask.py --input-search-pattern "./images/*.tif" --mask-search-pattern "./masks/*_mask.tif" --alpha 0.7
  
  # Use random colormap (adjacent masks have different colors) - DEFAULT
  python plot_mask.py --input-search-pattern "./images/*.tif" --mask-search-pattern "./masks/*_mask.tif" --colormap random
  
  # Use sequential HSV colormap (adjacent masks have similar colors)
  python plot_mask.py --input-search-pattern "./images/*.tif" --mask-search-pattern "./masks/*_mask.tif" --colormap hsv
  
  # Use matplotlib categorical colormaps
  python plot_mask.py --input-search-pattern "./images/*.tif" --mask-search-pattern "./masks/*_mask.tif" --colormap tab20
  python plot_mask.py --input-search-pattern "./images/*.tif" --mask-search-pattern "./masks/*_mask.tif" --colormap Set3
        """
    )
    
    parser.add_argument('--input-search-pattern', type=str, required=True,
                       help='Glob pattern for input images')
    parser.add_argument('--mask-search-pattern', type=str, required=True,
                       help='Glob pattern for mask images')
    parser.add_argument('--output-folder', type=str, default=None,
                       help='Folder to save plots. If not specified, displays interactively.')
    parser.add_argument('--mode', type=str, default='examples3',
                       help='Processing mode: "all" = all files, "examples[N]" = first, middle, last N files (default: 3), '
                            '"first[N]" = first N files (default: 3), "random[N]" = N random files (default: 3)')
    parser.add_argument('--search-subfolders', action='store_true',
                       help='Enable recursive search for files')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Transparency of mask overlay (0-1, default: 0.5)')
    parser.add_argument('--channel', type=int, default=None,
                       help='Which channel to display. If not specified, shows all channels (default: all)')
    parser.add_argument('--z-slice', type=int, default=None,
                       help='Which Z slice to display. If not specified, uses maximum intensity projection.')
    parser.add_argument('--colormap', type=str, default='random',
                       help='Colormap for mask labels: "random" (shuffled hues, adjacent labels dissimilar - default), '
                            '"hsv" (sequential hues), or any matplotlib colormap name (e.g., "tab20", "Set3")')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Debug: Show what channel argument was received
    if args.channel is None:
        logging.info("Channel argument: None (will show ALL channels)")
    else:
        logging.info(f"Channel argument: {args.channel} (will show only this channel)")
    
    # Find matching files using get_files_to_process2
    image_files = rp.get_files_to_process2(args.input_search_pattern, args.search_subfolders)
    
    if not image_files:
        raise FileNotFoundError(f"No files found matching pattern: {args.input_search_pattern}")
    
    logging.info(f"Found {len(image_files)} image files")
    
    # Create output folder if needed
    if args.output_folder:
        os.makedirs(args.output_folder, exist_ok=True)
        logging.info(f"Output folder: {args.output_folder}")
    
    # Extract prefix from input pattern
    import re
    def get_prefix(pattern):
        m = re.search(r'(.*)\*', pattern)
        return m.group(1) if m else ''
    
    input_prefix = get_prefix(args.input_search_pattern)
    
    # Prepare file pairs by matching patterns
    file_pairs = []
    
    for image_path in image_files:
        # Get relative path for filename matching
        rel_img = image_path[len(input_prefix):] if image_path.startswith(input_prefix) else os.path.basename(image_path)
        rel_img_noext = os.path.splitext(rel_img)[0]
        
        # Replace * with the image filename (without extension)
        specific_mask_pattern = args.mask_search_pattern.replace('*', rel_img_noext)
        mask_matches = rp.get_files_to_process2(specific_mask_pattern, args.search_subfolders)
        
        if len(mask_matches) == 0:
            logging.warning(f"Mask not found for {image_path} using pattern {specific_mask_pattern}, skipping.")
            continue
        if len(mask_matches) > 1:
            logging.warning(f"Multiple masks found for {image_path} using pattern {specific_mask_pattern}, skipping.")
            continue
        
        file_pairs.append((image_path, mask_matches[0]))
    
    if not file_pairs:
        raise ValueError(f"No matching image-mask pairs found!")
    
    logging.info(f"Matched {len(file_pairs)} image-mask pairs")
    
    # Parse mode and count
    mode = args.mode.lower()
    n_files = len(file_pairs)
    
    # Extract mode type and count
    import re
    mode_match = re.match(r'(all|examples|first|random)(\d+)?$', mode)
    if not mode_match:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'all', 'examples[N]', 'first[N]', or 'random[N]'")
    
    mode_type = mode_match.group(1)
    mode_count = int(mode_match.group(2)) if mode_match.group(2) else 3
    
    # Select files based on mode
    if mode_type == 'all':
        selected_pairs = file_pairs
        logging.info(f"Mode: all - processing all {len(file_pairs)} files")
    
    elif mode_type == 'examples':
        if n_files <= mode_count:
            selected_pairs = file_pairs
            logging.info(f"Mode: examples{mode_count} - showing all {n_files} files (requested {mode_count})")
        else:
            # Select evenly spaced examples: first, middle(s), last
            indices = []
            if mode_count == 1:
                indices = [n_files // 2]  # Just middle
            elif mode_count == 2:
                indices = [0, n_files - 1]  # First and last
            else:
                # First, evenly spaced middles, last
                indices = [0]
                for i in range(1, mode_count - 1):
                    idx = int(i * n_files / (mode_count - 1))
                    indices.append(idx)
                indices.append(n_files - 1)
            
            selected_pairs = [file_pairs[i] for i in indices]
            logging.info(f"Mode: examples{mode_count} - showing {len(selected_pairs)} evenly spaced files from {n_files} total")
    
    elif mode_type == 'first':
        count = min(mode_count, n_files)
        selected_pairs = file_pairs[:count]
        logging.info(f"Mode: first{mode_count} - showing first {count} files from {n_files} total")
    
    elif mode_type == 'random':
        import random
        count = min(mode_count, n_files)
        selected_pairs = random.sample(file_pairs, count)
        logging.info(f"Mode: random{mode_count} - showing {count} random files from {n_files} total")
    
    # Process each pair
    for img_path, mask_path in selected_pairs:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing: {Path(img_path).name}")
        
        if args.output_folder:
            output_name = f"{Path(img_path).stem}_overlay.png"
            output_path = os.path.join(args.output_folder, output_name)
        else:
            output_path = None
        
        try:
            plot_image_with_mask(
                img_path,
                mask_path,
                output_path=output_path,
                alpha=args.alpha,
                channel=args.channel,
                z_slice=args.z_slice,
                colormap=args.colormap
            )
        except Exception as e:
            logging.error(f"Error processing {Path(img_path).name}: {e}")
            continue
    
    logging.info(f"\n{'='*60}")
    logging.info("Processing complete!")


if __name__ == "__main__":
    main()
    

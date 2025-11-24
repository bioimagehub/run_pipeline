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
from matplotlib.patches import Circle
from pathlib import Path
from typing import Tuple, Optional, List
import logging
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

def extract_roi_positions(yaml_path: str) -> Optional[List[Tuple[float, float]]]:
    """
    Extract ROI positions from metadata YAML file.
    
    Args:
        yaml_path: Path to metadata YAML file
        
    Returns:
        List of (x_pixels, y_pixels) tuples, or None if no ROIs found
    """
    if not yaml_path or not os.path.exists(yaml_path):
        return None
    
    try:
        import yaml
        with open(yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        # Navigate to ROIs in metadata
        if not metadata or 'Image metadata' not in metadata:
            return None
        
        image_metadata = metadata['Image metadata']
        if 'ROIs' not in image_metadata or not image_metadata['ROIs']:
            return None
        
        # ROI positions are already in pixel coordinates
        roi_positions = []
        for roi_entry in image_metadata['ROIs']:
            if 'Roi' not in roi_entry:
                continue
            
            roi = roi_entry['Roi']
            positions = roi.get('Positions', {})
            
            # ROI positions are already in pixels
            x_pixels = positions.get('x', None)
            y_pixels = positions.get('y', None)
            
            logging.info(f"ROI position from YAML: x={x_pixels}, y={y_pixels} (pixels)")
            
            if x_pixels is not None and y_pixels is not None:
                roi_positions.append((x_pixels, y_pixels))
                logging.info(f"Added ROI at ({x_pixels:.1f}, {y_pixels:.1f}) pixels")
        
        return roi_positions if roi_positions else None
        
    except Exception as e:
        logging.warning(f"Failed to extract ROI from {yaml_path}: {e}")
        return None

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
    colormap: str = 'random',
    roi_positions: Optional[List[Tuple[float, float]]] = None
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
        roi_positions: Optional list of (x, y) pixel coordinates for ROI markers
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
        
        # Properly destroy the tkinter root to avoid cleanup errors
        # Order matters: destroy BEFORE quit to clean up while event loop is active
        try:
            root.update()  # Final update
        except:
            pass
        try:
            root.destroy()
        except:
            pass
        # Give time for cleanup before quitting
        try:
            root.quit()
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
            
            # Overlay ROI markers if provided
            if roi_positions:
                for roi_x, roi_y in roi_positions:
                    # Validate coordinates are within image bounds
                    if 0 <= roi_x < X and 0 <= roi_y < Y:
                        # Draw larger crosshair using line plots (in data coordinates)
                        line_length = max(Y, X) * 0.03  # 3% of image size
                        ax.plot([roi_x - line_length, roi_x + line_length], [roi_y, roi_y], 
                               'r-', linewidth=2, alpha=0.8)  # Horizontal line
                        ax.plot([roi_x, roi_x], [roi_y - line_length, roi_y + line_length], 
                               'r-', linewidth=2, alpha=0.8)  # Vertical line
                        # Draw circle around ROI
                        circle_radius = line_length * 1.5
                        circle = Circle((roi_x, roi_y), circle_radius, 
                                       color='red', fill=False, linewidth=2, alpha=0.8)
                        ax.add_patch(circle)
                        logging.info(f"Plotted ROI marker at ({roi_x:.1f}, {roi_y:.1f}) - image bounds: ({X}, {Y})")
                    else:
                        logging.warning(f"ROI position ({roi_x:.1f}, {roi_y:.1f}) is outside image bounds ({X}, {Y})")
            
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


def interactive_qc(
    file_pairs: List[Tuple[str, str, Optional[str]]],
    qc_key: str,
    input_search_pattern: str,
    mask_search_pattern: str,
    output_folder: Optional[str],
    alpha: float = 0.5,
    channel: Optional[int] = None,
    z_slice: Optional[int] = None,
    colormap: str = 'random'
) -> None:
    """
    Interactive QC tool to navigate images and record pass/fail status.
    
    Navigation:
    - Left arrow: Previous image (does not loop from first to last)
    - Right arrow: Next image (does not loop from last to first)
    
    QC Recording:
    - Left click: Mark as PASS
    - Right click: Mark as FAIL
    
    Args:
        file_pairs: List of tuples (image_path, mask_path, yaml_path). yaml_path is optional and can be None.
        qc_key: Key to store QC results under (e.g., 'nuc_segmentation')
        input_search_pattern: Original input search pattern for documentation
        mask_search_pattern: Original mask search pattern for documentation
        output_folder: Folder where plots would be saved (for documentation)
        alpha: Transparency of mask overlay (0-1)
        channel: Which channel to display. If None, displays all channels
        z_slice: Which Z slice to display. If None, uses maximum intensity projection
        colormap: Color generation strategy for mask labels
    """
    
    if not file_pairs:
        logging.error("No file pairs provided for interactive QC")
        return
    
    # Enable interactive mode so figures stay responsive
    plt.ion()
    
    current_index = 0
    fig = None
    force_mode = False  # Toggle for reviewing already QC'd images
    
    def is_already_qcd(image_path: str) -> bool:
        """Check if image has already been QC'd with a valid status for this qc_key."""
        if force_mode:
            return False  # Force mode: always show images
        
        qc_file = get_qc_file_path(image_path)
        if not os.path.exists(qc_file):
            return False
        
        try:
            with open(qc_file, 'r') as f:
                qc_data = yaml.safe_load(f)
            
            # Check if this specific qc_key has a valid status
            if qc_data and 'qc' in qc_data and qc_key in qc_data['qc']:
                status = qc_data['qc'][qc_key].get('status', '').lower()
                if status in ['passed', 'failed']:
                    return True
        except Exception as e:
            logging.warning(f"Could not read QC file {qc_file}: {e}")
        
        return False
    
    def get_qc_file_path(image_path: str) -> str:
        """Get the QC YAML file path for an image."""
        return str(Path(image_path).with_suffix('')) + "_QC.yaml"
    
    def load_existing_qc(image_path: str) -> Optional[dict]:
        """Load existing QC data if available."""
        qc_file = get_qc_file_path(image_path)
        if os.path.exists(qc_file):
            try:
                with open(qc_file, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logging.warning(f"Could not load existing QC file {qc_file}: {e}")
        return None
    
    def save_qc_result(image_path: str, status: str) -> None:
        """Save QC result to a YAML file next to the image."""
        qc_file = get_qc_file_path(image_path)
        
        qc_data = {
            "qc": {
                qc_key: {
                    "status": "passed" if status == "pass" else "failed",
                    "comment": f"User defined that segmentation is {'correct' if status == 'pass' else 'incorrect'}",
                    "input_search_pattern": input_search_pattern,
                    "mask_search_pattern": mask_search_pattern,
                    "output_folder": output_folder if output_folder else "N/A"
                }
            }
        }
        
        try:
            with open(qc_file, "w") as f:
                yaml.dump(qc_data, f, default_flow_style=False, sort_keys=False)
            logging.info(f"QC result saved to {qc_file}: {status.upper()}")
            print(f"\n✓ Saved QC: {status.upper()} -> {qc_file}")
        except Exception as e:
            logging.error(f"Failed to save QC file {qc_file}: {e}")
            print(f"\n✗ Error saving QC file: {e}")
    
    def update_display(index: int) -> None:
        """Update the display with the current image and mask."""
        nonlocal fig
        
        if fig is not None:
            plt.close(fig)
        
        image_path, mask_path, yaml_path = file_pairs[index]
        
        # Extract ROI positions if YAML provided
        roi_positions = extract_roi_positions(yaml_path) if yaml_path else None
        
        # Load existing QC status if available
        existing_qc = load_existing_qc(image_path)
        qc_status = ""
        if existing_qc and 'qc' in existing_qc and qc_key in existing_qc['qc']:
            status = existing_qc['qc'][qc_key].get('status', 'unknown')
            qc_status = f" [Current: {status.upper()}]"
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Displaying image {index + 1}/{len(file_pairs)}: {Path(image_path).name}")
        
        # Load images without displaying
        img = rp.load_tczyx_image(image_path)
        mask = rp.load_tczyx_image(mask_path)
        
        T, C, Z, Y, X = img.shape
        mT, mC, mZ, mY, mX = mask.shape
        
        # Validate dimensions
        if (Y, X) != (mY, mX):
            raise ValueError(f"Image and mask XY dimensions must match. Got image: ({Y}, {X}), mask: ({mY}, {mX})")
        
        # Determine which channels to show
        if channel is not None:
            if channel >= C:
                raise ValueError(f"Channel {channel} not available. Image has {C} channels (0-{C-1})")
            channels_to_show = [channel]
        else:
            channels_to_show = list(range(C))
        
        # Determine which timepoints to show
        if T == 1:
            timepoints = [0]
        elif T == 2:
            timepoints = [0, 1]
        else:
            middle_t = T // 2
            timepoints = [0, middle_t, T - 1]
        
        # Handle mask timepoint mapping
        if mT == 1 and T > 1:
            mask_t_map = {t: 0 for t in timepoints}
        elif mT == T:
            mask_t_map = {t: t for t in timepoints}
        else:
            raise ValueError(f"Incompatible timepoints: image has {T}, mask has {mT}")
        
        # Create figure
        n_rows = len(channels_to_show)
        n_cols = len(timepoints)
        
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            root.update_idletasks()
            monitor_width_px = root.winfo_screenwidth()
            monitor_height_px = root.winfo_screenheight()
            # Cleanup order: destroy BEFORE quit
            try:
                root.update()
            except:
                pass
            try:
                root.destroy()
            except:
                pass
            try:
                root.quit()
            except:
                pass
            
            dpi = 96
            monitor_width_in = monitor_width_px / dpi
            monitor_height_in = monitor_height_px / dpi
            max_width = monitor_width_in * 0.85
            max_height = monitor_height_in * 0.85
            width_per_col = max_width / n_cols
            height_per_row = max_height / n_rows
            figsize = (width_per_col * n_cols, height_per_row * n_rows)
        except:
            base_width_per_col = 3
            base_height_per_row = 2.5
            figsize = (base_width_per_col * n_cols, base_height_per_row * n_rows)
        
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
                    img_2d = np.max(img.data[t, c, :, :, :], axis=0)
                img_min = min(img_min, np.percentile(img_2d, 1))
                img_max = max(img_max, np.percentile(img_2d, 99))
            channel_ranges[c] = (img_min, img_max)
        
        # Get number of unique labels for colormap
        max_label = 0
        for t in timepoints:
            mt = mask_t_map[t]
            if z_slice is not None:
                mask_2d = mask.data[mt, 0, z_slice, :, :]
            else:
                mask_2d = np.max(mask.data[mt, 0, :, :, :], axis=0)
            max_label = max(max_label, int(mask_2d.max()))
        
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
                mask_overlay = np.ma.masked_where(mask_2d == 0, mask_2d)
                ax.imshow(mask_overlay, cmap=mask_cmap, alpha=1.0, interpolation='nearest', 
                         vmin=0, vmax=max_label)
                
                # Overlay ROI markers if provided
                if roi_positions:
                    for roi_x, roi_y in roi_positions:
                        # Validate coordinates are within image bounds
                        if 0 <= roi_x < X and 0 <= roi_y < Y:
                            # Draw larger crosshair using line plots (in data coordinates)
                            line_length = max(Y, X) * 0.03  # 3% of image size
                            ax.plot([roi_x - line_length, roi_x + line_length], [roi_y, roi_y], 
                                   'r-', linewidth=2, alpha=0.8)  # Horizontal line
                            ax.plot([roi_x, roi_x], [roi_y - line_length, roi_y + line_length], 
                                   'r-', linewidth=2, alpha=0.8)  # Vertical line
                            # Draw circle around ROI
                            circle_radius = line_length * 1.5
                            circle = Circle((roi_x, roi_y), circle_radius, 
                                           color='red', fill=False, linewidth=2, alpha=0.8)
                            ax.add_patch(circle)
                            logging.info(f"Plotted ROI marker at ({roi_x:.1f}, {roi_y:.1f}) - image bounds: ({X}, {Y})")
                        else:
                            logging.warning(f"ROI position ({roi_x:.1f}, {roi_y:.1f}) is outside image bounds ({X}, {Y})")
                
                # Count objects in this timepoint
                n_objects = len(np.unique(mask_2d)) - 1
                
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
            base_title = f'{img_name}\nMask: {mask_name} (Channel {channel})'
        else:
            base_title = f'{img_name}\nMask: {mask_name} (All {C} channels)'
        
        force_indicator = " [FORCE MODE ON]" if force_mode else ""
        nav_info = f"\n\n[{index + 1}/{len(file_pairs)}]{force_indicator} <-- Prev | Next --> | L-Click=PASS | R-Click=FAIL | F=Toggle Force{qc_status}"
        fig.suptitle(base_title + nav_info, fontsize=12)
        
        # Set window title
        fig.canvas.manager.set_window_title(f'QC Mode [{index + 1}/{len(file_pairs)}]: {Path(image_path).stem}')
        
        # Maximize the window
        try:
            manager = fig.canvas.manager
            # Try different backends
            if hasattr(manager, 'window'):
                if hasattr(manager.window, 'showMaximized'):
                    manager.window.showMaximized()  # Qt backend
                elif hasattr(manager.window, 'state'):
                    manager.window.state('zoomed')  # TkAgg backend
            elif hasattr(manager, 'full_screen_toggle'):
                manager.full_screen_toggle()  # Some other backends
        except Exception as e:
            # Silently fail if maximization doesn't work
            pass
        
        # Show the figure (non-blocking in interactive mode)
        plt.show(block=False)
        plt.pause(0.01)  # Small pause to ensure rendering
    
    def on_key(event) -> None:
        """Handle keyboard navigation."""
        nonlocal current_index, fig, force_mode
        
        if event.key == "right":
            if current_index < len(file_pairs) - 1:
                current_index += 1
                # Skip already QC'd images if not in force mode
                while current_index < len(file_pairs) and is_already_qcd(file_pairs[current_index][0]):
                    print(f"Skipping already QC'd image: {Path(file_pairs[current_index][0]).name}")
                    current_index += 1
                
                if current_index < len(file_pairs):
                    print(f"\n--> Moving to image {current_index + 1}/{len(file_pairs)}: {Path(file_pairs[current_index][0]).name}")
                    update_display(current_index)
                    # Reconnect handlers after creating new figure
                    fig.canvas.mpl_connect("key_press_event", on_key)
                    fig.canvas.mpl_connect("button_press_event", on_click)
                else:
                    current_index = len(file_pairs) - 1  # Stay at last image
                    print("\n--> All remaining images already QC'd. Press 'f' to toggle force mode.")
            else:
                print("\n--> Already at last image")
        
        elif event.key == "left":
            if current_index > 0:
                current_index -= 1
                # Skip already QC'd images if not in force mode
                while current_index > 0 and is_already_qcd(file_pairs[current_index][0]):
                    print(f"Skipping already QC'd image: {Path(file_pairs[current_index][0]).name}")
                    current_index -= 1
                
                if current_index >= 0 and not (is_already_qcd(file_pairs[current_index][0])):
                    print(f"\n<-- Moving to image {current_index + 1}/{len(file_pairs)}: {Path(file_pairs[current_index][0]).name}")
                    update_display(current_index)
                    # Reconnect handlers after creating new figure
                    fig.canvas.mpl_connect("key_press_event", on_key)
                    fig.canvas.mpl_connect("button_press_event", on_click)
                else:
                    current_index = 0  # Stay at first image
                    print("\n<-- All previous images already QC'd. Press 'f' to toggle force mode.")
            else:
                print("\n<-- Already at first image")
        
        elif event.key == "f":
            force_mode = not force_mode
            mode_status = "ON" if force_mode else "OFF"
            print(f"\n[F] Force mode toggled: {mode_status}")
            if force_mode:
                print("    All images will be shown, including already QC'd ones.")
            else:
                print("    Already QC'd images will be skipped automatically.")
            # Update display to show current force mode status in title
            update_display(current_index)
            fig.canvas.mpl_connect("key_press_event", on_key)
            fig.canvas.mpl_connect("button_press_event", on_click)
    
    def on_click(event) -> None:
        """Handle mouse clicks for QC recording."""
        nonlocal current_index, fig
        
        if event.inaxes is None:
            return
        
        image_path, _, _ = file_pairs[current_index]
        
        if event.button == 1:  # Left click = PASS
            save_qc_result(image_path, "pass")
            # Automatically move to next image after recording
            if current_index < len(file_pairs) - 1:
                current_index += 1
                # Skip already QC'd images if not in force mode
                while current_index < len(file_pairs) and is_already_qcd(file_pairs[current_index][0]):
                    print(f"Skipping already QC'd image: {Path(file_pairs[current_index][0]).name}")
                    current_index += 1
                
                if current_index < len(file_pairs):
                    print(f"--> Auto-advancing to image {current_index + 1}/{len(file_pairs)}: {Path(file_pairs[current_index][0]).name}")
                    update_display(current_index)
                    # Reconnect handlers after creating new figure
                    fig.canvas.mpl_connect("key_press_event", on_key)
                    fig.canvas.mpl_connect("button_press_event", on_click)
                else:
                    print("\n✓ All images reviewed! Closing QC mode.")
                    plt.close(fig)
            else:
                print("\n✓ All images reviewed! Closing QC mode.")
                plt.close(fig)
        
        elif event.button == 3:  # Right click = FAIL
            save_qc_result(image_path, "fail")
            # Automatically move to next image after recording
            if current_index < len(file_pairs) - 1:
                current_index += 1
                # Skip already QC'd images if not in force mode
                while current_index < len(file_pairs) and is_already_qcd(file_pairs[current_index][0]):
                    print(f"Skipping already QC'd image: {Path(file_pairs[current_index][0]).name}")
                    current_index += 1
                
                if current_index < len(file_pairs):
                    print(f"--> Auto-advancing to image {current_index + 1}/{len(file_pairs)}: {Path(file_pairs[current_index][0]).name}")
                    update_display(current_index)
                    # Reconnect handlers after creating new figure
                    fig.canvas.mpl_connect("key_press_event", on_key)
                    fig.canvas.mpl_connect("button_press_event", on_click)
                else:
                    print("\n✓ All images reviewed! Closing QC mode.")
                    plt.close(fig)
            else:
                print("\n✓ All images reviewed! Closing QC mode.")
                plt.close(fig)
    
    # Print instructions FIRST
    print("\n" + "="*60)
    print("INTERACTIVE QC MODE")
    print("="*60)
    print("Navigation:")
    print("  <-- Left Arrow  = Previous image (skip already QC'd)")
    print("  --> Right Arrow = Next image (skip already QC'd)")
    print("\nQC Recording (auto-advances to next):")
    print("  Left Click  = Mark as PASS")
    print("  Right Click = Mark as FAIL")
    print("\nOptions:")
    print("  F Key = Toggle Force Mode (show/skip already QC'd images)")
    print(f"\nQC files will be saved as: <image_name>_QC.yaml")
    print(f"QC Key: {qc_key}")
    print(f"Total images: {len(file_pairs)}")
    print(f"Force Mode: {'ON' if force_mode else 'OFF'} (default: OFF)")
    print("="*60 + "\n")
    
    # Skip already QC'd images at startup if not in force mode
    while current_index < len(file_pairs) and is_already_qcd(file_pairs[current_index][0]):
        print(f"Skipping already QC'd image: {Path(file_pairs[current_index][0]).name}")
        current_index += 1
    
    if current_index >= len(file_pairs):
        print("\n✓ All images already QC'd! No new images to review.")
        print("  Press 'f' to toggle force mode and review already QC'd images.")
        current_index = 0  # Reset to first image for force mode toggle
    
    # Display the first image
    update_display(current_index)
    
    # Connect event handlers
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    
    # Try to give the figure focus
    try:
        fig.canvas.manager.window.activateWindow()
        fig.canvas.manager.window.raise_()
    except:
        pass
    
    # Make sure the canvas can receive keyboard focus
    try:
        fig.canvas.setFocusPolicy(2)  # Qt.StrongFocus
        fig.canvas.setFocus()
    except:
        pass
    
    print("\n[INFO] Click on the figure window to ensure it has keyboard focus, then use arrow keys to navigate.\n")
    
    # Keep the plot interactive and responsive
    plt.show(block=True)
    
    # Clean up matplotlib/tkinter resources properly to avoid cleanup errors
    try:
        plt.close('all')  # Close all figures
        # Force garbage collection of matplotlib objects
        import gc
        gc.collect()
    except Exception as e:
        logging.debug(f"Cleanup warning (can be ignored): {e}")


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
  
  # Show with ROI markers from metadata YAML files
  python plot_mask.py --input-search-pattern "./images/*.tif" --mask-search-pattern "./masks/*_mask.tif" --yaml-search-pattern "./metadata/*_metadata.yaml"
  
Example YAML config for run_pipeline.exe:
---
run:
- name: QC mask overlay with ROI markers
  environment: uv@3.11:qc-mask
  commands:
  - python
  - '%REPO%/standard_code/python/plots/qc_mask.py'
  - --input-search-pattern: '%YAML%/input_tif/**/*.tif'
  - --mask-search-pattern: '%YAML%/output_masks/**/*_mask.tif'
  - --yaml-search-pattern: '%YAML%/input_tif/*_metadata.yaml'
  - --mode: group1
  - --colormap: random
        """
    )
    
    parser.add_argument('--input-search-pattern', type=str, required=True,
                       help='Glob pattern for input images')
    parser.add_argument('--mask-search-pattern', type=str, required=True,
                       help='Glob pattern for mask images')
    parser.add_argument('--output-folder', type=str, default=None,
                       help='Folder to save plots. If not specified, displays interactively.')
    parser.add_argument('--mode', type=str, default='group1',
                       help='Processing mode: "all" = all files, "examples[N]" = first, middle, last N files (default: 3), '
                            '"first[N]" = first N files (default: 3), "random[N]" = N random files (default: 3), '
                            '"group[N]" = N samples per experimental group (default: group1)')
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
    parser.add_argument('--qc-key', type=str, default=None,
                       help='Enable interactive QC mode with this key (e.g., "nuc_segmentation"). '
                            'Allows navigation with arrow keys and QC recording with mouse clicks.')
    parser.add_argument('--yaml-search-pattern', type=str, default=None,
                       help='Optional glob pattern for metadata YAML files. If provided, ROI positions will be extracted '
                            'and displayed as markers on the plots.')
    
    args = parser.parse_args()


    # Setup logging - use INFO level to see ROI extraction details
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Debug: Show what channel argument was received
    if args.channel is None:
        logging.info("Channel argument: None (will show ALL channels)")
    else:
        logging.info(f"Channel argument: {args.channel} (will show only this channel)")
    
    # Create output folder if needed
    if args.output_folder:
        os.makedirs(args.output_folder, exist_ok=True)
        logging.info(f"Output folder: {args.output_folder}")
    
    # Use standardized file grouping function
    search_patterns = {
        'image': args.input_search_pattern,
        'mask': args.mask_search_pattern
    }
    
    # Add optional YAML pattern if provided
    if args.yaml_search_pattern:
        search_patterns['yaml'] = args.yaml_search_pattern
    
    grouped_files = rp.get_grouped_files_to_process(search_patterns, args.search_subfolders)
    
    if not grouped_files:
        raise FileNotFoundError(f"No matching image-mask pairs found!")
    
    # Convert grouped files to list of tuples (image, mask, yaml)
    file_pairs = []
    for basename, files in grouped_files.items():
        if 'image' not in files:
            logging.warning(f"Missing image for basename '{basename}', skipping.")
            continue
        if 'mask' not in files:
            logging.warning(f"Missing mask for basename '{basename}', skipping.")
            continue
        # YAML is optional
        yaml_path = files.get('yaml', None)
        file_pairs.append((files['image'], files['mask'], yaml_path))
    
    if not file_pairs:
        raise ValueError(f"No complete image-mask pairs found!")
    
    logging.info(f"Matched {len(file_pairs)} image-mask pairs")
    
    # Parse mode and count
    mode = str(args.mode).lower()  # Convert to string in case YAML passes boolean
    n_files = len(file_pairs)
    
    # Extract mode type and count
    import re
    mode_match = re.match(r'(all|examples|first|random|group)(\d+)?$', mode)
    if not mode_match:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'all', 'examples[N]', 'first[N]', 'random[N]', or 'group[N]'")
    
    mode_type = mode_match.group(1)
    mode_count = int(mode_match.group(2)) if mode_match.group(2) else (1 if mode_type == 'group' else 3)
    
    # Select files based on mode
    if mode_type == 'group':
        # Group mode: select N samples per experimental group
        from collections import defaultdict
        
        # Import the extract_experimental_group function from quantify_distance_heatmap
        import sys
        from pathlib import Path
        parent_dir = Path(__file__).parent
        sys.path.insert(0, str(parent_dir))
        from quantify_distance_heatmap import extract_experimental_group
        
        groups = defaultdict(list)
        for pair in file_pairs:
            img_path = pair[0]  # First element is the image path
            group = extract_experimental_group(img_path)
            groups[group].append(pair)
        
        logging.info(f"Found {len(groups)} experimental groups")
        
        selected_pairs = []
        for group_name, group_pairs in sorted(groups.items()):
            n_available = len(group_pairs)
            n_select = min(mode_count, n_available)
            
            # Select evenly spaced samples from the group
            if n_available <= mode_count:
                selected = group_pairs
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
                selected = [group_pairs[i] for i in indices]
            
            selected_pairs.extend(selected)
            logging.info(f"  Group '{group_name}': selected {len(selected)}/{n_available} samples")
    
    elif mode_type == 'all':
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
    
    # Check if interactive QC mode is enabled
    if args.qc_key:
        logging.info(f"\n{'='*60}")
        logging.info(f"Starting Interactive QC Mode with key: {args.qc_key}")
        logging.info(f"Processing {len(selected_pairs)} image-mask pairs")
        logging.info(f"{'='*60}")
        
        interactive_qc(
            file_pairs=selected_pairs,
            qc_key=args.qc_key,
            input_search_pattern=args.input_search_pattern,
            mask_search_pattern=args.mask_search_pattern,
            output_folder=args.output_folder,
            alpha=args.alpha,
            channel=args.channel,
            z_slice=args.z_slice,
            colormap=args.colormap
        )
        return
    
    # Process each pair
    for img_path, mask_path, yaml_path in selected_pairs:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing: {Path(img_path).name}")
        
        # Extract ROI positions if YAML provided
        roi_positions = extract_roi_positions(yaml_path) if yaml_path else None
        
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
                colormap=args.colormap,
                roi_positions=roi_positions
            )
        except Exception as e:
            logging.error(f"Error processing {Path(img_path).name}: {e}")
            continue
    
    logging.info(f"\n{'='*60}")
    logging.info("Processing complete!")
    
    # Clean up matplotlib/tkinter resources to prevent cleanup errors
    try:
        plt.close('all')
        import gc
        gc.collect()
    except:
        pass


if __name__ == "__main__":
    main()
    

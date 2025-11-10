from typing import Union, Optional
from bioio import BioImage
import os
import tempfile
import bioio_ome_tiff, bioio_tifffile, bioio_nd2, bioio_lif, bioio_czi, bioio_dv
import numpy as np
import warnings



# Suppress all cryptography-related warnings (TripleDES, Blowfish deprecations from paramiko)
warnings.filterwarnings('ignore', category=Warning)
warnings.simplefilter('ignore')


def _configure_bioformats_safe_io(input_path: str) -> None:
    """Best-effort: prevent Bio-Formats from touching the source folder.

    Bio-Formats may write .bfmemo cache files next to inputs (e.g., .ims).
    We try to disable memoization or at least redirect it to a temp dir.
    """
    # Only apply for file types typically handled by Bio-Formats where sidecar writes are common
    if not input_path.lower().endswith((".ims", ".czi", ".lif", ".nd2", ".oib", ".oif")):
        return

    tmp_dir = os.environ.get("BIOFORMATS_MEMO_DIR", None) or tempfile.gettempdir()

    # Set a variety of known env vars / system properties consulted by wrappers
    os.environ.setdefault("BIOFORMATS_DISABLE_MEMOIZATION", "1")
    os.environ.setdefault("OME_BIOFORMATS_MEMOIZER_DISABLED", "1")
    os.environ.setdefault("LOCI_FORMATS_MEMOIZER_DISABLED", "1")
    os.environ.setdefault("BIOFORMATS_MEMO_DIR", tmp_dir)
    os.environ.setdefault("OME_BIOFORMATS_MEMOIZER_DIR", tmp_dir)
    os.environ.setdefault("LOCI_FORMATS_MEMOIZER_DIR", tmp_dir)

    # Reduce Java-side log verbosity (SLF4J/Logback/SciJava) before JVM starts
    try:
        import scyjava  # type: ignore
        # Prepare a minimal Logback configuration to clamp logs to WARN.
        logback_path = os.path.join(tmp_dir, "bioformats-logback.xml")
        if not os.path.exists(logback_path):
            try:
                with open(logback_path, "w", encoding="utf-8") as fh:
                    fh.write(
                        """
<configuration>
    <contextListener class="ch.qos.logback.classic.jul.LevelChangePropagator">
        <resetJUL>true</resetJUL>
    </contextListener>

    <appender name="STDOUT" class="ch.qos.logback.core.ConsoleAppender">
        <encoder>
            <pattern>%d{HH:mm:ss.SSS} [%thread] %-5level %logger - %msg%n</pattern>
        </encoder>
    </appender>

    <logger name="loci" level="WARN"/>
    <logger name="ome" level="WARN"/>
    <logger name="org.scijava" level="WARN"/>
    <logger name="org.janelia" level="WARN"/>
    <logger name="net.imagej" level="WARN"/>

    <root level="WARN">
        <appender-ref ref="STDOUT"/>
    </root>
</configuration>
""".strip()
                    )
            except Exception:
                # If writing fails, continue with system properties below
                pass

        # Force Logback to use our configuration if available
        if os.path.exists(logback_path):
            scyjava.config.add_option(f"-Dlogback.configurationFile={logback_path}")

        # SLF4J simple logger (harmless if not the active binding)
        scyjava.config.add_option("-Dorg.slf4j.simpleLogger.defaultLogLevel=warn")
        scyjava.config.add_option("-Dorg.slf4j.simpleLogger.showDateTime=false")
        scyjava.config.add_option("-Dorg.slf4j.simpleLogger.showThreadName=false")
        # SciJava logger level
        scyjava.config.add_option("-Dscijava.log.level=WARN")
        # Fallback envs in case properties aren’t picked up by the active binding
        os.environ.setdefault("SCIJAVA_LOG_LEVEL", "WARN")
    except Exception:
        # If scyjava isn't available yet, the properties below may still be picked up via env when JVM starts
        os.environ.setdefault("org.slf4j.simpleLogger.defaultLogLevel", "warn")
        os.environ.setdefault("scijava.log.level", "WARN")


def load_tczyx_image(path: str) -> BioImage:
    """
    Load an image as a BioImage object, ensuring the data is always 5D (TCZYX).
    The file format is determined by the file extension. This function standardizes
    all images to TCZYX order for safe downstream processing.


    example use:
    from bioio import BioImage

    # Get a BioImage object
    img = BioImage("my_file.tiff")  # selects the first scene found
    img.data  # returns 5D TCZYX numpy array
    img.xarray_data  # returns 5D TCZYX xarray data array backed by numpy
    img.dims  # returns a Dimensions object
    img.dims.order  # returns string "TCZYX"
    img.dims.X  # returns size of X dimension
    img.shape  # returns tuple of dimension sizes in TCZYX order
    img.get_image_data("CZYX", T=0)  # returns 4D CZYX numpy array

    """
    # Load the image using the appropriate reader based on the file extension
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    
    if path.endswith(".tif"):
        try: # Will work for ometif
            img = BioImage(path, reader=bioio_ome_tiff.Reader)
            return img
        except Exception:
            pass
        try: 
            img = BioImage(path, reader=bioio_tifffile.Reader)
            return img
        except Exception:
            pass
        try: 
            img = BioImage(path)
            return img
        except Exception:
            pass
        try: # For imageJ tifs
            img = BioImage(path)
            return img
        except Exception:
            pass
    elif path.endswith(".nd2"):
        img = BioImage(path, reader=bioio_nd2.Reader)
        return img
    elif path.endswith(".lif"):
        img = BioImage(path, reader=bioio_lif.Reader)
        return img
    elif path.endswith(".czi"):
        img = BioImage(path, reader=bioio_czi.Reader)
        return img
    elif path.endswith(".dv"):
        img = BioImage(path, reader=bioio_dv.Reader)
        return img
    elif path.endswith(".ims"):
        # Try custom bioio_imaris reader first (faster, pure Python)
        try:
            from standard_code.python.bioio_imaris import Reader as ImarisReader
            img = BioImage(path, reader=ImarisReader)
            return img
        except Exception:
            pass
        # Fall back to Bio-Formats if bioio_imaris fails
        _configure_bioformats_safe_io(path)
        try:
            import bioio_bioformats  # type: ignore
            img = BioImage(path, reader=bioio_bioformats.Reader)
            return img
        except Exception:
            pass
    else:
        _configure_bioformats_safe_io(path)
        import bioio_bioformats  # type: ignore
        img = BioImage(path, reader=bioio_bioformats.Reader)
        return img
    raise ValueError(f"Unsupported file format for: {path}")

# Deprecated alias for backward compatibility
def load_bioio(path: str) -> BioImage:
    import warnings
    warnings.warn("load_bioio is deprecated, use load_tczyx_image instead.", DeprecationWarning)
    return load_tczyx_image(path)


def save_tczyx_image(img: Union[BioImage, np.ndarray], path: str, **kwargs) -> None:
    """
    Save a BioImage or numpy array to disk as OME-TIFF, ensuring TCZYX order.
    This function should be used for all image saving to guarantee consistency.
    """
    try:
        from bioio.writers import OmeTiffWriter
    except ImportError:
        from bioio_ome_tiff import OmeTiffWriter
    # If BioImage, extract .data; if np.ndarray, use as is
    arr = getattr(img, 'data', img)
    # Ensure 5D TCZYX
    import numpy as np
    arr = np.asarray(arr)
    while arr.ndim < 5:
        arr = arr[np.newaxis, ...]
    # Remove dim_order from kwargs if present to avoid multiple values error
    if "dim_order" in kwargs:
        kwargs.pop("dim_order")

    ome_xml = None
    # Only try to preserve OME-XML if input is BioImage
    from bioio import BioImage
    if isinstance(img, BioImage):
        if hasattr(img, 'ome_xml') and img.ome_xml is not None:
            ome_xml = img.ome_xml
        elif hasattr(img, 'metadata') and isinstance(img.metadata, dict):
            ome_xml = img.metadata.get('ome_xml', None)

    # force overwrite if file exists
    if os.path.exists(path):
        os.remove(path)
    if ome_xml is not None:

        OmeTiffWriter.save(arr, path, dim_order="TCZYX", ome_xml=ome_xml, **kwargs)
    else:
        OmeTiffWriter.save(arr, path, dim_order="TCZYX", **kwargs)

# Deprecated alias for backward compatibility
def save_bioio(img, path, **kwargs):
    import warnings
    warnings.warn("save_bioio is deprecated, use save_tczyx_image instead.", DeprecationWarning)
    return save_tczyx_image(img, path, **kwargs)

def get_files_to_process(folder_path: str, extension: str, search_subfolders: bool) -> list:
    """
    Get a list of files in the specified folder with the specified extension.
    WARNING: This function will be deprecated in the future. Please use get_files_to_process2 with a glob pattern instead.
    """
    import warnings
    warnings.warn("get_files_to_process will be deprecated in the future. Please use get_files_to_process2 with a glob pattern instead.", FutureWarning)
    files_to_process = []
    if search_subfolders:
        for dirpath, _, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith(extension):
                    files_to_process.append(os.path.join(dirpath, filename))
    else:
        with os.scandir(folder_path) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(extension):
                    files_to_process.append(entry.path)
    files_to_process = [file_path.replace("\\", "/") for file_path in files_to_process]
    files_to_process = sorted(files_to_process)
    return files_to_process

def get_files_to_process2(search_pattern: str, search_subfolders: bool) -> list:
    """
    Get a list of files matching a glob pattern. Example: 'folder/*.tif' or 'folder/somefile*.tif'.
    If search_subfolders is True, will use '**' for recursive search if not already present in the pattern.
    Returns a sorted list of file paths with forward slashes.
    """
    import glob
    import os
    # If recursive search requested and not already in pattern, add '**/'
    if search_subfolders and '**' not in search_pattern:
        parts = os.path.split(search_pattern)
        search_pattern = os.path.join(parts[0], '**', parts[1])
    files_to_process = glob.glob(search_pattern, recursive=search_subfolders)
    files_to_process = [file_path.replace("\\", "/") for file_path in files_to_process]
    files_to_process = sorted(files_to_process)
    return files_to_process

def collapse_filename(file_path: str, base_folder: str, delimiter: str = "__") -> str:
    """
    Collapse a full file path into a single string filename that encodes the relative
    path structure using a custom delimiter.

    Parameters:
    - file_path: The full path to the file.
    - base_folder: The base directory from which the relative path is derived.
    - delimiter: The string used to replace path separators (default: "__").

    Returns:
    - A string that represents the relative path as a flat filename, with
      directory separators replaced by the delimiter.
    """
    # Compute the path relative to the base folder
    rel_path = os.path.relpath(file_path, start=base_folder)
    
    # Replace os-specific separators with the chosen delimiter
    collapsed = delimiter.join(rel_path.split(os.sep))
    
    return collapsed

def uncollapse_filename(collapsed: str, base_folder: str, delimiter: str = "__") -> str:
    """
    Reconstruct the original file path from a collapsed filename.
    
    Parameters:
    - collapsed: The collapsed filename string.
    - base_folder: The base directory to prepend to the reconstructed path.
    - delimiter: The delimiter used in the collapse_filename function.
    
    Returns:
    - The reconstructed original file path.
    """
    parts = collapsed.split(delimiter)
    rel_path = os.path.join(*parts)

    original_path:str = os.path.join(base_folder, rel_path)
    return original_path

# ...existing code...

def get_grouped_files_to_process(
    search_patterns: dict[str, str],
    search_subfolders: bool
) -> dict[str, dict[str, str]]:
    """
    Group files from multiple search patterns by their common basename.
    
    This function finds files matching multiple glob patterns and groups them by
    the portion of the filename that matches the '*' wildcard. This is useful when
    you have related files with different suffixes or in different locations.
    
    Parameters:
    -----------
    search_patterns : dict[str, str]
        Dictionary mapping pattern names to glob patterns. The patterns should
        contain a '*' wildcard that will be used for matching. The pattern name
        becomes the key in the nested result dictionary.
        Example: {
            'image': 'input/*.tif',
            'mask': 'masks/*_mask.tif',
            'tracking': 'tracking/*_tracked.tif'
        }
    
    search_subfolders : bool
        If True, searches recursively using '**' pattern.
    
    Returns:
    --------
    dict[str, dict[str, str]]
        Nested dictionary where:
        - Outer key: basename (the part matching '*')
        - Inner key: pattern name (from input dict)
        - Inner value: full file path
        
        Example result: {
            'image001': {
                'image': 'input/image001.tif',
                'mask': 'masks/image001_mask.tif',
                'tracking': 'tracking/image001_tracked.tif'
            },
            'image002': {
                'image': 'input/image002.tif',
                'mask': 'masks/image002_mask.tif'
            }
        }
    
    Raises:
    -------
    ValueError
        If any pattern doesn't contain a '*' wildcard
        If pattern names are duplicated
    
    Notes:
    ------
    - Files are only included in groups where the basename matches
    - A group may have missing patterns if no matching file is found
    - Use this when you need to process related files together
    
    Examples:
    ---------
    >>> patterns = {
    ...     'image': './input/*.tif',
    ...     'mask': './masks/*_mask.tif'
    ... }
    >>> groups = get_grouped_files_to_process(patterns, search_subfolders=False)
    >>> for basename, files in groups.items():
    ...     if 'image' in files and 'mask' in files:
    ...         process_pair(files['image'], files['mask'])
    """
    import re
    import os
    
    # Validate inputs
    if not search_patterns:
        raise ValueError("search_patterns dictionary cannot be empty")
    
    # Check for duplicate pattern names
    if len(search_patterns) != len(set(search_patterns.keys())):
        raise ValueError("Pattern names must be unique")
    
    # Validate all patterns have '*'
    for name, pattern in search_patterns.items():
        if '*' not in pattern:
            raise ValueError(f"Pattern '{name}' must contain a '*' wildcard: {pattern}")
    
    # For each pattern, find files and extract basename
    pattern_files = {}  # pattern_name -> [(basename, full_path), ...]
    
    for pattern_name, pattern in search_patterns.items():
        files = get_files_to_process2(pattern, search_subfolders)
        
        # Split pattern at the last '*' to get prefix and suffix
        # This handles patterns like 'folder/**/*_metadata.yaml' correctly
        last_star_idx = pattern.rfind('*')
        prefix = pattern[:last_star_idx]
        suffix = pattern[last_star_idx + 1:]  # Everything after the last '*'
        
        basenames_and_paths = []
        for file_path in files:
            # Get just the filename (not full path)
            filename = os.path.basename(file_path)
            
            # Remove suffix from filename to get basename
            if suffix and filename.endswith(suffix):
                basename = filename[:-len(suffix)]
            elif suffix:
                # If suffix doesn't match exactly, try removing extension
                basename_no_ext = os.path.splitext(filename)[0]
                suffix_no_ext = os.path.splitext(suffix)[0]
                if suffix_no_ext and basename_no_ext.endswith(suffix_no_ext):
                    basename = basename_no_ext[:-len(suffix_no_ext)]
                else:
                    # Last resort: just remove file extension
                    basename = basename_no_ext
            else:
                # No suffix specified, use filename without extension
                basename = os.path.splitext(filename)[0]
            
            basenames_and_paths.append((basename, file_path))
        
        pattern_files[pattern_name] = basenames_and_paths
    
    # Group by basename
    grouped: dict[str, dict[str, str]] = {}
    
    for pattern_name, basename_path_list in pattern_files.items():
        for basename, file_path in basename_path_list:
            if basename not in grouped:
                grouped[basename] = {}
            grouped[basename][pattern_name] = file_path
    
    # Sort by basename for consistency
    grouped = dict(sorted(grouped.items()))
    
    return grouped




def split_comma_separated_strstring(value:str) -> list[str]:
    return list(map(str, value.split(',')))    

def split_comma_separated_intstring(value:str) -> list[int]:
    return list(map(int, value.split(',')))    


def show_image(
    image: Union[BioImage, np.ndarray, str],
    mask: Union[BioImage, np.ndarray, str, None] = None,
    title: Optional[str] = None,
    alpha: float = 0.5
) -> None:
    """
    Quick visualization of mask segmentation over time.
    
    Layout:
    - Top row: First and last timepoint overlays (max Z projection)
    - Bottom: Stacked area chart showing pixel counts per object ID across all timepoints
    
    Args:
        image: BioImage, numpy array (TCZYX), or path to image file
        mask: Optional mask as BioImage, numpy array (TCZYX), or path to mask file
        title: Optional title for the figure. If None, uses image filename if available.
        alpha: Transparency of mask overlay (0-1, default: 0.5)
    
    Examples:
        >>> show_image("path/to/image.tif", mask="path/to/mask.tif")
        >>> show_image(img_array, mask=mask_array, title="My Segmentation")
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from pathlib import Path
    
    # Load image if it's a path
    if isinstance(image, str):
        image_path = image
        image = load_tczyx_image(image)
    else:
        image_path = None
    
    # Convert to numpy array if BioImage
    if hasattr(image, 'data'):
        img_data = np.asarray(image.data)  # Force conversion from memoryview to numpy array
    else:
        img_data = np.asarray(image)
    
    # Ensure 5D
    while img_data.ndim < 5:
        img_data = img_data[np.newaxis, ...]
    
    T, C, Z, Y, X = img_data.shape
    
    # Load and process mask if provided
    mask_data = None
    mT = T  # Default to image timepoints
    if mask is not None:
        if isinstance(mask, str):
            mask = load_tczyx_image(mask)
        
        if hasattr(mask, 'data'):
            mask_data = np.asarray(mask.data)  # Force conversion from memoryview to numpy array
        else:
            mask_data = np.asarray(mask)
        
        # Ensure 5D
        while mask_data.ndim < 5:
            mask_data = mask_data[np.newaxis, ...]
        
        mT, mC, mZ, mY, mX = mask_data.shape
        
        # Validate dimensions
        if (Y, X) != (mY, mX):
            raise ValueError(f"Image and mask XY dimensions must match. Got image: ({Y}, {X}), mask: ({mY}, {mX})")
    
    # Create figure layout based on whether we have mask data
    if mask_data is not None:
        # With mask: 2x2 grid (top: first/last timepoint, bottom: area chart)
        fig = plt.figure(figsize=(10, 7))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.2)
        ax_first = fig.add_subplot(gs[0, 0])
        ax_last = fig.add_subplot(gs[0, 1])
        ax_chart = fig.add_subplot(gs[1, :])
    else:
        # Without mask: just show first and last timepoint side by side
        fig, (ax_first, ax_last) = plt.subplots(1, 2, figsize=(10, 4))
        ax_chart = None
    
    # Helper function to create distinct colors for each label
    def get_label_colors(max_label: int) -> dict:
        """Generate distinct colors for each label ID."""
        np.random.seed(42)
        colors = {}
        for label_id in range(1, max_label + 1):
            hue = (label_id * 0.618033988749895) % 1.0  # Golden ratio for good distribution
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
            
            colors[label_id] = (r, g, b)
        return colors
    
    # Get max projection of first channel for display
    img_first = np.max(img_data[0, 0, :, :, :], axis=0)
    img_last = np.max(img_data[-1, 0, :, :, :], axis=0)
    
    # Normalize intensities
    vmin = np.percentile(img_data[0, 0], 1)
    vmax = np.percentile(img_data[0, 0], 99)
    
    # Show first timepoint
    ax_first.imshow(img_first, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
    ax_first.set_title(f'T=0 (MIP)', fontsize=10)
    ax_first.axis('off')
    
    # Show last timepoint
    ax_last.imshow(img_last, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
    ax_last.set_title(f'T={T-1} (MIP)', fontsize=10)
    ax_last.axis('off')
    
    # Overlay masks if provided
    if mask_data is not None:
        # Get all unique labels across all timepoints
        all_labels = set()
        for t in range(mT):
            mask_mip = np.max(mask_data[t, 0, :, :, :], axis=0)
            all_labels.update(np.unique(mask_mip))
        all_labels.discard(0)  # Remove background
        max_label = max(all_labels) if all_labels else 0
        
        if max_label > 0:
            label_colors = get_label_colors(max_label)
            
            # Create colormap for overlays
            colors_list = [(0, 0, 0, 0)]  # Background transparent
            for i in range(1, max_label + 1):
                if i in label_colors:
                    r, g, b = label_colors[i]
                    colors_list.append((r, g, b, alpha))
                else:
                    colors_list.append((0, 0, 0, alpha))
            mask_cmap = ListedColormap(colors_list)
            
            # Overlay on first timepoint
            mask_first = np.max(mask_data[0, 0, :, :, :], axis=0)
            mask_overlay_first = np.ma.masked_where(mask_first == 0, mask_first)
            ax_first.imshow(mask_overlay_first, cmap=mask_cmap, alpha=1.0, interpolation='nearest',
                          vmin=0, vmax=max_label)
            n_obj_first = len(np.unique(mask_first)) - 1
            ax_first.set_title(f'T=0 ({n_obj_first} objects)', fontsize=10)
            
            # Overlay on last timepoint
            mask_last = np.max(mask_data[-1, 0, :, :, :], axis=0)
            mask_overlay_last = np.ma.masked_where(mask_last == 0, mask_last)
            ax_last.imshow(mask_overlay_last, cmap=mask_cmap, alpha=1.0, interpolation='nearest',
                         vmin=0, vmax=max_label)
            n_obj_last = len(np.unique(mask_last)) - 1
            ax_last.set_title(f'T={T-1} ({n_obj_last} objects)', fontsize=10)
            
            # Calculate pixel counts per label per timepoint (only if we have chart axis)
            if ax_chart is not None:
                timepoints = []
                pixel_counts = {label: [] for label in sorted(all_labels)}
                
                for t in range(mT):
                    mask_mip = np.max(mask_data[t, 0, :, :, :], axis=0)
                    timepoints.append(t)
                    
                    for label_id in sorted(all_labels):
                        count = np.sum(mask_mip == label_id)
                        pixel_counts[label_id].append(count)
                
                # Create stacked area chart
                bottom = np.zeros(len(timepoints))
                for label_id in sorted(all_labels):
                    counts = pixel_counts[label_id]
                    color = label_colors.get(label_id, (0.5, 0.5, 0.5))
                    ax_chart.fill_between(timepoints, bottom, bottom + counts, 
                                         color=color, alpha=0.7, label=f'ID {label_id}')
                    bottom += counts
                
                ax_chart.set_xlabel('Timepoint', fontsize=10)
                ax_chart.set_ylabel('Pixel Count', fontsize=10)
                ax_chart.set_title('Object Pixel Counts Over Time', fontsize=10)
                ax_chart.grid(True, alpha=0.3)
                
                # Only show legend if not too many labels
                if len(all_labels) <= 20:
                    ax_chart.legend(loc='upper left', fontsize=8, ncol=min(5, len(all_labels)))
    
    # Set figure title
    if title is None and image_path:
        title = Path(image_path).stem
    
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')
    
    # Show plot and block until user closes the window
    plt.show(block=True)


def save_imagej_roi(
    coordinates: np.ndarray,
    output_path: str,
    t: int = 0,
    c: int = 0,
    z: int = 0
) -> None:
    """
    Save a contour as an ImageJ ROI file.
    
    Args:
        coordinates: Nx2 array of (row, col) or (y, x) coordinates from find_contours
        output_path: Path where to save the .roi file
        t: Timepoint index (0-based)
        c: Channel index (0-based)
        z: Z-slice index (0-based)
    
    Example:
        >>> from skimage import measure
        >>> mask_single = (labeled == prop.label)
        >>> contours = measure.find_contours(mask_single, 0.5)
        >>> contour = max(contours, key=len)
        >>> save_imagej_roi(contour, "output.roi", t=0, c=0, z=5)
    """
    from roifile import ImagejRoi
    
    # Convert coordinates: find_contours returns (row, col), ImageJ expects (x, y)
    # So we need to flip: row,col -> col,row -> x,y
    coords_xy = np.fliplr(coordinates).astype(np.int16)
    
    if len(coords_xy) < 3:
        raise ValueError(f"ROI must have at least 3 points, got {len(coords_xy)}")
    
    # Create ROI and set position
    roi = ImagejRoi.frompoints(coords_xy)
    roi.position = z + 1  # ImageJ uses 1-based indexing for position
    
    # Save to file
    roi.tofile(output_path)


def save_imagej_rois_from_mask(
    mask: Union[np.ndarray, BioImage],
    output_path: str,
    name_pattern: str = "T{t}_C{c}_Z{z}_obj{label}"
) -> int:
    """
    Convert a labeled mask to individual ImageJ ROI files.
    
    Creates one .roi file per object in the mask. Each unique label (except 0)
    in each T,C,Z plane is converted to a separate ROI file.
    
    Args:
        mask: Labeled mask as numpy array (TCZYX or lower dimensions) or BioImage.
              Each unique integer value (except 0) represents a different object.
        output_path: Directory path where ROI files will be saved, or path to .zip file
                     for saving all ROIs in a single archive.
        name_pattern: Format string for ROI filenames. Available placeholders:
                      {t}, {c}, {z}, {label}. Only used if output_path is a directory.
                      Default: "T{t}_C{c}_Z{z}_obj{label}"
    
    Returns:
        Number of ROI files created
    
    Example:
        >>> # Save individual ROI files
        >>> count = save_imagej_rois_from_mask(
        ...     mask, 
        ...     "output_folder",
        ...     name_pattern="T{t}_C{c}_Z{z}_obj{label}.roi"
        ... )
        >>> print(f"Saved {count} ROI files")
        
        >>> # Save as a single zip archive
        >>> count = save_imagej_rois_from_mask(mask, "output_rois.zip")
    """
    from skimage import measure
    from roifile import ImagejRoi, roiwrite
    
    # Convert to numpy array if BioImage
    if hasattr(mask, 'data'):
        mask_data = np.asarray(mask.data)
    else:
        mask_data = np.asarray(mask)
    
    # Ensure 5D
    while mask_data.ndim < 5:
        mask_data = mask_data[np.newaxis, ...]
    
    T, C, Z, Y, X = mask_data.shape
    
    # Determine if we're saving to a zip or individual files
    save_as_zip = output_path.lower().endswith('.zip')
    
    if not save_as_zip:
        os.makedirs(output_path, exist_ok=True)
    
    rois = []
    roi_count = 0
    
    for t in range(T):
        for c in range(C):
            for z in range(Z):
                plane = mask_data[t, c, z]
                labels = np.unique(plane)
                
                for label in labels:
                    if label == 0:  # Skip background
                        continue
                    
                    # Create binary mask for this label
                    mask_single = (plane == label).astype(np.uint8)
                    
                    # Find contours
                    contours = measure.find_contours(mask_single, 0.5)
                    
                    if not contours:
                        continue
                    
                    # Use the largest contour
                    contour = max(contours, key=len)
                    
                    if len(contour) < 3:
                        continue
                    
                    # Convert to ImageJ format (flip y,x to x,y)
                    coords_xy = np.fliplr(contour).astype(np.int16)
                    roi = ImagejRoi.frompoints(coords_xy)
                    roi.position = z + 1  # ImageJ uses 1-based indexing
                    
                    if save_as_zip:
                        # Add to list for bulk save
                        rois.append(roi)
                    else:
                        # Save individual file
                        roi_name = name_pattern.format(t=t, c=c, z=z, label=int(label))
                        if not roi_name.endswith('.roi'):
                            roi_name += '.roi'
                        roi_path = os.path.join(output_path, roi_name)
                        roi.tofile(roi_path)
                    
                    roi_count += 1
    
    # Save as zip if requested
    if save_as_zip and rois:
        roiwrite(output_path, rois)
    
    return roi_count


if __name__ == "__main__":
    # # Example usage
    folder_path = r"Z:\Schink\Oyvind\biphub_user_data\6849908 - IMB - Coen - Sarah - Photoconv\input_tif"
    # extension = ".tif"
    # search_subfolders = False

    # files_to_process = get_files_to_process(folder_path, extension, search_subfolders)
    # print("Files to process:", files_to_process)

    # for file_path in files_to_process:
    #     collapsed_name = collapse_filename(file_path, folder_path)
    #     print("Collapsed filename:", collapsed_name)
    #     original_path = uncollapse_filename(collapsed_name, folder_path)
    #     print("Original path:", original_path)

    # # Load a BioImage object
    #     img = load_bioio(file_path)
    #     print(img.shape)

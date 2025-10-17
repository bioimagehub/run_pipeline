from typing import Union
from bioio import BioImage
import os
import tempfile
import bioio_ome_tiff, bioio_tifffile, bioio_nd2
import numpy as np


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
    elif path.endswith(".ims"):
        try:
            from bioio_imaris import Reader as ImarisReader
            img = BioImage(path, reader=ImarisReader)
            return img
        except Exception:
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
        
        # Extract the part before the '*' to use as prefix
        prefix_match = re.search(r'(.*)\*', pattern)
        prefix = prefix_match.group(1) if prefix_match else ''
        
        # Extract the part after the '*' to use as suffix
        suffix_match = re.search(r'\*(.*)', pattern)
        suffix = suffix_match.group(1) if suffix_match else ''
        
        basenames_and_paths = []
        for file_path in files:
            # Get the part of filename that matched the '*'
            if prefix:
                # Get relative path from prefix
                if file_path.startswith(prefix):
                    rel_path = file_path[len(prefix):]
                else:
                    # Prefix might be relative, try basename matching
                    rel_path = os.path.basename(file_path)
            else:
                rel_path = os.path.basename(file_path)
            
            # Remove suffix if present
            if suffix:
                # Handle file extension in suffix
                if rel_path.endswith(suffix):
                    basename = rel_path[:-len(suffix)]
                else:
                    # Try without extension matching
                    basename_no_ext = os.path.splitext(rel_path)[0]
                    suffix_no_ext = os.path.splitext(suffix)[0]
                    if basename_no_ext.endswith(suffix_no_ext):
                        basename = basename_no_ext[:-len(suffix_no_ext)]
                    else:
                        basename = rel_path
            else:
                basename = os.path.splitext(rel_path)[0]
            
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

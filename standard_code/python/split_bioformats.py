"""
Split multi-dimensional images using BioFormats (bfconvert).
Splits images into individual T, C, Z slices and saves them to a folder.

Supports all formats readable by BioFormats including:
- OME-TIFF, standard TIFF
- Zeiss (LSM, CZI)
- Leica (LEI, LIF)
- Olympus (OIB, OIF)
- Nikon (ND2)
- And many others

MIT License - BIPHUB, University of Oslo
"""

import os
import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
import shutil
import shlex
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logger = logging.getLogger(__name__)


def build_output_pattern(input_file: str, output_folder: str, split_string: str) -> str:
    """Compose bfconvert output pattern including base name and optional split token."""
    base_name = Path(input_file).stem
    split_part = (split_string or "").strip()
    # Normalize to single % tokens for bfconvert regardless of user-provided escaping
    split_part = split_part.replace("%%", "%")

    if split_part:
        separator = "_" if not split_part.startswith(("_", "-", ".")) else ""
        filename = f"{base_name}{separator}{split_part}.ome.tif"
    else:
        filename = f"{base_name}.ome.tif"

    return os.path.join(output_folder, filename)


def escape_for_windows_batch(text: str, nested: bool = False) -> str:
    """Escape % tokens for cmd.exe batch files.

    If calling a nested batch (e.g., bfconvert.bat), percents are parsed twice.
    Use '%%%%' to ensure the final program receives a single '%'.
    """
    if "%" not in text:
        return text
    return text.replace("%", "%%%%" if nested else "%%")


def translate_template_to_bfconvert(template: str | None) -> str:
    """
    Translate brace-based template tokens to bfconvert percent tokens.

    Supported tokens:
    - {s}: series
    - {c}: channel
    - {z}: Z plane
    - {t}: timepoint

    This avoids `%` in YAML and is converted here.
    """
    if not template:
        return ""
    t = template
    # Accept uppercase variants too
    replacements = {
        "{s}": "%s", "{S}": "%s",
        "{c}": "%c", "{C}": "%c",
        "{z}": "%z", "{Z}": "%z",
        "{t}": "%t", "{T}": "%t",
    }
    for k, v in replacements.items():
        t = t.replace(k, v)
    return t


def find_bfconvert() -> str:
    """
    Find bfconvert executable in the standard locations.
    
    Returns:
        Path to bfconvert executable
        
    Raises:
        FileNotFoundError: If bfconvert cannot be found
    """
    # Check if running from repo context
    repo_root = Path(__file__).parent.parent.parent
    potential_paths = [
        repo_root / "external" / "bftools" / "bfconvert.bat",
        repo_root / "external" / "bftools" / "bfconvert",
        shutil.which("bfconvert"),
        shutil.which("bfconvert.bat"),
    ]
    
    for path in potential_paths:
        if path and os.path.exists(path):
            logger.info(f"Found bfconvert at: {path}")
            return str(path)
    
    raise FileNotFoundError(
        "bfconvert not found. Ensure BioFormats is installed or available in the repo at "
        "external/bftools/bfconvert.bat"
    )


def check_java_installed() -> bool:
    """
    Check if Java is installed and accessible.
    
    Returns:
        True if Java is found, False otherwise
        
    Raises:
        RuntimeError: If Java is not found with installation instructions
    """
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        # Java prints version to stderr
        version_output = result.stderr + result.stdout
        
        if result.returncode == 0 and "version" in version_output.lower():
            # Extract version info
            version_line = version_output.split('\n')[0]
            logger.info(f"Java found: {version_line}")
            return True
        else:
            raise RuntimeError("Java command executed but returned unexpected output")
            
    except FileNotFoundError:
        error_msg = """
╔════════════════════════════════════════════════════════════════════════════╗
║                          JAVA NOT FOUND                                    ║
╚════════════════════════════════════════════════════════════════════════════╝

BioFormats tools (bfconvert, showinf) require Java to be installed.

INSTALLATION INSTRUCTIONS:

Windows:
  1. Download Java from: https://adoptium.net/temurin/releases/
  2. Choose the latest LTS version (Java 17 or 21)
  3. Download the Windows .msi installer
  4. Run the installer and check "Set JAVA_HOME variable"
  5. Restart your terminal/command prompt after installation
  6. Verify installation: java -version

macOS:
  1. Using Homebrew: brew install openjdk
  2. Or download from: https://adoptium.net/temurin/releases/
  3. Verify installation: java -version

Linux:
  Ubuntu/Debian: sudo apt-get install default-jdk
  Fedora/RHEL:   sudo dnf install java-17-openjdk
  Verify installation: java -version

After installation, restart your terminal and try again.
"""
        logger.error(error_msg)
        raise RuntimeError("Java is not installed or not in PATH. See installation instructions above.")
    
    except subprocess.TimeoutExpired:
        raise RuntimeError("Java check timed out. Java may be installed but not responding.")
    
    except Exception as e:
        raise RuntimeError(f"Error checking for Java: {e}")


def get_image_dimensions(input_file: str) -> Tuple[int, int, int, int]:
    """
    Get image dimensions using bfconvert showinf.
    
    Args:
        input_file: Path to image file
        
    Returns:
        Tuple of (T, C, Z, Y, X) dimensions
    """
    repo_root = Path(__file__).parent.parent.parent
    showinf_path = repo_root / "external" / "bftools" / "showinf.bat"
    
    if not os.path.exists(showinf_path):
        showinf_path = shutil.which("showinf")
    
    if not showinf_path:
        logger.warning("showinf not found, cannot determine image dimensions")
        return (1, 1, 1, None, None)
    
    try:
        result = subprocess.run(
            [str(showinf_path), "-nopix", input_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Parse output to find dimensions
        lines = result.stdout.split('\n')
        dimensions = {'T': 1, 'C': 1, 'Z': 1, 'Y': None, 'X': None}
        
        for line in lines:
            # showinf outputs dimensions like "SizeX = 2048" or "SizeT = 1"
            # Also handle "Width = 2048" and "Height = 2048"
            # Also handle old format "Dimensions: T=1 C=3 Z=5 Y=512 X=512"
            for dim in ['T', 'C', 'Z']:
                # Try "SizeX = value" format first (standard showinf output)
                if f'Size{dim}' in line and '=' in line:
                    try:
                        # Extract value after '=' - handle both "SizeX = 2048" and "SizeX=2048"
                        parts = line.split('=')
                        if len(parts) >= 2:
                            val_str = parts[1].strip().split()[0].strip()
                            val = int(val_str)
                            dimensions[dim] = val
                    except (IndexError, ValueError) as e:
                        logger.debug(f"Failed to parse {dim} from line: {line} - {e}")
                # Also try "X=value" format (compact format)
                elif f'{dim}=' in line:
                    try:
                        val = int(line.split(f'{dim}=')[1].split()[0])
                        dimensions[dim] = val
                    except (IndexError, ValueError):
                        pass
            
            # Handle X and Y dimensions - can be "Width/Height" or "SizeX/SizeY"
            if 'Width' in line and '=' in line and dimensions['X'] is None:
                try:
                    parts = line.split('=')
                    if len(parts) >= 2:
                        val_str = parts[1].strip().split()[0].strip()
                        dimensions['X'] = int(val_str)
                except (IndexError, ValueError):
                    pass
            elif 'SizeX' in line and '=' in line and dimensions['X'] is None:
                try:
                    parts = line.split('=')
                    if len(parts) >= 2:
                        val_str = parts[1].strip().split()[0].strip()
                        dimensions['X'] = int(val_str)
                except (IndexError, ValueError):
                    pass
            
            if 'Height' in line and '=' in line and dimensions['Y'] is None:
                try:
                    parts = line.split('=')
                    if len(parts) >= 2:
                        val_str = parts[1].strip().split()[0].strip()
                        dimensions['Y'] = int(val_str)
                except (IndexError, ValueError):
                    pass
            elif 'SizeY' in line and '=' in line and dimensions['Y'] is None:
                try:
                    parts = line.split('=')
                    if len(parts) >= 2:
                        val_str = parts[1].strip().split()[0].strip()
                        dimensions['Y'] = int(val_str)
                except (IndexError, ValueError):
                    pass
        
        logger.info(f"Parsed dimensions: T={dimensions['T']}, C={dimensions['C']}, Z={dimensions['Z']}, Y={dimensions['Y']}, X={dimensions['X']}")
        return (dimensions['T'], dimensions['C'], dimensions['Z'], 
                dimensions['Y'], dimensions['X'])
    
    except Exception as e:
        logger.warning(f"Error reading image dimensions: {e}")
        return (1, 1, 1, None, None)


def split_single_file(
    input_file: str,
    output_folder: str,
    bfconvert_path: str,
    split_string: str = "",
    open_terminal: bool = True,
    show_progress: bool = True,
    suppress_warnings: bool = True
) -> bool:
    """
    Split a single image file using bfconvert.
    
    Args:
        input_file: Path to input image file
        output_folder: Path to output folder
        bfconvert_path: Path to bfconvert executable
        split_string: Optional template for output filenames
        open_terminal: Whether to open a new terminal for bfconvert
        show_progress: Whether to show progress
        suppress_warnings: Whether to suppress Java non-fatal warnings (default: True)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get image dimensions
        t, c, z, y, x = get_image_dimensions(input_file)
        logger.info(f"Image dimensions: T={t}, C={c}, Z={z}, Y={y}, X={x}")
        
        output_pattern = build_output_pattern(input_file, output_folder, split_string)
        
        # Build command
        cmd = [
            str(bfconvert_path),
            "-padded",  # Zero-pad filename indexes (1 -> 001)
            "-no-upgrade",  # Don't upgrade file format
            "-noflat",  # Don't flatten RGB
            "-overwrite",  # Overwrite existing files without prompting
            "-swap", "XYZCT",  # Set dimension order for ImageJ compatibility
            str(input_file),
            output_pattern
        ]
        
        # For logging: if bfconvert is a .bat file, show escaped version
        is_bat = str(bfconvert_path).lower().endswith((".bat", ".cmd"))
        if is_bat and sys.platform == "win32":
            log_cmd = [escape_for_windows_batch(x, nested=True) if '%' in x else x for x in cmd]
            logger.info(f"Running command (with batch escaping): {' '.join(log_cmd)}")
        else:
            logger.info(f"Running command: {' '.join(cmd)}")
        
        if open_terminal:
            # Open new terminal window on Windows
            if sys.platform == "win32":
                # Create a batch file to run in terminal
                # Use folder name as prefix to avoid conflicts when processing multiple files
                folder_name = os.path.basename(output_folder)
                batch_file = os.path.join(output_folder, f"{folder_name}_run_split.bat")
                done_marker = os.path.join(output_folder, f"{folder_name}_conversion_done.txt")
                
                # Remove old done marker if it exists
                if os.path.exists(done_marker):
                    os.remove(done_marker)
                
                with open(batch_file, 'w') as f:
                    f.write("@echo off\n")
                    f.write(f"cd /d \"{output_folder}\"\n")
                    
                    # Set Java options as environment variables if suppressing warnings
                    if suppress_warnings:
                        f.write("rem Set Java options to reduce warnings\n")
                        f.write("set BF_MAX_MEM=4g\n")
                        f.write("set JAVA_OPTS=-Xmx4g --enable-native-access=ALL-UNNAMED -XX:+IgnoreUnrecognizedVMOptions\n")
                    
                    f.write(f"echo Converting image to individual slices...\n")
                    # If bfconvert is a .bat/.cmd, escape percents for nested batch parsing
                    nested = str(bfconvert_path).lower().endswith((".bat", ".cmd"))
                    cmd_str = " ".join([
                        f'"{escape_for_windows_batch(x, nested=nested)}"' if " " in x else escape_for_windows_batch(x, nested=nested)
                        for x in cmd
                    ])
                    f.write(cmd_str + "\n")
                    f.write("set EXITCODE=%ERRORLEVEL%\n")
                    f.write(f"echo %EXITCODE% > \"{done_marker}\"\n")
                    f.write("if %EXITCODE% EQU 0 (\n")
                    f.write("    echo Conversion complete!\n")
                    f.write(") else (\n")
                    f.write("    echo Conversion failed with error code %EXITCODE%\n")
                    f.write(")\n")
                    f.write("pause\n")
                
                # Execute batch file in new terminal
                logger.info(f"Starting conversion in new terminal window...")
                subprocess.Popen(
                    ["cmd.exe", "/c", "start", "cmd.exe", "/k", batch_file],
                    cwd=output_folder
                )
                
                # Wait for the done marker file to appear
                import time
                max_wait = 3600  # Maximum 1 hour
                check_interval = 2  # Check every 2 seconds
                elapsed = 0
                
                logger.info(f"Waiting for conversion to complete (checking every {check_interval}s)...")
                while elapsed < max_wait:
                    if os.path.exists(done_marker):
                        # Read the exit code
                        try:
                            with open(done_marker, 'r') as f:
                                exit_code = int(f.read().strip())
                            
                            if exit_code == 0:
                                logger.info(f"✓ Conversion completed successfully in terminal")
                                return True
                            else:
                                logger.error(f"✗ Conversion failed with exit code: {exit_code}")
                                return False
                        except Exception as e:
                            logger.error(f"Error reading exit code: {e}")
                            return False
                    
                    time.sleep(check_interval)
                    elapsed += check_interval
                
                logger.error(f"Timeout waiting for conversion to complete (waited {max_wait}s)")
                return False
            else:
                # Linux/macOS - run in background and wait
                process = subprocess.Popen(
                    ["gnome-terminal", "--", "bash", "-c", f"{shlex.join(cmd)}; bash"],
                    cwd=output_folder
                )
                logger.info(f"Opened terminal for conversion, waiting for completion...")
                process.wait()
                logger.info(f"Terminal process completed")
                return True
        else:
            # Run in current process
            # Set environment variables for Java if suppressing warnings
            env = os.environ.copy()
            if suppress_warnings:
                env['BF_MAX_MEM'] = '4g'
                env['JAVA_OPTS'] = '-Xmx4g --enable-native-access=ALL-UNNAMED -XX:+IgnoreUnrecognizedVMOptions'
            
            # When calling .bat files on Windows via subprocess, we need to escape % signs
            # because subprocess invokes them through cmd.exe
            exec_cmd = cmd
            if is_bat and sys.platform == "win32":
                exec_cmd = [escape_for_windows_batch(x, nested=False) for x in cmd]
                logger.info(f"Escaped command for batch execution: {' '.join(exec_cmd)}")
            
            result = subprocess.run(
                exec_cmd,
                capture_output=True,
                text=True,
                timeout=None,
                env=env
            )
            
            if result.returncode != 0:
                logger.error(f"bfconvert failed: {result.stderr}")
                return False
            
            logger.info(f"Successfully split image: {input_file}")
            if show_progress and result.stdout:
                logger.info(result.stdout)
            return True
    
    except Exception as e:
        logger.error(f"Error splitting file {input_file}: {e}")
        return False


def _process_single_worker(args: Tuple[str, str, str, str, str, bool, bool, bool, bool, bool, str, str]) -> Tuple[str, bool]:
    """
    Worker function for parallel processing. Must be at module level for Windows pickling.
    
    Args:
        args: Tuple of (input_file, output_folder, bfconvert_path, split_string, 
              split_template, open_terminal, dry_run, suppress_warnings, 
              create_subfolders, preserve_paths, base_folder, collapse_delimiter)
    
    Returns:
        Tuple of (input_file, success)
    """
    input_file, output_folder, bfconvert_path, split_string, split_template, open_terminal, dry_run, suppress_warnings, create_subfolders, preserve_paths, base_folder, collapse_delimiter = args
    
    # Determine output folder structure
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    if output_folder:
        if create_subfolders:
            # Create subfolders per input file
            if preserve_paths:
                # Preserve original directory structure
                rel_path = os.path.relpath(input_file, base_folder)
                rel_dir = os.path.dirname(rel_path)
                file_output_folder = os.path.join(output_folder, rel_dir, base_name)
            else:
                # Flatten structure using collapse_filename pattern
                try:
                    import bioimage_pipeline_utils as rp
                    collapsed = rp.collapse_filename(input_file, base_folder, collapse_delimiter)
                    collapsed_base = os.path.splitext(collapsed)[0]
                    file_output_folder = os.path.join(output_folder, collapsed_base)
                except ImportError:
                    # Fallback: simple collapse using delimiter
                    rel_path = os.path.relpath(input_file, base_folder)
                    collapsed = rel_path.replace(os.sep, collapse_delimiter)
                    collapsed_base = os.path.splitext(collapsed)[0]
                    file_output_folder = os.path.join(output_folder, collapsed_base)
        else:
            # Default: All files go directly into output_folder
            file_output_folder = output_folder
    else:
        # Create folder in same directory as input
        input_dir = os.path.dirname(input_file)
        file_output_folder = os.path.join(input_dir, base_name)
    
    logger.info(f"Processing: {input_file}")
    logger.info(f"Output folder: {file_output_folder}")
    
    if dry_run:
        logger.info(f"[DRY RUN] Would split: {input_file} -> {file_output_folder}")
        return (input_file, True)
    
    # Prefer brace-based template over raw split string
    effective_split = split_string or translate_template_to_bfconvert(split_template)

    success = split_single_file(
        input_file,
        file_output_folder,
        bfconvert_path,
        split_string=effective_split,
        open_terminal=open_terminal,
        show_progress=True,
        suppress_warnings=suppress_warnings
    )
    return (input_file, success)


def process_files(
    input_pattern: str,
    output_folder: str = None,
    split_string: str = "",
    split_template: str = "",
    open_terminal: bool = True,
    no_parallel: bool = False,
    dry_run: bool = False,
    suppress_warnings: bool = True,
    create_subfolders: bool = False,
    preserve_paths: bool = False,
    collapse_delimiter: str = "__"
) -> None:
    """
    Process multiple image files and split them.
    
    Args:
        input_pattern: Glob pattern for input files
        output_folder: Base output folder (default: same as input with '_split' suffix)
        split_string: Optional split template string
        split_template: Brace-based split template
        open_terminal: Whether to open terminal for each conversion
        no_parallel: Process files sequentially instead of parallel
        dry_run: Print planned actions without executing
        suppress_warnings: Whether to suppress Java non-fatal warnings (default: True)
        create_subfolders: Create a subfolder per input file (default: False, all files in output_folder)
        preserve_paths: When creating subfolders, preserve directory structure (default: False, flatten with delimiter)
        collapse_delimiter: Delimiter for flattening paths when create_subfolders=True (default: '__')
    """
    from glob import glob
    
    # Expand glob pattern
    input_files = glob(input_pattern, recursive=True)
    
    if not input_files:
        logger.warning(f"No files found matching pattern: {input_pattern}")
        return
    
    logger.info(f"Found {len(input_files)} file(s) to process")
    
    # Determine base folder for path collapsing
    if '**' in input_pattern:
        base_folder = input_pattern.split('**')[0].rstrip('/\\')
        if not base_folder:
            base_folder = os.getcwd()
        base_folder = os.path.abspath(base_folder)
    else:
        base_folder = str(Path(input_files[0]).parent) if input_files else os.getcwd()
    
    logger.info(f"Base folder for path handling: {base_folder}")
    if create_subfolders:
        if preserve_paths:
            logger.info(f"Creating subfolders preserving directory structure")
        else:
            logger.info(f"Creating subfolders with flattened paths (delimiter: '{collapse_delimiter}')")
    else:
        logger.info(f"All output files will go directly into: {output_folder}")
    
    # Check Java installation first
    try:
        check_java_installed()
    except RuntimeError as e:
        logger.error(f"Cannot proceed without Java: {e}")
        return
    
    # Find bfconvert
    try:
        bfconvert_path = find_bfconvert()
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Execute processing
    if no_parallel or len(input_files) == 1:
        # Sequential processing
        for input_file in input_files:
            args = (input_file, output_folder, bfconvert_path, split_string, 
                   split_template, open_terminal, dry_run, suppress_warnings,
                   create_subfolders, preserve_paths, base_folder, collapse_delimiter)
            _process_single_worker(args)
    else:
        # Parallel processing with worker function
        args_list = [
            (f, output_folder, bfconvert_path, split_string, split_template, open_terminal, dry_run, suppress_warnings,
             create_subfolders, preserve_paths, base_folder, collapse_delimiter)
            for f in input_files
        ]
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(_process_single_worker, args): args[0] for args in args_list}
            for future in as_completed(futures):
                try:
                    input_file, success = future.result()
                    if success:
                        logger.info(f"✓ Successfully processed: {input_file}")
                    else:
                        logger.error(f"✗ Failed to process: {input_file}")
                except Exception as e:
                    logger.error(f"Error in parallel processing: {e}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Split multi-dimensional images using BioFormats (bfconvert). "
                    "Saves individual T, C, Z slices to a folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
Example YAML config for run_pipeline.exe:
---
run:
- name: Split images (default - all files in output folder)
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/split_bioformats.py'
  - --input-search-pattern: '%YAML%/input/**/*.nd2'
  - --output-folder: '%YAML%/output'
  - --split-template: 'S{s}_C{c}_Z{z}_T{t}'
  - --no-terminal
  # Output: All split files directly in output/
  # Example: output/file1_S0_C0_Z0_T0.ome.tif, output/file1_S0_C1_Z0_T0.ome.tif

- name: Split with subfolders (flattened paths)
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/split_bioformats.py'
  - --input-search-pattern: '%YAML%/input/**/*.czi'
  - --output-folder: '%YAML%/output'
  - --create-subfolders
  - --no-terminal
  # Output: input/sub1/sub2/file.czi -> output/sub1__sub2__file/file_S0_C0.ome.tif

- name: Split with subfolders (preserve directory structure)
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/split_bioformats.py'
  - --input-search-pattern: '%YAML%/input/**/*.nd2'
  - --output-folder: '%YAML%/output'
  - --create-subfolders
  - --preserve-paths
  - --no-terminal
  # Output: input/sub1/file.nd2 -> output/sub1/file/file_S0_C0.ome.tif

- name: Dry run (preview what would happen)
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/split_bioformats.py'
  - --input-search-pattern: '%YAML%/input_images/**/*.tif'
  - --dry-run
"""
        )
    )
    
    parser.add_argument(
        "--input-search-pattern",
        type=str,
        required=True,
        help="Input file pattern (supports wildcards, use '**' for recursive search)"
    )
    
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output base folder (default: input_folder + '_split'). "
             "By default, all split files go directly into this folder. "
             "Use --create-subfolders to organize files into subfolders per input file."
    )

    parser.add_argument(
        "--split-string",
        type=str,
        default="",
        help=(
            "Optional bfconvert split tokens to insert between the base filename and .ome.tif. "
            "Use bfconvert placeholders like %s (series), %c (channel), %z (Z), %t (time). "
            "Example: S%s_C%c_Z%z_T%t. Default: none (just convert to base_name.ome.tif)."
        )
    )

    parser.add_argument(
        "--split-template",
        type=str,
        default="",
        help=(
            "Brace-based split template to avoid YAML percent issues. "
            "Use tokens {s},{c},{z},{t} e.g. 'S{s}_C{c}_Z{z}_T{t}'. "
            "This will be translated to bfconvert placeholders internally."
        )
    )
    
    parser.add_argument(
        "--open-terminal",
        action="store_true",
        default=True,
        help="Open new terminal window for bfconvert (shows progress, default: True)"
    )
    
    parser.add_argument(
        "--no-terminal",
        action="store_true",
        help="Run bfconvert in background without terminal window"
    )
    
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing (process files sequentially)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without executing"
    )
    
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Show Java warnings (by default, non-actionable Java warnings are suppressed)"
    )
    
    parser.add_argument(
        "--create-subfolders",
        action="store_true",
        help="Create a subfolder per input file to organize output (default: all files in output folder)"
    )
    
    parser.add_argument(
        "--preserve-paths",
        action="store_true",
        help="When using --create-subfolders, preserve directory structure (default: flatten with delimiter)"
    )
    
    parser.add_argument(
        "--collapse-delimiter",
        type=str,
        default="__",
        help="Delimiter for flattening subfolder paths when using --create-subfolders (default: '__')"
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit"
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.version:
        version_file = Path(__file__).parent.parent.parent / "VERSION"
        try:
            version = version_file.read_text(encoding='utf-8').strip()
        except Exception:
            version = "unknown"
        print(f"split_bioformats.py version: {version}")
        return
    
    # Process files
    open_terminal = not args.no_terminal if args.no_terminal else args.open_terminal
    
    process_files(
        input_pattern=args.input_search_pattern,
        output_folder=args.output_folder,
        split_string=args.split_string,
        split_template=args.split_template,
        open_terminal=open_terminal,
        no_parallel=args.no_parallel,
        dry_run=args.dry_run,
        suppress_warnings=not args.show_warnings,
        create_subfolders=args.create_subfolders,
        preserve_paths=args.preserve_paths,
        collapse_delimiter=args.collapse_delimiter
    )


if __name__ == "__main__":
    main()

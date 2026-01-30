"""
Simple CLI wrapper for BioFormats bfconvert tool.

Provides a convenient interface to run bfconvert on multiple files with any flags.
Handles Java installation checks and bfconvert path detection.

Supports all formats readable by BioFormats including:
- OME-TIFF, standard TIFF
- Zeiss (LSM, CZI)
- Leica (LEI, LIF)
- Olympus (OIB, OIF)
- Nikon (ND2)
- And many others

MIT License - BIPHUB, University of Oslo
"""
import argparse
import logging
import os
import subprocess
import textwrap
from pathlib import Path
from glob import glob
import shutil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Dict, Any

try:
    import bioimage_pipeline_utils as rp
except ImportError:
    rp = None

# Configure logging
logger = logging.getLogger(__name__)


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


def find_showinf() -> str:
    """
    Find showinf executable in the standard locations.
    
    Returns:
        Path to showinf executable
        
    Raises:
        FileNotFoundError: If showinf cannot be found
    """
    # Check if running from repo context
    repo_root = Path(__file__).parent.parent.parent
    potential_paths = [
        repo_root / "external" / "bftools" / "showinf.bat",
        repo_root / "external" / "bftools" / "showinf",
        shutil.which("showinf"),
        shutil.which("showinf.bat"),
    ]
    
    for path in potential_paths:
        if path and os.path.exists(path):
            logger.debug(f"Found showinf at: {path}")
            return str(path)
    
    raise FileNotFoundError(
        "showinf not found. Ensure BioFormats is installed or available in the repo at "
        "external/bftools/showinf.bat"
    )


def get_series_count(input_file: str) -> int:
    """
    Get number of series in an image using showinf.
    
    Args:
        input_file: Path to image file
        
    Returns:
        Number of series in the image (minimum 1)
    """
    try:
        showinf_path = find_showinf()
    except FileNotFoundError:
        logger.warning("showinf not found, assuming 1 series")
        return 1
    
    try:
        result = subprocess.run(
            [str(showinf_path), "-nopix", input_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Parse output to find series count
        lines = result.stdout.split('\n')
        series_count = 1  # Default to 1 if not found
        
        for line in lines:
            # Look for "Series count = N" or "SeriesCount = N"
            if 'series count' in line.lower() and '=' in line:
                try:
                    parts = line.split('=')
                    if len(parts) >= 2:
                        val_str = parts[1].strip().split()[0].strip()
                        series_count = int(val_str)
                        logger.info(f"Detected {series_count} series in {Path(input_file).name}")
                        break
                except (IndexError, ValueError) as e:
                    logger.debug(f"Failed to parse series count from line: {line} - {e}")
        
        return max(1, series_count)  # Ensure at least 1
    
    except Exception as e:
        logger.warning(f"Error reading series count: {e}, assuming 1 series")
        return 1


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


def run_bfconvert(
    input_file: str,
    output_file: str,
    bfconvert_path: str,
    extra_flags: list[str] = None,
    series: int = None,
    max_memory: str = "8g"
) -> bool:
    """
    Run bfconvert on a single file.
    
    Args:
        input_file: Path to input image file
        output_file: Path to output file
        bfconvert_path: Path to bfconvert executable
        extra_flags: Additional flags to pass to bfconvert
        series: Series number to extract (None = all series)
        max_memory: Maximum Java heap size (default: 8g for 8GB)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Build command
        cmd = [str(bfconvert_path)]
        
        # Add series flag if specified
        if series is not None:
            cmd.extend(["-series", str(series)])
        
        # Add user-specified flags
        if extra_flags:
            cmd.extend(extra_flags)
        
        # Add input and output
        cmd.extend([str(input_file), output_file])
        
        # Set up environment with increased Java heap size
        env = os.environ.copy()
        env['BF_MAX_MEM'] = max_memory
        env['_JAVA_OPTIONS'] = f'-Xmx{max_memory} -Xms{max_memory}'
        
        logger.info(f"Running with max memory: {max_memory}")
        logger.info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=None,
            env=env
        )
        
        if result.returncode != 0:
            logger.error(f"bfconvert failed for {input_file}:")
            logger.error(result.stderr)
            return False
        
        logger.info(f"✓ Successfully converted: {input_file} -> {output_file}")
        if result.stdout and result.stdout.strip():
            logger.debug(result.stdout)
        return True
    
    except Exception as e:
        logger.error(f"Error converting file {input_file}: {e}")
        return False


def process_single_file(args: Tuple[str, str, str, list, int, str]) -> Dict[str, Any]:
    """
    Process a single file (wrapper for parallel processing).
    
    Args:
        args: Tuple of (input_file, output_file, bfconvert_path, extra_flags, series, max_memory)
        
    Returns:
        Dict with processing results
    """
    input_file, output_file, bfconvert_path, extra_flags, series, max_memory = args
    
    try:
        success = run_bfconvert(
            input_file=input_file,
            output_file=output_file,
            bfconvert_path=bfconvert_path,
            extra_flags=extra_flags,
            series=series,
            max_memory=max_memory
        )
        return {
            'input': input_file,
            'output': output_file,
            'series': series,
            'success': success,
            'error': None
        }
    except Exception as e:
        logger.error(f"Failed to process {input_file} series {series}: {e}")
        return {
            'input': input_file,
            'output': output_file,
            'series': series,
            'success': False,
            'error': str(e)
        }


def process_files(
    input_pattern: str,
    output_folder: str,
    output_suffix: str = ".ome.tif",
    extra_flags: list[str] = None,
    dry_run: bool = False,
    create_subfolders: bool = False,
    max_memory: str = "8g",
    max_workers: int = None,
    no_parallel: bool = False
) -> None:
    """
    Process multiple image files with bfconvert.
    
    Args:
        input_pattern: Glob pattern for input files
        output_folder: Output folder path
        output_suffix: Suffix to append to input basename (default: .ome.tif)
        extra_flags: Additional flags to pass to bfconvert
        dry_run: Print planned actions without executing
        create_subfolders: Create a subfolder per input file (default: False)
        max_memory: Maximum Java heap size (default: 8g for 8GB)
        max_workers: Number of parallel workers (default: CPU count - 1, use None for auto)
        no_parallel: Disable parallel processing (default: False, parallel is enabled)
    """
    # Expand glob pattern
    input_files = glob(input_pattern, recursive=True)
    
    if not input_files:
        logger.warning(f"No files found matching pattern: {input_pattern}")
        return
    
    logger.info(f"Found {len(input_files)} file(s) to process")
    
    # Determine base folder for path collapsing
    # If pattern has **, use the part before ** as base folder
    if '**' in input_pattern:
        base_folder = input_pattern.split('**')[0].rstrip('/\\')
        if not base_folder:
            base_folder = os.getcwd()
        base_folder = os.path.abspath(base_folder)
    else:
        # Use parent directory of first file as base
        base_folder = str(Path(input_files[0]).parent) if input_files else os.getcwd()
    
    logger.info(f"Base folder for path handling: {base_folder}")
    if create_subfolders:
        logger.info(f"Creating subfolders per input file")
    
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
    
    # Determine optimal number of workers for parallel processing
    if not no_parallel:
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave 1 core for system
        logger.info(f"Using {max_workers} parallel workers")
    else:
        logger.info("Parallel processing disabled")
    
    # Build task list
    tasks = []
    for input_file in input_files:
        # Use collapse_filename to encode directory structure in filename
        if rp is not None:
            # Use bioimage_pipeline_utils for proper path collapsing
            collapsed = rp.collapse_filename(input_file, base_folder, delimiter="__")
            collapsed_base = os.path.splitext(collapsed)[0]
        else:
            # Fallback: simple collapse using delimiter
            rel_path = os.path.relpath(input_file, base_folder)
            collapsed = rel_path.replace(os.sep, "__")
            collapsed_base = os.path.splitext(collapsed)[0]
        
        # ALWAYS detect series count to force splitting by series
        series_count = get_series_count(input_file)
        
        logger.info(f"Processing: {input_file}")
        
        # Process each series separately
        for series_idx in range(series_count):
            # Add _S{n} suffix if there are multiple series
            if series_count > 1:
                series_suffix = f"_S{series_idx}"
            else:
                series_suffix = ""
            
            # Determine output path based on subfolder option
            if create_subfolders:
                # Create a subfolder per output file (includes series suffix in folder name)
                file_output_folder = str(Path(output_folder) / f"{collapsed_base}{series_suffix}")
                output_file = str(Path(file_output_folder) / f"{collapsed_base}{series_suffix}{output_suffix}")
            else:
                # All files directly in output folder
                output_file = str(Path(output_folder) / f"{collapsed_base}{series_suffix}{output_suffix}")
            
            if series_count > 1:
                logger.info(f"  Series {series_idx}/{series_count-1}: {output_file}")
            else:
                logger.info(f"  Output: {output_file}")
            
            if dry_run:
                logger.info(f"[DRY RUN] Would convert: {input_file} (series {series_idx}) -> {output_file}")
                logger.info(f"[DRY RUN] Flags: {extra_flags}")
                logger.info(f"[DRY RUN] Max memory: {max_memory}")
            else:
                tasks.append((
                    input_file,
                    output_file,
                    bfconvert_path,
                    extra_flags,
                    series_idx,
                    max_memory
                ))
    
    if dry_run:
        logger.info(f"\n[DRY RUN] Would process {len(tasks)} tasks")
        if not no_parallel:
            logger.info(f"[DRY RUN] Using {max_workers} parallel workers")
        return
    
    # Execute tasks (parallel or sequential)
    success_count = 0
    fail_count = 0
    
    if no_parallel:
        # Sequential processing
        logger.info(f"Processing {len(tasks)} tasks sequentially...")
        for i, task in enumerate(tasks, 1):
            result = process_single_file(task)
            if result['success']:
                success_count += 1
                logger.info(f"✓ [{i}/{len(tasks)}] {result['output']}")
            else:
                fail_count += 1
                logger.error(f"✗ [{i}/{len(tasks)}] {result['input']} series {result['series']}")
    else:
        # Parallel processing
        logger.info(f"Processing {len(tasks)} tasks across {max_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_single_file, task): task for task in tasks}
            
            # Process results as they complete
            for future in as_completed(futures):
                result = future.result()
                
                if result['success']:
                    success_count += 1
                    logger.info(f"✓ [{success_count + fail_count}/{len(tasks)}] {result['output']}")
                else:
                    fail_count += 1
                    logger.error(f"✗ [{success_count + fail_count}/{len(tasks)}] {result['input']} series {result['series']}")
    
    logger.info(f"\nCompleted: {success_count} successful, {fail_count} failed")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Simple wrapper for BioFormats bfconvert tool. "
                    "Pass any bfconvert flags directly to the tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
Example YAML config for run_pipeline.exe:
---
run:
- name: Convert to OME-TIFF (parallel processing by default)
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/bfconvert_wrapper.py'
  - --input-pattern: '%YAML%/input/**/*.nd2'
  - --output-folder: '%YAML%/output'
  - --output-suffix: '.ome.tif'
  - --flag=-padded
  - --flag=-no-upgrade
  - --flag=-overwrite
  # Uses CPU count - 1 workers by default

- name: Convert with custom worker count
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/bfconvert_wrapper.py'
  - --input-pattern: '%YAML%/input/**/*.czi'
  - --output-folder: '%YAML%/output'
  - --max-workers: 4
  - --flag=-padded
  - --flag=-overwrite

- name: Convert sequentially (no parallel)
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/bfconvert_wrapper.py'
  - --input-pattern: '%YAML%/input/**/*.lif'
  - --output-folder: '%YAML%/output'
  - --no-parallel
  - --flag=-padded
  - --flag=-overwrite

- name: Split by series
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/bfconvert_wrapper.py'
  - --input-pattern: '%YAML%/input/**/*.lif'
  - --output-folder: '%YAML%/output'
  - --output-suffix: '_S%%s.ome.tif'
  - --flag=-padded
  - --flag=-overwrite

- name: Split by series (with subfolders)
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/bfconvert_wrapper.py'
  - --input-pattern: '%YAML%/input/**/*.lif'
  - --output-folder: '%YAML%/output'
  - --output-suffix: '_S%%s.ome.tif'
  - --flag=-padded
  - --flag=-overwrite
  - --create-subfolders
  # Input: input/sub1/file.lif -> Output: output/sub1__file/sub1__file_S0.ome.tif

- name: Split by channel and timepoint
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/bfconvert_wrapper.py'
  - --input-pattern: '%YAML%/input/**/*.czi'
  - --output-folder: '%YAML%/output'
  - --output-suffix: '_C%%c_T%%t.ome.tif'
  - --flag=-series
  - --flag='0'
  - --flag=-padded

- name: Convert with compression
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/bfconvert_wrapper.py'
  - --input-pattern: '%YAML%/input/**/*.oib'
  - --output-folder: '%YAML%/output'
  - --output-suffix: '.ome.tif'
  - --flag=-compression
  - --flag=LZW
  - --flag=-overwrite

Common bfconvert flags:
  -series N               Select specific series
  -timepoint N            Select specific timepoint
  -channel N              Select specific channel
  -z N                    Select specific Z slice
  -range START END        Select range of planes
  -tilex N -tiley N       Set tile dimensions
  -compression TYPE       Set compression (LZW, JPEG-2000, JPEG, Uncompressed)
  -padded                 Zero-pad series/channel/z/t numbers
  -overwrite              Overwrite existing files
  -no-upgrade             Don't upgrade file format
  -noflat                 Don't flatten RGB images

For full bfconvert documentation, see:
https://docs.openmicroscopy.org/bio-formats/latest/users/comlinetools/conversion.html
"""
        )
    )
    
    parser.add_argument(
        "--input-pattern",
        type=str,
        required=True,
        help="Input file pattern (supports wildcards, use '**' for recursive search)"
    )
    
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output folder path"
    )
    
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=".ome.tif",
        help="Suffix to append to input basename. Can include bfconvert placeholders like "
             "%%c (channel), %%t (timepoint), %%z (Z), %%s (series). "
             "Examples: '_S%%s.ome.tif' or '_c%%c_t%%t.ome.tif' (default: .ome.tif)"
    )
    
    parser.add_argument(
        "--flag",
        action="append",
        dest="flags",
        metavar="ARG",
        help="Add a flag or argument to pass to bfconvert. Can be used multiple times. "
             "For flags starting with -, use quotes or =: --flag=-padded or --flag=\"-padded\". "
             "Example: --flag=-padded --flag=-overwrite --flag=-compression --flag=LZW"
    )
    
    parser.add_argument(
        "--create-subfolders",
        action="store_true",
        help="Create a subfolder per input file to organize output (default: all files in output folder)"
    )
    
    parser.add_argument(
        "--max-memory",
        type=str,
        default="8g",
        help="Maximum Java heap size for bfconvert (default: 8g). "
             "Examples: 4g, 8g, 16g, 32g. Increase if you get OutOfMemoryError."
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1). "
             "Set to 1 for sequential processing or use --no-parallel."
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
        print(f"bfconvert_wrapper.py version: {version}")
        return
    
    # Process files
    process_files(
        input_pattern=args.input_pattern,
        output_folder=args.output_folder,
        output_suffix=args.output_suffix,
        extra_flags=args.flags or [],
        dry_run=args.dry_run,
        create_subfolders=args.create_subfolders,
        max_memory=args.max_memory,
        max_workers=args.max_workers,
        no_parallel=args.no_parallel
    )


if __name__ == "__main__":
    main()

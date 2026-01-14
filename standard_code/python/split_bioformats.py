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
from concurrent.futures import ProcessPoolExecutor, as_completed

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
            if 'Dimensions' in line or 'Series' in line:
                # Look for pattern like "Dimensions: T=1 C=3 Z=5 Y=512 X=512"
                for dim in ['T', 'C', 'Z', 'Y', 'X']:
                    if f'{dim}=' in line:
                        try:
                            val = int(line.split(f'{dim}=')[1].split()[0])
                            dimensions[dim] = val
                        except (IndexError, ValueError):
                            pass
        
        return (dimensions['T'], dimensions['C'], dimensions['Z'], 
                dimensions['Y'], dimensions['X'])
    
    except Exception as e:
        logger.warning(f"Error reading image dimensions: {e}")
        return (1, 1, 1, None, None)


def split_single_file(
    input_file: str,
    output_folder: str,
    bfconvert_path: str,
    open_terminal: bool = True,
    show_progress: bool = True
) -> bool:
    """
    Split a single image file using bfconvert.
    
    Args:
        input_file: Path to input image file
        output_folder: Path to output folder
        bfconvert_path: Path to bfconvert executable
        open_terminal: Whether to open a new terminal for bfconvert
        show_progress: Whether to show progress
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get image dimensions
        t, c, z, y, x = get_image_dimensions(input_file)
        logger.info(f"Image dimensions: T={t}, C={c}, Z={z}, Y={y}, X={x}")
        
        # Construct bfconvert command
        # Use format strings for splitting: %%s for series, %%c for channel, %%z for Z, %%t for timepoint (double %% for Windows)
        # Use OME-TIFF format (.ome.tif) to preserve channel metadata and prevent NIS Elements from misinterpreting as RGB
        output_pattern = os.path.join(output_folder, "S%%s_C%%c_Z%%z_T%%t.ome.tif")
        
        cmd = [
            str(bfconvert_path),
            "-padded",  # Zero-pad filename indexes (1 -> 001)
            "-no-upgrade",  # Don't upgrade file format
            "-noflat",  # Don't flatten RGB
            str(input_file),
            output_pattern
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        if open_terminal:
            # Open new terminal window on Windows
            if sys.platform == "win32":
                # Create a batch file to run in terminal
                batch_file = os.path.join(output_folder, "_run_split.bat")
                with open(batch_file, 'w') as f:
                    f.write("@echo off\n")
                    f.write(f"cd /d \"{output_folder}\"\n")
                    f.write(f"echo Converting image to individual slices...\n")
                    # Escape % as %% for batch file (need %%%% to output %%)
                    cmd_str = " ".join([f'"{x}"' if " " in x else x for x in cmd])
                    cmd_str = cmd_str.replace("%%", "%%%%")
                    f.write(cmd_str + "\n")
                    f.write("echo Conversion complete!\n")
                    f.write("pause\n")
                
                # Execute batch file in new terminal
                subprocess.Popen(
                    ["cmd.exe", "/c", "start", "cmd.exe", "/k", batch_file],
                    cwd=output_folder
                )
                logger.info(f"Opened terminal for conversion in: {output_folder}")
                return True
            else:
                # Linux/macOS
                subprocess.Popen(
                    ["gnome-terminal", "--", "bash", "-c", f"{' '.join(cmd)}; bash"],
                    cwd=output_folder
                )
                logger.info(f"Opened terminal for conversion in: {output_folder}")
                return True
        else:
            # Run in current process
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=None
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


def process_files(
    input_pattern: str,
    output_folder: str = None,
    open_terminal: bool = True,
    no_parallel: bool = False,
    dry_run: bool = False
) -> None:
    """
    Process multiple image files and split them.
    
    Args:
        input_pattern: Glob pattern for input files
        output_folder: Base output folder (default: same as input with '_split' suffix)
        open_terminal: Whether to open terminal for each conversion
        no_parallel: Process files sequentially instead of parallel
        dry_run: Print planned actions without executing
    """
    from glob import glob
    
    # Expand glob pattern
    input_files = glob(input_pattern, recursive=True)
    
    if not input_files:
        logger.warning(f"No files found matching pattern: {input_pattern}")
        return
    
    logger.info(f"Found {len(input_files)} file(s) to process")
    
    # Find bfconvert
    try:
        bfconvert_path = find_bfconvert()
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Process each file
    def process_single(input_file: str) -> Tuple[str, bool]:
        # Create output folder with same name as input (without extension)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        if output_folder:
            file_output_folder = os.path.join(output_folder, base_name)
        else:
            # Create folder in same directory as input
            input_dir = os.path.dirname(input_file)
            file_output_folder = os.path.join(input_dir, base_name)
        
        logger.info(f"Processing: {input_file}")
        logger.info(f"Output folder: {file_output_folder}")
        
        if dry_run:
            logger.info(f"[DRY RUN] Would split: {input_file} -> {file_output_folder}")
            return (input_file, True)
        
        success = split_single_file(
            input_file,
            file_output_folder,
            bfconvert_path,
            open_terminal=open_terminal,
            show_progress=True
        )
        return (input_file, success)
    
    # Execute processing
    if no_parallel or len(input_files) == 1:
        for input_file in input_files:
            process_single(input_file)
    else:
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(process_single, f): f for f in input_files}
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
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Split image into slices (opens terminal)
  environment: uv@3.11:split-bioformats
  commands:
  - python
  - '%REPO%/standard_code/python/split_bioformats.py'
  - --input-search-pattern: '%YAML%/input_images/**/*.czi'
  - --open-terminal

- name: Split images (batch processing without terminal)
  environment: uv@3.11:split-bioformats
  commands:
  - python
  - '%REPO%/standard_code/python/split_bioformats.py'
  - --input-search-pattern: '%YAML%/input_images/**/*.nd2'
  - --output-folder: '%YAML%/output_split'
  - --no-terminal

- name: Dry run (preview what would happen)
  environment: uv@3.11:split-bioformats
  commands:
  - python
  - '%REPO%/standard_code/python/split_bioformats.py'
  - --input-search-pattern: '%YAML%/input_images/**/*.tif'
  - --dry-run
"""
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
             "Output files will be saved in subfolders named after input files."
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
        open_terminal=open_terminal,
        no_parallel=args.no_parallel,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

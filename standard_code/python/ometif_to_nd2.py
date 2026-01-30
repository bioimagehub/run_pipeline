"""
Convert OME-TIFF files to ND2 format using NIS-Elements.
Works after bfconvert_wrapper.py to convert individual OME-TIFF files to ND2.

Workflow:
1. Takes a search pattern for OME-TIFF files
2. Opens each file in NIS-Elements
3. Saves as ND2 (same basename, different extension)
4. Optionally deletes source OME-TIFF file

MIT License - BIPHUB, University of Oslo
"""

import os
import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional
from glob import glob
import tempfile
import time

# Configure logging
logger = logging.getLogger(__name__)


def find_nis_executable() -> str:
    """
    Find NIS-Elements executable.
    
    Returns:
        Path to NIS-Elements executable
        
    Raises:
        FileNotFoundError: If NIS-Elements cannot be found
    """
    potential_paths = [
        "C:\\Program Files\\NIS-Elements\\nis_ar.exe",
        "C:\\Program Files\\Nikon\\NIS-Elements\\Nikon.NIS-Elements.exe",
        "C:\\Program Files (x86)\\Nikon\\NIS-Elements\\Nikon.NIS-Elements.exe",
        "D:\\biphub\\Program_Files\\NIS\\Nikon.NIS-Elements.exe",
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            logger.info(f"Found NIS-Elements at: {path}")
            return path
    
    raise FileNotFoundError(
        "NIS-Elements executable not found. Checked: " + ", ".join(potential_paths)
    )


def create_nis_save_macro(output_nd2_path: str, done_marker_path: str) -> str:
    """
    Create a NIS-Elements macro that saves the currently open image as ND2.
    
    Args:
        output_nd2_path: Path where ND2 should be saved
        done_marker_path: Path to create a marker file when done
        
    Returns:
        NIS-Elements macro code as string
    """
    # Escape backslashes for macro
    escaped_path = output_nd2_path.replace("\\", "\\\\")
    # Use forward slashes for the done marker path (for Python within macro)
    done_marker_forward = done_marker_path.replace("\\", "/")
    
    macro = f"""// Auto-generated macro to save as ND2
ImageSaveAs("{escaped_path}", 14, 0);
CloseAllDocuments(0);
Python_RunString("import os; done_path = '{done_marker_forward}'; open(done_path, 'w').write('done')");
"""
    return macro


def convert_to_nd2(
    ometif_file: str,
    output_nd2_path: str,
    nis_exe_path: str,
    timeout: int = 7200
) -> bool:
    """
    Open OME-TIFF in NIS-Elements and save as ND2.
    
    Args:
        ometif_file: Path to OME-TIFF file
        output_nd2_path: Where to save the ND2 file
        nis_exe_path: Path to NIS-Elements executable
        timeout: Maximum time to wait for conversion (seconds)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create done marker file path in same directory as output
        output_dir = os.path.dirname(output_nd2_path)
        done_marker_path = os.path.join(output_dir, f".nis_done_{os.getpid()}.tmp")
        
        # Remove old done marker if it exists
        if os.path.exists(done_marker_path):
            os.remove(done_marker_path)
        
        # Create temporary macro file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mac', delete=False) as f:
            macro_path = f.name
            macro_content = create_nis_save_macro(output_nd2_path, done_marker_path)
            f.write(macro_content)
        
        logger.debug(f"Created macro file: {macro_path}")
        logger.info(f"Opening: {ometif_file}")
        logger.info(f"Saving as: {output_nd2_path}")
        
        # Open NIS-Elements with the file and macro
        # -f opens file, -mv runs macro after opening
        cmd = [nis_exe_path, "-f", ometif_file, "-mv", macro_path]
        
        logger.debug(f"Launching NIS-Elements: {' '.join(cmd)}")
        
        # Launch NIS-Elements in background (don't wait)
        subprocess.Popen(cmd)
        
        # Poll for done marker file
        poll_interval = 2
        elapsed = 0
        
        logger.info(f"Waiting for NIS-Elements to complete conversion (timeout: {timeout}s)...")
        while elapsed < timeout:
            time.sleep(poll_interval)
            elapsed += poll_interval
            
            if os.path.exists(done_marker_path):
                logger.info(f"✓ Conversion completed (took {elapsed}s)")
                break
            
            if elapsed % 20 == 0:  # Log every 20 seconds
                logger.info(f"Still waiting... ({elapsed}s elapsed)")
        else:
            # Timeout reached
            logger.error(f"✗ Timeout waiting for conversion ({timeout}s)")
            return False
        
        # Clean up marker file
        if os.path.exists(done_marker_path):
            try:
                os.remove(done_marker_path)
            except Exception:
                pass
        
        # Clean up macro file
        try:
            os.remove(macro_path)
            logger.debug(f"Cleaned up macro file: {macro_path}")
        except Exception as e:
            logger.warning(f"Could not delete macro file {macro_path}: {e}")
        
        # Check if ND2 file was created
        if os.path.exists(output_nd2_path):
            file_size = os.path.getsize(output_nd2_path) / (1024*1024)  # MB
            logger.info(f"✓ Successfully created ND2: {output_nd2_path} ({file_size:.2f} MB)")
            return True
        else:
            logger.error(f"✗ ND2 file was not created: {output_nd2_path}")
            return False
    
    except Exception as e:
        logger.error(f"Error during NIS-Elements conversion: {e}")
        return False


def process_file(
    ometif_file: str,
    nis_exe_path: str,
    output_folder: str = None,
    delete_source: bool = False,
    timeout: int = 7200
) -> bool:
    """
    Process a single OME-TIFF file: convert to ND2, optionally delete source.
    
    Args:
        ometif_file: Path to OME-TIFF file
        nis_exe_path: Path to NIS-Elements executable
        output_folder: Optional output folder (default: same as source)
        delete_source: Whether to delete the source OME-TIFF file after conversion
        timeout: Maximum time to wait for conversion (seconds)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Determine output path
        if output_folder:
            # Use specified output folder
            base_name = os.path.splitext(os.path.basename(ometif_file))[0]
            # Remove .ome if present in basename
            if base_name.endswith('.ome'):
                base_name = base_name[:-4]
            output_nd2_path = os.path.join(output_folder, f"{base_name}.nd2")
            # Create output folder if needed
            os.makedirs(output_folder, exist_ok=True)
        else:
            # Same directory as source file
            output_nd2_path = os.path.splitext(ometif_file)[0]
            # Remove .ome if present
            if output_nd2_path.endswith('.ome'):
                output_nd2_path = output_nd2_path[:-4]
            output_nd2_path += ".nd2"
        
        # Run NIS-Elements conversion
        success = convert_to_nd2(ometif_file, output_nd2_path, nis_exe_path, timeout=timeout)
        
        if success and delete_source:
            try:
                logger.info(f"Deleting source file: {ometif_file}")
                os.remove(ometif_file)
                logger.info(f"✓ Deleted source file")
            except Exception as e:
                logger.warning(f"Could not delete source file {ometif_file}: {e}")
        
        return success
    
    except Exception as e:
        logger.error(f"Error processing file {ometif_file}: {e}")
        return False


def process_files(
    input_pattern: str,
    output_folder: str = None,
    delete_source: bool = False,
    dry_run: bool = False,
    timeout: int = 7200
) -> None:
    """
    Process multiple OME-TIFF files.
    
    Args:
        input_pattern: Glob pattern for input OME-TIFF files
        output_folder: Optional output folder (default: same as source files)
        delete_source: Whether to delete source files after conversion
        dry_run: Print planned actions without executing
        timeout: Maximum time to wait for each conversion (seconds)
    """
    # Expand glob pattern
    input_files = sorted(glob(input_pattern, recursive=True))
    
    if not input_files:
        logger.warning(f"No files found matching pattern: {input_pattern}")
        return
    
    logger.info(f"Found {len(input_files)} file(s) to process")
    
    # Find NIS-Elements
    if not dry_run:
        try:
            nis_exe_path = find_nis_executable()
        except FileNotFoundError as e:
            logger.error(str(e))
            return
    else:
        nis_exe_path = "NIS-Elements.exe"
    
    # Process each file
    success_count = 0
    fail_count = 0
    
    for ometif_file in input_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {ometif_file}")
        
        if dry_run:
            if output_folder:
                base_name = os.path.splitext(os.path.basename(ometif_file))[0]
                if base_name.endswith('.ome'):
                    base_name = base_name[:-4]
                output_nd2_path = os.path.join(output_folder, f"{base_name}.nd2")
            else:
                output_nd2_path = os.path.splitext(ometif_file)[0]
                if output_nd2_path.endswith('.ome'):
                    output_nd2_path = output_nd2_path[:-4]
                output_nd2_path += ".nd2"
            
            logger.info(f"[DRY RUN] Would convert to: {output_nd2_path}")
            if delete_source:
                logger.info(f"[DRY RUN] Would delete source: {ometif_file}")
            success_count += 1
        else:
            if process_file(ometif_file, nis_exe_path, output_folder=output_folder, 
                          delete_source=delete_source, timeout=timeout):
                success_count += 1
            else:
                fail_count += 1
    
    logger.info(f"\n{'='*60}")
    logger.info(f"=== Processing Complete ===")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"{'='*60}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert OME-TIFF files to ND2 format using NIS-Elements. "
                    "Works on individual OME-TIFF files produced by bfconvert_wrapper.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Convert to OME-TIFF with automatic series splitting
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/bfconvert_wrapper.py'
  - --input-pattern: '%YAML%/input/**/*.lif'
  - --output-folder: '%YAML%/output'
  - --output-suffix: '.ome.tif'
  - --create-subfolders
  - --flag=-padded
  - --flag=-overwrite
  # This automatically splits multi-series files into separate OME-TIFFs

- name: Convert OME-TIFF to ND2
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/ometif_to_nd2.py'
  - --input-pattern: '%YAML%/output/**/*.ome.tif'
  - --delete-source
  # Converts all OME-TIFFs to ND2 and removes originals

- name: Convert with custom output folder
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/ometif_to_nd2.py'
  - --input-pattern: '%YAML%/ometif_files/**/*.ome.tif'
  - --output-folder: '%YAML%/nd2_files'
  - --timeout: 1200
  # Save ND2 files to different folder with longer timeout

- name: Dry run (preview conversion)
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/ometif_to_nd2.py'
  - --input-pattern: '%YAML%/output/**/*.ome.tif'
  - --dry-run
  # Preview what would be converted without executing

Example workflow (LIF -> OME-TIFF -> ND2):
  1. bfconvert_wrapper.py: Splits multi-series LIF into individual OME-TIFFs
  2. ometif_to_nd2.py: Converts each OME-TIFF to ND2 format
"""
    )
    
    parser.add_argument(
        "--input-pattern",
        type=str,
        required=True,
        help="Search pattern for OME-TIFF files. "
             "Uses glob syntax with ** for recursive search. "
             "Example: '%YAML%/output/**/*.ome.tif'"
    )
    
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder for ND2 files (default: same directory as source files)"
    )
    
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete source OME-TIFF file after successful ND2 conversion"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Maximum time to wait for each conversion in seconds (default: 7200 = 2 hours)"
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
        print(f"ometif_to_nd2.py version: {version}")
        return
    
    # Process files
    process_files(
        input_pattern=args.input_pattern,
        output_folder=args.output_folder,
        delete_source=args.delete_source,
        dry_run=args.dry_run,
        timeout=args.timeout
    )


if __name__ == "__main__":
    main()

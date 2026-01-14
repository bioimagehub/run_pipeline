"""
Convert split image series to ND2 format using NIS-Elements.
Works after split_bioformats.py to import split series and save as ND2.

Workflow:
1. Takes a search pattern for original image files
2. Finds corresponding split folders (created by split_bioformats.py)
3. Opens first slice file in NIS-Elements (auto-assembles entire series)
4. Saves as ND2 next to original file
5. Optionally deletes temporary split folder

MIT License - BIPHUB, University of Oslo
"""

import os
import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple
from glob import glob
import tempfile

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


def find_first_slice(folder_path: str) -> Optional[str]:
    """
    Find the first slice file in a split folder.
    Handles dynamic padding: S0_C0_Z0_T0, S00_C00_Z00_T00, S000_C000_Z000_T000, etc.
    
    Args:
        folder_path: Path to the split folder
        
    Returns:
        Path to first slice file
    """
    import re
    
    # List all files in the folder
    if not os.path.isdir(folder_path):
        return None
    
    files = os.listdir(folder_path)
    
    # Filter for slice files (S*_C*_Z*_T*.ome.tif or .tif)
    slice_files = [f for f in files if re.match(r'S\d+_C\d+_Z\d+_T\d+', f)]
    
    if not slice_files:
        logger.warning(f"No slice files found in: {folder_path}")
        return None
    
    # Sort and get the first one (S0/S00/S000..., then C0/C00/C000..., then Z0, then T0)
    slice_files.sort(key=lambda f: tuple(map(int, re.findall(r'\d+', f))))
    
    first_slice = os.path.join(folder_path, slice_files[0])
    logger.info(f"Found first slice: {first_slice}")
    return first_slice


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


def create_and_run_nis_conversion(
    first_slice_file: str,
    output_nd2_path: str,
    nis_exe_path: str,
    split_folder: str
) -> bool:
    """
    Create a macro file and open image in NIS-Elements with auto-save to ND2.
    
    Args:
        first_slice_file: Path to first slice file
        output_nd2_path: Where to save the ND2 file
        nis_exe_path: Path to NIS-Elements executable
        split_folder: Path to split folder (for done marker)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import time
        
        # Create done marker file path
        done_marker_path = os.path.join(split_folder, "done.txt")
        
        # Remove old done marker if it exists
        if os.path.exists(done_marker_path):
            os.remove(done_marker_path)
        
        # Create temporary macro file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mac', delete=False) as f:
            macro_path = f.name
            macro_content = create_nis_save_macro(output_nd2_path, done_marker_path)
            f.write(macro_content)
        
        logger.info(f"Created macro file: {macro_path}")
        logger.info(f"Opening NIS-Elements with: {first_slice_file}")
        logger.info(f"Will save as: {output_nd2_path}")
        
        # Open NIS-Elements with the file and macro
        # -f opens file, -mv runs macro after opening
        cmd = [nis_exe_path, "-f", first_slice_file, "-mv", macro_path]
        
        logger.info(f"Launching NIS-Elements: {' '.join(cmd)}")
        
        # Launch NIS-Elements in background (don't wait)
        subprocess.Popen(cmd)
        
        # Poll for done marker file (wait up to 10 minutes)
        max_wait_seconds = 600
        poll_interval = 2
        elapsed = 0
        
        logger.info("Waiting for NIS-Elements to complete conversion...")
        while elapsed < max_wait_seconds:
            time.sleep(poll_interval)
            elapsed += poll_interval
            
            if os.path.exists(done_marker_path):
                logger.info(f"✓ Conversion completed (detected done marker)")
                break
            
            if elapsed % 10 == 0:  # Log every 10 seconds
                logger.info(f"Still waiting... ({elapsed}s elapsed)")
        
        # Clean up marker file
        if os.path.exists(done_marker_path):
            try:
                os.remove(done_marker_path)
            except Exception:
                pass
        
        # Clean up macro file
        try:
            os.remove(macro_path)
            logger.info(f"Cleaned up macro file: {macro_path}")
        except Exception as e:
            logger.warning(f"Could not delete macro file {macro_path}: {e}")
        
        # Check if ND2 file was created
        if os.path.exists(output_nd2_path):
            logger.info(f"✓ Successfully created ND2: {output_nd2_path}")
            return True
        else:
            logger.error(f"✗ ND2 file was not created: {output_nd2_path}")
            return False
    
    except Exception as e:
        logger.error(f"Error during NIS-Elements conversion: {e}")
        return False


def process_file(
    input_file: str,
    nis_exe_path: str,
    delete_tmp: bool = False
) -> bool:
    """
    Process a single input file: find split folder, convert to ND2, optionally delete folder.
    
    Args:
        input_file: Path to original input file
        nis_exe_path: Path to NIS-Elements executable, split_folder, split_folder
        delete_tmp: Whether to delete the temporary split folder
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get base name without extension
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        input_dir = os.path.dirname(input_file)
        
        # Find the split folder
        split_folder = os.path.join(input_dir, base_name)
        
        if not os.path.isdir(split_folder):
            logger.error(f"Split folder not found: {split_folder}")
            return False
        
        logger.info(f"Found split folder: {split_folder}")
        
        # Find first slice
        first_slice = find_first_slice(split_folder)
        if not first_slice:
            logger.error(f"No slice files found in: {split_folder}")
            return False
        
        # Create ND2 output path (next to original file)
        output_nd2_path = os.path.splitext(input_file)[0] + ".nd2"
        
        # Run NIS-Elements conversion
        success = create_and_run_nis_conversion(first_slice, output_nd2_path, nis_exe_path, split_folder)
        
        if success and delete_tmp:
            try:
                import shutil
                logger.info(f"Deleting temporary split folder: {split_folder}")
                shutil.rmtree(split_folder)
                logger.info(f"✓ Deleted split folder")
            except Exception as e:
                logger.warning(f"Could not delete split folder {split_folder}: {e}")
        
        return success
    
    except Exception as e:
        logger.error(f"Error processing file {input_file}: {e}")
        return False


def process_files(
    input_pattern: str,
    delete_tmp: bool = False,
    dry_run: bool = False
) -> None:
    """
    Process multiple image files.
    
    Args:
        input_pattern: Glob pattern for input files (same as used for split_bioformats.py)
        delete_tmp: Whether to delete temporary split folders after conversion
        dry_run: Print planned actions without executing
    """
    # Expand glob pattern
    input_files = sorted(glob(input_pattern, recursive=True))
    
    if not input_files:
        logger.warning(f"No files found matching pattern: {input_pattern}")
        return
    
    logger.info(f"Found {len(input_files)} file(s) to process")
    
    # Find NIS-Elements
    try:
        nis_exe_path = find_nis_executable()
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Process each file
    success_count = 0
    fail_count = 0
    
    for input_file in input_files:
        logger.info(f"\nProcessing: {input_file}")
        
        if dry_run:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            input_dir = os.path.dirname(input_file)
            split_folder = os.path.join(input_dir, base_name)
            output_nd2_path = os.path.splitext(input_file)[0] + ".nd2"
            
            logger.info(f"[DRY RUN] Would process:")
            logger.info(f"  Split folder: {split_folder}")
            logger.info(f"  Output ND2: {output_nd2_path}")
            if delete_tmp:
                logger.info(f"  Then delete: {split_folder}")
            success_count += 1
        else:
            if process_file(input_file, nis_exe_path, delete_tmp=delete_tmp):
                success_count += 1
            else:
                fail_count += 1
    
    logger.info(f"\n=== Processing Complete ===")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {fail_count}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert split image series to ND2 format using NIS-Elements. "
                    "Operates on folders created by split_bioformats.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Split images with BioFormats
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/split_bioformats.py'
  - --input-search-pattern: '%YAML%/input_images/**/*.lif'

- name: Convert split series to ND2
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/split_to_nd2.py'
  - --input-search-pattern: '%YAML%/input_images/**/*.lif'
  - --delete-tmp-folder

- name: Dry run (preview conversion)
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/split_to_nd2.py'
  - --input-search-pattern: '%YAML%/input_images/**/*.nd2'
  - --dry-run
"""
    )
    
    parser.add_argument(
        "--input-search-pattern",
        type=str,
        required=True,
        help="Search pattern for original image files (same as used in split_bioformats.py). "
             "Uses glob syntax with ** for recursive search."
    )
    
    parser.add_argument(
        "--delete-tmp-folder",
        action="store_true",
        help="Delete the temporary split folder after successful ND2 conversion"
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
        print(f"split_to_nd2.py version: {version}")
        return
    
    # Process files
    process_files(
        input_pattern=args.input_search_pattern,
        delete_tmp=args.delete_tmp_folder,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

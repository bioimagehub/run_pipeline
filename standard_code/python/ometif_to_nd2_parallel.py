"""
Convert OME-TIFF files to ND2 format using NIS-Elements (PARALLEL VERSION).
Works after bfconvert_wrapper.py to convert individual OME-TIFF files to ND2.

This parallel version uses the -q (new instance) flag to launch multiple
NIS-Elements instances simultaneously, significantly speeding up batch processing
if your license and system resources support it.

Workflow:
1. Takes a search pattern for OME-TIFF files
2. Opens multiple files in separate NIS-Elements instances (using -q flag)
3. Saves as ND2 (same basename, different extension)
4. Optionally deletes source OME-TIFF files

MIT License - BIPHUB, University of Oslo
"""

import os
import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, List
from glob import glob
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logger = logging.getLogger(__name__)

# Thread-safe counter for tracking progress
class ProgressCounter:
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.lock = threading.Lock()
    
    def increment_success(self):
        with self.lock:
            self.completed += 1
            self.successful += 1
            logger.info(f"Progress: {self.completed}/{self.total} files processed ({self.successful} successful, {self.failed} failed)")
    
    def increment_failure(self):
        with self.lock:
            self.completed += 1
            self.failed += 1
            logger.info(f"Progress: {self.completed}/{self.total} files processed ({self.successful} successful, {self.failed} failed)")


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
    timeout: int = 7200,
    worker_id: int = 0
) -> bool:
    """
    Open OME-TIFF in NIS-Elements and save as ND2.
    Uses -q flag to allow multiple NIS-Elements instances.
    
    Args:
        ometif_file: Path to OME-TIFF file
        output_nd2_path: Where to save the ND2 file
        nis_exe_path: Path to NIS-Elements executable
        timeout: Maximum time to wait for conversion (seconds)
        worker_id: Worker thread ID for unique marker files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create done marker file path in same directory as output
        output_dir = os.path.dirname(output_nd2_path)
        # Use worker_id and timestamp to ensure unique marker files
        done_marker_path = os.path.join(output_dir, f".nis_done_w{worker_id}_{int(time.time()*1000)}.tmp")
        
        # Remove old done marker if it exists
        if os.path.exists(done_marker_path):
            os.remove(done_marker_path)
        
        # Create temporary macro file (unique per worker)
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'_w{worker_id}.mac', delete=False) as f:
            macro_path = f.name
            macro_content = create_nis_save_macro(output_nd2_path, done_marker_path)
            f.write(macro_content)
        
        logger.debug(f"[Worker {worker_id}] Created macro file: {macro_path}")
        logger.info(f"[Worker {worker_id}] Opening: {ometif_file}")
        logger.info(f"[Worker {worker_id}] Saving as: {output_nd2_path}")
        
        # Open NIS-Elements with the file and macro
        # -q allows new instance, -f opens file, -mv runs macro after opening
        cmd = [nis_exe_path, "-q", "-f", ometif_file, "-mv", macro_path]
        
        logger.debug(f"[Worker {worker_id}] Launching NIS-Elements: {' '.join(cmd)}")
        
        # Launch NIS-Elements in background (don't wait)
        subprocess.Popen(cmd)
        
        # Poll for done marker file
        poll_interval = 2
        elapsed = 0
        
        logger.info(f"[Worker {worker_id}] Waiting for NIS-Elements to complete conversion (timeout: {timeout}s)...")
        while elapsed < timeout:
            time.sleep(poll_interval)
            elapsed += poll_interval
            
            if os.path.exists(done_marker_path):
                logger.info(f"[Worker {worker_id}] ✓ Conversion completed (took {elapsed}s)")
                break
            
            if elapsed % 30 == 0:  # Log every 30 seconds (less frequent for parallel)
                logger.debug(f"[Worker {worker_id}] Still waiting... ({elapsed}s elapsed)")
        else:
            # Timeout reached
            logger.error(f"[Worker {worker_id}] ✗ Timeout waiting for conversion ({timeout}s)")
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
            logger.debug(f"[Worker {worker_id}] Cleaned up macro file: {macro_path}")
        except Exception as e:
            logger.warning(f"[Worker {worker_id}] Could not delete macro file {macro_path}: {e}")
        
        # Check if ND2 file was created
        if os.path.exists(output_nd2_path):
            file_size = os.path.getsize(output_nd2_path) / (1024*1024)  # MB
            logger.info(f"[Worker {worker_id}] ✓ Successfully created ND2: {output_nd2_path} ({file_size:.2f} MB)")
            return True
        else:
            logger.error(f"[Worker {worker_id}] ✗ ND2 file was not created: {output_nd2_path}")
            return False
    
    except Exception as e:
        logger.error(f"[Worker {worker_id}] Error during NIS-Elements conversion: {e}")
        return False


def process_file(
    ometif_file: str,
    nis_exe_path: str,
    output_folder: str = None,
    delete_source: bool = False,
    timeout: int = 7200,
    worker_id: int = 0
) -> Tuple[str, bool]:
    """
    Process a single OME-TIFF file: convert to ND2, optionally delete source.
    
    Args:
        ometif_file: Path to OME-TIFF file
        nis_exe_path: Path to NIS-Elements executable
        output_folder: Optional output folder (default: same as source)
        delete_source: Whether to delete the source OME-TIFF file after conversion
        timeout: Maximum time to wait for conversion (seconds)
        worker_id: Worker thread ID for logging
        
    Returns:
        Tuple of (filename, success_bool)
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
        success = convert_to_nd2(ometif_file, output_nd2_path, nis_exe_path, timeout=timeout, worker_id=worker_id)
        
        if success and delete_source:
            try:
                logger.info(f"[Worker {worker_id}] Deleting source file: {ometif_file}")
                os.remove(ometif_file)
                logger.info(f"[Worker {worker_id}] ✓ Deleted source file")
            except Exception as e:
                logger.warning(f"[Worker {worker_id}] Could not delete source file {ometif_file}: {e}")
        
        return (ometif_file, success)
    
    except Exception as e:
        logger.error(f"[Worker {worker_id}] Error processing file {ometif_file}: {e}")
        return (ometif_file, False)


def process_files_parallel(
    input_pattern: str,
    output_folder: str = None,
    delete_source: bool = False,
    dry_run: bool = False,
    timeout: int = 7200,
    max_workers: int = 4
) -> None:
    """
    Process multiple OME-TIFF files in parallel using multiple NIS-Elements instances.
    
    Args:
        input_pattern: Glob pattern for input OME-TIFF files
        output_folder: Optional output folder (default: same as source files)
        delete_source: Whether to delete source files after conversion
        dry_run: Print planned actions without executing
        timeout: Maximum time to wait for each conversion (seconds)
        max_workers: Maximum number of parallel NIS-Elements instances
    """
    # Expand glob pattern
    input_files = sorted(glob(input_pattern, recursive=True))
    
    if not input_files:
        logger.warning(f"No files found matching pattern: {input_pattern}")
        return
    
    logger.info(f"Found {len(input_files)} file(s) to process")
    logger.info(f"Processing with {max_workers} parallel workers")
    
    # Find NIS-Elements
    if not dry_run:
        try:
            nis_exe_path = find_nis_executable()
        except FileNotFoundError as e:
            logger.error(str(e))
            return
    else:
        nis_exe_path = "NIS-Elements.exe"
    
    # Initialize progress counter
    progress = ProgressCounter(len(input_files))
    
    if dry_run:
        # Sequential dry run
        for ometif_file in input_files:
            logger.info(f"\n{'='*60}")
            logger.info(f"[DRY RUN] Would process: {ometif_file}")
            
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
            progress.increment_success()
    else:
        # Parallel processing using ThreadPoolExecutor
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting parallel conversion with {max_workers} workers")
        logger.info(f"{'='*60}\n")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {}
            for idx, ometif_file in enumerate(input_files):
                worker_id = idx % max_workers
                future = executor.submit(
                    process_file,
                    ometif_file,
                    nis_exe_path,
                    output_folder=output_folder,
                    delete_source=delete_source,
                    timeout=timeout,
                    worker_id=worker_id
                )
                future_to_file[future] = ometif_file
            
            # Process results as they complete
            for future in as_completed(future_to_file):
                filename, success = future.result()
                if success:
                    progress.increment_success()
                else:
                    progress.increment_failure()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"=== Processing Complete ===")
    logger.info(f"Total files: {progress.total}")
    logger.info(f"Successful: {progress.successful}")
    logger.info(f"Failed: {progress.failed}")
    logger.info(f"{'='*60}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert OME-TIFF files to ND2 format using NIS-Elements (PARALLEL VERSION). "
                    "Uses the -q flag to launch multiple NIS-Elements instances simultaneously. "
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

- name: Convert OME-TIFF to ND2 (PARALLEL - 4 workers)
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/ometif_to_nd2_parallel.py'
  - --input-pattern: '%YAML%/output/**/*.ome.tif'
  - --max-workers: 4
  - --delete-source
  # Converts all OME-TIFFs to ND2 using 4 parallel instances

- name: Convert with custom output folder (8 parallel workers)
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/ometif_to_nd2_parallel.py'
  - --input-pattern: '%YAML%/ometif_files/**/*.ome.tif'
  - --output-folder: '%YAML%/nd2_files'
  - --max-workers: 8
  - --timeout: 1200
  # Save ND2 files to different folder with 8 workers

- name: Conservative parallel (2 workers)
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/ometif_to_nd2_parallel.py'
  - --input-pattern: '%YAML%/output/**/*.ome.tif'
  - --max-workers: 2
  # Use only 2 parallel instances if license or resources are limited

- name: Dry run (preview parallel conversion)
  environment: uv@3.11:convert-to-tif
  commands:
  - python
  - '%REPO%/standard_code/python/ometif_to_nd2_parallel.py'
  - --input-pattern: '%YAML%/output/**/*.ome.tif'
  - --dry-run
  # Preview what would be converted without executing

Example workflow (LIF -> OME-TIFF -> ND2):
  1. bfconvert_wrapper.py: Splits multi-series LIF into individual OME-TIFFs
  2. ometif_to_nd2_parallel.py: Converts OME-TIFFs to ND2 in parallel

Performance notes:
  - Start with --max-workers=2 to test if your license supports multiple instances
  - Increase workers based on CPU cores and RAM (recommend 1-2 per CPU core)
  - Monitor system resources during processing
  - Each NIS-Elements instance may use significant RAM
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
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel NIS-Elements instances (default: 4). "
             "Start with 2 to test license compatibility, increase based on CPU/RAM."
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
    
    # Setup logging with thread names for parallel debugging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
    )
    
    if args.version:
        version_file = Path(__file__).parent.parent.parent / "VERSION"
        try:
            version = version_file.read_text(encoding='utf-8').strip()
        except Exception:
            version = "unknown"
        print(f"ometif_to_nd2_parallel.py version: {version}")
        return
    
    # Validate max_workers
    if args.max_workers < 1:
        logger.error("--max-workers must be at least 1")
        return
    
    if args.max_workers > 16:
        logger.warning(f"--max-workers={args.max_workers} is very high. This may consume excessive resources.")
        logger.warning("Consider starting with 2-4 workers and increasing if system can handle it.")
    
    # Process files in parallel
    process_files_parallel(
        input_pattern=args.input_pattern,
        output_folder=args.output_folder,
        delete_source=args.delete_source,
        dry_run=args.dry_run,
        timeout=args.timeout,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()

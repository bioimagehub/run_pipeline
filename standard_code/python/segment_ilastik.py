import subprocess
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
import logging
import tempfile
import time
import uuid
import multiprocessing

import bioimage_pipeline_utils as rp

# Add tqdm for progress bars (fallback to no-op if not installed)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False
    def tqdm(iterable=None, **kwargs):
        if iterable is not None:
            return iterable
        # Return a simple progress bar mock for context manager usage
        class SimplePbar:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def update(self, n=1):
                pass
            def close(self):
                pass
        return SimplePbar()

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')

def get_expected_output_path(input_file):
    """Get the expected H5 output path for an input file."""
    in_dir = os.path.dirname(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    h5_path = os.path.join(in_dir, base_name + "_Probabilities.h5")
    return h5_path

def monitor_file_completion(input_files, pbar_position, batch_id, stop_event, check_interval=1):
    """
    Monitor for completed output files and update progress bar in real-time.
    Runs in a separate thread to track files as they're created.
    
    Monitors H5 files as they appear during Ilastik execution.
    """
    completed = set()
    
    # Create a progress bar for this specific batch
    pbar = tqdm(total=len(input_files), 
                desc=f"Batch {batch_id}", 
                unit="file",
                position=pbar_position,
                leave=True)
    
    while not stop_event.is_set():
        newly_completed = []
        
        for input_file in input_files:
            if input_file in completed:
                continue
                
            h5_path = get_expected_output_path(input_file)
            
            # Check if H5 file exists (appears during Ilastik execution)
            # This gives real-time progress as Ilastik processes files
            if os.path.exists(h5_path):
                # Try to validate it's complete by opening it
                try:
                    with h5py.File(h5_path, 'r') as f:
                        if len(f.keys()) > 0:
                            # H5 file exists and has data, mark as complete
                            completed.add(input_file)
                            newly_completed.append(input_file)
                except:
                    # File exists but can't be opened yet (still being written)
                    pass
        
        # Update progress bar for all newly completed files
        if newly_completed:
            pbar.update(len(newly_completed))
        
        time.sleep(check_interval)
    
    pbar.close()
    return len(completed)

def process_batch(args, input_files, batch_id=0, pbar_position=0):
    """Process a batch of files with a single Ilastik process."""
    if not input_files:
        return 0
    
    # Start file monitoring thread with its own progress bar
    stop_event = None
    monitor_thread = None
    from threading import Event, Thread
    stop_event = Event()
    monitor_thread = Thread(target=monitor_file_completion, 
                           args=(input_files, pbar_position, batch_id, stop_event))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Check which files already have H5 output and validate them
    files_needing_processing = []
    files_with_h5 = []
    
    for input_file in input_files:
        h5_output_path = get_expected_output_path(input_file)
        if os.path.exists(h5_output_path):
            # Validate that the H5 file can be opened and read
            try:
                with h5py.File(h5_output_path, 'r') as f:
                    # Check that the file has data
                    if len(f.keys()) == 0:
                        files_needing_processing.append(input_file)
                    else:
                        # Try to read the data shape to ensure it's valid
                        group_key = list(f.keys())[0]
                        dataset = f[group_key]
                        if hasattr(dataset, 'shape'):
                            # Valid dataset, file is good
                            files_with_h5.append(input_file)
                        else:
                            files_needing_processing.append(input_file)
            except Exception as e:
                logging.warning(f"Cannot read {os.path.basename(h5_output_path)}: {e}")
                files_needing_processing.append(input_file)
        else:
            files_needing_processing.append(input_file)
    
    if files_with_h5:
        print(f"\n[Batch {batch_id}] Found {len(files_with_h5)}/{len(input_files)} files with valid H5 outputs")
    
    if files_needing_processing:
        print(f"[Batch {batch_id}] Processing {len(files_needing_processing)} files (new or corrupted)")
        
        # Delete any corrupted H5 files to ensure clean re-processing
        corrupted_count = 0
        for input_file in files_needing_processing:
            h5_output_path = get_expected_output_path(input_file)
            if os.path.exists(h5_output_path):
                try:
                    os.remove(h5_output_path)
                    corrupted_count += 1
                except Exception as e:
                    logging.warning(f"Could not remove {os.path.basename(h5_output_path)}: {e}")
        
        if corrupted_count > 0:
            print(f"[Batch {batch_id}] Removed {corrupted_count} corrupted H5 files")
    
    # Only run Ilastik if there are files that need processing
    proc = None
    if files_needing_processing:
        # Prepare a unique Ilastik log directory per process to avoid Windows file-lock races
        unique_log_dir = os.path.join(tempfile.gettempdir(), f"ilastik_logs_{uuid.uuid4().hex}")
        os.makedirs(unique_log_dir, exist_ok=True)
        env = os.environ.copy()
        env["ILASTIK_LOG_DIR"] = unique_log_dir

        # Run Ilastik headless with files that need processing
        try:
            cmd = [
                args.ilastik_path,
                '--headless',
                f'--project={args.project_path}',
            ] + files_needing_processing  # Add only files needing processing
            
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        except FileNotFoundError as e:
            logging.error(f"Failed to run Ilastik executable: {args.ilastik_path}")
            logging.error(f"Error: {e}")
            logging.error(f"Please check that the ilastik-path is correct and the file exists")
            if stop_event:
                stop_event.set()
            raise
        finally:
            # Stop monitoring thread
            if stop_event:
                stop_event.set()
            if monitor_thread:
                monitor_thread.join(timeout=5)
    else:
        # No files need processing, just stop the monitor
        if stop_event:
            stop_event.set()
        if monitor_thread:
            monitor_thread.join(timeout=5)

    # Verify H5 files were created
    files_processed = 0
    failed_files = []
    for input_file in input_files:
        h5_output_path = get_expected_output_path(input_file)

        # Give the filesystem a brief moment for the output to appear
        if not os.path.exists(h5_output_path):
            for _ in range(10):  # wait up to ~10s
                time.sleep(1)
                if os.path.exists(h5_output_path):
                    break
        
        # Check if Ilastik produced output
        if not os.path.exists(h5_output_path):
            failed_files.append(input_file)
            continue
        
        # Validate H5 file can be opened
        try:
            with h5py.File(h5_output_path, 'r') as f:
                if len(f.keys()) == 0:
                    logging.error(f"H5 file is empty: {h5_output_path}")
                    failed_files.append(input_file)
                    continue
        except Exception as e:
            logging.error(f"Failed to read H5 file: {h5_output_path}")
            logging.error(f"Error: {e}")
            logging.error(f"This file may be corrupted. Consider deleting it and re-running.")
            failed_files.append(input_file)
            continue
        
        files_processed += 1
    
    # Report any failures at the end
    if failed_files:
        print(f"\n[Batch {batch_id}] Warning: {len(failed_files)}/{len(input_files)} files failed to process")
        if proc and proc.stderr and "error" in proc.stderr.lower():
            # Only show stderr if it contains actual errors (not just Qt warnings)
            print(f"[Batch {batch_id}] Ilastik stderr output:")
            # Filter out Qt warnings
            for line in proc.stderr.split('\n'):
                if line and not line.startswith("QObject::connect:"):
                    print(f"  {line}")
    
    return files_processed


def chunk_list(lst, n):
    """Split a list into n roughly equal chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def chunk_by_cmd_length(files, max_cmd_length=7000):
    """
    Split files into chunks that don't exceed Windows command line length limit.
    
    Args:
        files: List of file paths
        max_cmd_length: Maximum command line length (Windows limit is ~8191)
    
    Returns:
        List of file chunks
    """
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Base command length (ilastik path + flags)
    base_length = 300  # Conservative estimate for ilastik.exe path + flags
    
    for file in files:
        file_length = len(file) + 3  # +3 for space and potential quotes
        
        if current_length + file_length > max_cmd_length and current_chunk:
            # Start a new chunk
            chunks.append(current_chunk)
            current_chunk = [file]
            current_length = base_length + file_length
        else:
            current_chunk.append(file)
            current_length += file_length
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks



def process_folder(args: argparse.Namespace) -> None:
    # Find files to process using glob pattern
    pattern = args.input_search_pattern
    files_to_process = rp.get_files_to_process2(pattern, False)

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    if args.no_parallel:
        # Process all files respecting command line length limits
        print(f"\nProcessing {len(files_to_process)} files in sequential batches (no parallel)")
        
        # Split by command line length
        batches = chunk_by_cmd_length(files_to_process)
        print(f"Split into {len(batches)} batches to respect Windows command line limits")
        print(f"Files per batch: {[len(b) for b in batches]}\n")
        
        for i, batch in enumerate(batches):
            print(f"Processing batch {i+1}/{len(batches)}...")
            process_batch(args, batch, batch_id=i, pbar_position=0)
        
        print(f"\nCompleted processing {len(files_to_process)} files")
    else:
        # Split files among workers, respecting command line length
        n_workers = args.workers if args.workers else multiprocessing.cpu_count()
        
        # First split by command line length to get safe batches
        safe_batches = chunk_by_cmd_length(files_to_process)
        
        print(f"\nProcessing {len(files_to_process)} files using up to {n_workers} parallel Ilastik processes")
        print(f"Split into {len(safe_batches)} batches to respect Windows command line limits")
        print(f"Files per batch: {[len(b) for b in safe_batches]}")
        print("Each worker will show its own progress bar...\n")
        
        # Process batches with max n_workers in parallel
        files_completed = 0
        batch_id = 0
        
        # Process in waves to maintain max n_workers concurrent processes
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            batch_queue = list(safe_batches)
            
            # Submit initial batches (up to n_workers)
            for i in range(min(n_workers, len(batch_queue))):
                batch = batch_queue.pop(0)
                future = executor.submit(process_batch, args, batch, batch_id, batch_id % n_workers)
                futures[future] = (batch_id, len(batch))
                batch_id += 1
            
            # As batches complete, submit new ones
            while futures:
                done_future = next(as_completed(futures))
                done_batch_idx, batch_size = futures.pop(done_future)
                
                try:
                    n_processed = done_future.result()
                    files_completed += n_processed
                except Exception as exc:
                    print(f'\n  Batch {done_batch_idx+1} generated an exception: {exc}')
                
                # Submit next batch if any remain
                if batch_queue:
                    next_batch = batch_queue.pop(0)
                    future = executor.submit(process_batch, args, next_batch, batch_id, batch_id % n_workers)
                    futures[future] = (batch_id, len(next_batch))
                    batch_id += 1
        
        print(f"\nCompleted: {files_completed}/{len(files_to_process)} files processed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Ilastik segmentation in parallel to generate probability H5 files.",
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Run Ilastik segmentation
  environment: uv@3.11:segment-ilastik
  commands:
  - python
  - '%REPO%/standard_code/python/segment_ilastik.py'
  - --ilastik-path: 'C:/Program Files/ilastik-1.4.0/ilastik.exe'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data'
  - --project-path: '%YAML%/my_project.ilp'
  - --workers: 4

Note: Output H5 files are saved next to input files with '_Probabilities.h5' suffix.
      Post-processing (thresholding, TIF conversion) should be done in a separate step.
"""
    )
    parser.add_argument("--ilastik-path", type=str, required=True, help="Path to the ilastik executable")
    parser.add_argument("--input-search-pattern", type=str, required=True, help="Glob pattern for input images, e.g. './input_Ilastik/*.h5'")
    parser.add_argument("--output-folder", type=str, required=True, help="Path to save the output (currently unused, H5 files saved next to inputs)")
    parser.add_argument("--project-path", type=str, required=True, help="Path to trained Ilastik project file (.ilp)")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing (process all files in one batch)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")





    args = parser.parse_args()
    if not os.path.exists(args.project_path):
        print(f"Please start Ilastik and save the project file to: {args.project_path}")
    while not os.path.exists(args.project_path):
        time.sleep(10)
        continue

        
    # Process the folder
    process_folder(args)

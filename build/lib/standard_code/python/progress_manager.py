"""
Unified progress tracking and folder processing for bioimage pipeline.

This module provides a standardized approach to processing multiple files in parallel
with independent progress bars for each file. It separates the boilerplate of file
discovery, parallel execution, and progress tracking from the core processing logic.

Author: BIPHUB - Bioimage Informatics Hub, University of Oslo
"""

import os
import threading
from typing import List, Dict, Any, Callable, Optional, Protocol
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from tqdm import tqdm
import logging

import bioimage_pipeline_utils as rp

logger = logging.getLogger(__name__)


class FileProcessor(Protocol):
    """
    Protocol that file processing functions must implement.
    
    The file processor should accept progress_callback as a kwarg even if not used.
    """
    
    def __call__(self, 
                 *args,
                 progress_callback: Optional[Callable[[int], None]] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Process a single file.
        
        Args:
            *args: Positional arguments (typically input_path, output_path)
            progress_callback: Function to call with progress updates (timepoints processed)
            **kwargs: Additional processing parameters
            
        Returns:
            Dict with processing results including 'success' boolean and optional 'error' string
        """
        ...


class MultiFileProgressTracker:
    """
    Manages multiple progress bars for parallel file processing.
    Each file gets its own progress bar that updates independently.
    
    This class is thread-safe and can be used with concurrent.futures.
    """
    
    def __init__(self):
        self.progress_bars: Dict[str, tqdm] = {}
        self.positions: Dict[str, int] = {}
        self.next_position = 0
        self.lock = threading.Lock()
    
    def create_progress_bar(self, file_id: str, total_timepoints: int, description: str) -> None:
        """
        Create a new progress bar for a file.
        
        Args:
            file_id: Unique identifier for the file
            total_timepoints: Total number of timepoints/iterations expected
            description: Description text for the progress bar
        """
        with self.lock:
            self.positions[file_id] = self.next_position
            self.next_position += 1
            
            self.progress_bars[file_id] = tqdm(
                total=total_timepoints,
                desc=description,
                position=self.positions[file_id],
                leave=False,
                dynamic_ncols=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
    
    def update_progress(self, file_id: str, n: int = 1) -> None:
        """
        Update progress for a specific file.
        
        Args:
            file_id: Unique identifier for the file
            n: Number of steps to increment (default: 1)
        """
        with self.lock:
            if file_id in self.progress_bars:
                self.progress_bars[file_id].update(n)
    
    def reset_progress_bar(self, file_id: str, new_description: str, new_total: Optional[int] = None) -> None:
        """
        Reset progress bar for phase transitions (e.g., detect → apply).
        
        Args:
            file_id: Unique identifier for the file
            new_description: New description text
            new_total: New total if different from current (optional)
        """
        with self.lock:
            if file_id in self.progress_bars:
                pbar = self.progress_bars[file_id]
                pbar.set_description(new_description)
                pbar.n = 0
                if new_total is not None:
                    pbar.total = new_total
                pbar.refresh()
    
    def close_progress_bar(self, file_id: str, final_message: Optional[str] = None) -> None:
        """
        Close and clean up a progress bar.
        
        Args:
            file_id: Unique identifier for the file
            final_message: Optional final message to display before closing
        """
        with self.lock:
            if file_id in self.progress_bars:
                pbar = self.progress_bars[file_id]
                if final_message:
                    pbar.set_description(final_message)
                    pbar.refresh()
                pbar.close()
                del self.progress_bars[file_id]
    
    def close_all(self) -> None:
        """Close all progress bars."""
        with self.lock:
            for pbar in self.progress_bars.values():
                pbar.close()
            self.progress_bars.clear()


def process_folder_unified(
    input_files: List[str],
    output_folder: str,
    base_folder: str,
    file_processor: FileProcessor,
    collapse_delimiter: str = "__",
    output_extension: str = "",
    parallel: bool = True,
    n_jobs: Optional[int] = None,
    use_processes: bool = False,
    estimate_timepoints_func: Optional[Callable[[str], int]] = None,
    create_output_path_func: Optional[Callable[[str, str, str, str], str]] = None,
    **processor_kwargs
) -> List[Dict[str, Any]]:
    """
    Unified folder processing function with standardized progress tracking.
    
    This function handles:
    - Output path generation
    - Parallel or sequential processing
    - Progress tracking with independent bars per file
    - Error handling and result collection
    
    Args:
        input_files: List of input file paths to process
        output_folder: Output directory
        base_folder: Base folder for relative path calculation
        file_processor: Function that processes individual files
        collapse_delimiter: Delimiter for flattening nested paths
        output_extension: Extension to add to output filenames (e.g., "_corrected")
        parallel: Whether to use parallel processing (default: True)
        n_jobs: Number of parallel jobs (None for auto)
        use_processes: Use ProcessPoolExecutor instead of ThreadPoolExecutor
        estimate_timepoints_func: Function to estimate timepoints for progress bar
        create_output_path_func: Custom function to create output paths (optional)
        **processor_kwargs: Additional arguments passed to file_processor
        
    Returns:
        List of processing results with 'success', 'input_path', 'output_path', etc.
    """
    
    if not input_files:
        logger.warning("No files to process")
        return []
    
    # Setup output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Create progress tracker
    tracker = MultiFileProgressTracker()
    results = []
    
    def default_create_output_path(input_path: str, base_folder: str, output_folder: str, collapse_delimiter: str) -> str:
        """Default output path generation logic."""
        collapsed = rp.collapse_filename(input_path, base_folder, collapse_delimiter)
        base_name = os.path.splitext(collapsed)[0]
        output_filename = base_name + output_extension + os.path.splitext(input_path)[1]
        return os.path.join(output_folder, output_filename)
    
    # Use custom or default output path function
    create_output_path = create_output_path_func or default_create_output_path
    
    def process_single_file_wrapper(input_path: str) -> Dict[str, Any]:
        """Wrapper that handles progress tracking for a single file."""
        file_id = Path(input_path).stem
        output_path = create_output_path(input_path, base_folder, output_folder, collapse_delimiter)
        
        # Estimate timepoints for progress bar
        estimated_timepoints = 1  # Default fallback
        if estimate_timepoints_func:
            try:
                estimated_timepoints = estimate_timepoints_func(input_path)
            except Exception as e:
                logger.warning(f"Failed to estimate timepoints for {input_path}: {e}")
        
        # Create progress bar
        tracker.create_progress_bar(
            file_id,
            estimated_timepoints,
            f"🔄 Processing: {file_id}"
        )
        
        def progress_callback(n: int = 1) -> None:
            """Callback function for the file processor to report progress."""
            tracker.update_progress(file_id, n)
        
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Process the file - pass progress_callback as a kwarg
            result = file_processor(
                input_path,
                output_path,
                progress_callback=progress_callback,
                **processor_kwargs
            )
            
            # Handle different return types
            if result is None:
                # Function returned None, assume success
                result = {'success': True}
            elif not isinstance(result, dict):
                # Function returned non-dict, wrap it
                result = {'success': True, 'result': result}
            
            # Close progress bar with success message
            success = result.get('success', True)
            if success:
                final_message = f"✅ Completed: {file_id}"
            else:
                final_message = f"❌ Failed: {file_id}"
            
            tracker.close_progress_bar(file_id, final_message)
            
            return {
                'input_path': input_path,
                'output_path': output_path,
                **result
            }
            
        except Exception as e:
            error_msg = f"Exception processing {input_path}: {e}"
            logger.error(error_msg)
            tracker.close_progress_bar(file_id, f"❌ Failed: {file_id}")
            return {
                'input_path': input_path,
                'output_path': output_path,
                'success': False,
                'error': str(e)
            }
    
    try:
        if not parallel or len(input_files) == 1:
            # Sequential processing
            logger.info(f"Processing {len(input_files)} files sequentially")
            for input_path in input_files:
                result = process_single_file_wrapper(input_path)
                results.append(result)
        else:
            # Parallel processing
            executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
            max_workers = n_jobs if n_jobs is not None else None
            
            logger.info(f"Processing {len(input_files)} files in parallel with {max_workers or 'auto'} workers")
            
            with executor_class(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(process_single_file_wrapper, input_path): input_path
                    for input_path in input_files
                }
                
                for future in as_completed(future_to_file):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        input_path = future_to_file[future]
                        logger.error(f"Exception in future for {input_path}: {e}")
                        results.append({
                            'input_path': input_path,
                            'success': False,
                            'error': str(e)
                        })
    
    finally:
        # Cleanup progress bars
        import time
        time.sleep(0.2)  # Small delay to ensure final messages are visible
        tracker.close_all()
    
    # Print summary
    successful = sum(1 for r in results if r.get('success', True))
    failed = len(results) - successful
    logger.info(f"Processing complete: {successful} successful, {failed} failed")
    
    return results

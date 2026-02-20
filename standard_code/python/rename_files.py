"""
Rename files using pattern matching with wildcard substitution.
Supports moving files to a new folder while renaming.

MIT License - BIPHUB, University of Oslo
"""

import os
import argparse
import logging
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import bioimage_pipeline_utils as rp

# Module-level logger
logger = logging.getLogger(__name__)


def rename_single_file(
    input_path: str,
    output_path: str,
    copy_only: bool = False,
    dry_run: bool = False
) -> bool:
    """
    Rename and optionally copy/move a single file.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file (with new name)
        copy_only: If True, copy file to output path (keep original). If False, move file (rename/delete original).
        dry_run: If True, only print planned actions
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if dry_run:
            action = "Copy" if copy_only else "Rename/Move"
            logger.info(f"[DRY RUN] Would {action}: {input_path}")
            logger.info(f"[DRY RUN] To: {output_path}")
            return True
        
        # Ensure output folder exists
        output_folder = os.path.dirname(output_path)
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
        
        # Copy or move file
        if copy_only:
            # Copy to output path, keep original
            shutil.copy2(input_path, output_path)
            logger.info(f"Copied: {input_path} -> {output_path}")
        else:
            # Move/rename file (delete original)
            if os.path.dirname(input_path) != os.path.dirname(output_path):
                # Moving to different folder
                shutil.copy2(input_path, output_path)
                os.remove(input_path)
                logger.info(f"Moved: {input_path} -> {output_path}")
            else:
                # Renaming in same folder
                os.rename(input_path, output_path)
                logger.info(f"Renamed: {input_path} -> {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        return False


def process_files(
    input_pattern: str,
    replace_term: str,
    output_folder: str = None,
    no_parallel: bool = False,
    dry_run: bool = False
) -> None:
    """
    Process multiple files matching a pattern, renaming them.
    
    The input pattern should contain a '*' wildcard. The portion AFTER the '*'
    (the suffix) is replaced with replace_term, while the wildcard portion is
    preserved to keep filenames unique.
    
    Examples:
        input_pattern = "data/*.nd2" -> "<original>" + replace_term + ".nd2"
        input_pattern = "data/prefix_*.tif" -> "prefix_<original>" + replace_term + ".tif"
        input_pattern = "data/**/*.czi" -> recursive search with the same rename rule
    
    Args:
        input_pattern: File search pattern with * as wildcard
        replace_term: Replacement term for everything after the *
        output_folder: Optional output folder to move renamed files to
        no_parallel: Disable parallel processing
        dry_run: Only print planned actions without executing
    """
    # Validate pattern contains wildcard
    if '*' not in input_pattern:
        logger.error("Input pattern must contain a '*' wildcard")
        return
    
    # Find files
    search_subfolders = '**' in input_pattern
    files = rp.get_files_to_process2(input_pattern, search_subfolders=search_subfolders)
    
    if not files:
        logger.error(f"No files found matching pattern: {input_pattern}")
        return
    
    logger.info(f"Found {len(files)} file(s) to process")
    
    # Split pattern into prefix and suffix
    # Example: "data/prefix_*.txt" -> prefix="data/prefix_", suffix=".txt"
    pattern_parts = input_pattern.split('*', 1)
    prefix = pattern_parts[0]
    suffix = pattern_parts[1] if len(pattern_parts) > 1 else ""
    prefix_basename = os.path.basename(prefix)
    suffix_ext = os.path.splitext(suffix)[1]
    replace_ext = os.path.splitext(replace_term)[1]
    
    logger.info(f"Pattern prefix: '{prefix}'")
    logger.info(f"Pattern suffix: '{suffix}'")
    logger.info(f"Replace term: '{replace_term}'")
    
    # Prepare file pairs
    file_pairs = []
    for src in files:
        # Extract the middle part (between prefix and suffix)
        src_normalized = src.replace('\\', '/')
        prefix_normalized = prefix.replace('\\', '/')
        
        # Check if file matches the pattern
        if src_normalized.startswith(prefix_normalized):
            middle = src_normalized[len(prefix_normalized):]
            
            # Remove suffix if present
            if suffix and middle.endswith(suffix):
                middle = middle[:-len(suffix)]
            
            # Build new filename: prefix_basename + wildcard_part + replace_term (+ ext if needed)
            replacement = replace_term
            if suffix_ext and not replace_ext:
                replacement = replace_term + suffix_ext
            new_filename = prefix_basename + middle + replacement
            
            # Determine output path
            if output_folder:
                # Move to output folder
                output_path = os.path.join(output_folder, os.path.basename(new_filename))
            else:
                # Rename in same folder
                output_path = os.path.join(os.path.dirname(src), os.path.basename(new_filename))
            
            file_pairs.append((src, output_path))
    
    if not file_pairs:
        logger.error("No files matched the pattern")
        return
    
    logger.info(f"Will rename {len(file_pairs)} file(s)")
    for src, dst in file_pairs:
        logger.info(f"  {src} -> {dst}")
    
    # Dry run - just print plans
    if dry_run:
        print(f"[DRY RUN] Would process {len(file_pairs)} files")
        if output_folder:
            print(f"[DRY RUN] Output folder: {output_folder}")
        return
    
    # Process files
    if no_parallel or len(file_pairs) == 1:
        # Sequential processing
        for src, dst in file_pairs:
            rename_single_file(src, dst, copy_only=bool(output_folder), dry_run=False)
    else:
        # Parallel processing
        max_workers = min(os.cpu_count() or 4, len(file_pairs))
        logger.info(f"Processing with {max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(rename_single_file, src, dst, bool(output_folder), False): (src, dst)
                for src, dst in file_pairs
            }
            
            for future in as_completed(futures):
                src, dst = futures[future]
                try:
                    success = future.result()
                    if not success:
                        logger.error(f"Failed: {src}")
                except Exception as e:
                    logger.error(f"Exception processing {src}: {e}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Rename files using pattern matching with wildcard substitution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (YAML config for run_pipeline.exe):
---
run:
- name: Rename files with pattern matching
  environment: uv@3.11:rename-files
  commands:
  - python
  - '%REPO%/standard_code/python/rename_files.py'
    - --input-search-pattern: 'data/prefix_*.tif'
    - --replace-term: '_newname'
  
- name: Rename and move files to output folder
  environment: uv@3.11:rename-files
  commands:
  - python
  - '%REPO%/standard_code/python/rename_files.py'
  - --input-search-pattern: 'data/**/*.nd2'
  - --replace-term: 'processed'
  - --output-folder: '%YAML%/output_data'
  
- name: Dry run to preview changes
  environment: uv@3.11:rename-files
  commands:
  - python
  - '%REPO%/standard_code/python/rename_files.py'
  - --input-search-pattern: 'data/*.czi'
  - --replace-term: 'renamed'
  - --dry-run
        """
    )
    
    parser.add_argument(
        "--input-search-pattern",
        type=str,
        required=True,
        help="Input file pattern with '*' wildcard (e.g., 'data/*.nd2' or 'data/prefix_*.tif')"
    )
    
    parser.add_argument(
        "--replace-term",
        type=str,
        required=True,
        help="Replacement term for the suffix (the part after '*') in the pattern"
    )
    
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Optional output folder to move renamed files to (default: rename in place)"
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
        print(f"rename_files.py version: {version}")
        return
    
    # Process files
    process_files(
        input_pattern=args.input_search_pattern,
        replace_term=args.replace_term,
        output_folder=args.output_folder,
        no_parallel=args.no_parallel,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()

import argparse
import logging
import os
import shutil
from pathlib import Path
import sys
from typing import List, Optional

import bioimage_pipeline_utils as rp


logger = logging.getLogger(__name__)


def _determine_base_folder(input_pattern: str, files: List[str]) -> str:
    """Determine base folder used for filename collapsing."""
    if "**" in input_pattern:
        base_folder = input_pattern.split("**")[0].rstrip("/\\")
        if not base_folder:
            base_folder = os.getcwd()
        return os.path.abspath(base_folder)
    return str(Path(files[0]).parent)


def _normalize_failed_basename(name: str) -> str:
    """Normalize failed basenames by stripping the '_failed...' suffix when present."""
    marker = "_failed"
    idx = name.find(marker)
    if idx == -1:
        return name
    return name[:idx]


def _ensure_unique_path(path: str) -> str:
    """Return a unique path by appending _N if the file already exists."""
    if not os.path.exists(path):
        return path

    base_name, ext = os.path.splitext(path)
    counter = 1
    candidate = f"{base_name}_{counter}{ext}"
    while os.path.exists(candidate):
        counter += 1
        candidate = f"{base_name}_{counter}{ext}"
    return candidate


def _preview(values: List[str], max_items: int = 10) -> str:
    """Return a compact preview string for logging."""
    if not values:
        return "[]"
    shown = values[:max_items]
    suffix = "" if len(values) <= max_items else f", ... (+{len(values) - max_items} more)"
    return "[" + ", ".join(shown) + suffix + "]"


def process_files(
    input_pattern: str,
    output_folder: Optional[str] = None,
    collapse_delimiter: str = "__",
    optional_input_search_pattern: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Copy files matching a search pattern into a single output folder."""

    search_subfolders = "**" in input_pattern
    files = rp.get_files_to_process2(input_pattern, search_subfolders=search_subfolders)

    if not files:
        logger.error("No files found matching pattern: %s", input_pattern)
        return

    base_folder = _determine_base_folder(input_pattern, files)

    # Default output folder
    if output_folder is None:
        output_folder = base_folder + "_copied"

    os.makedirs(output_folder, exist_ok=True)
    logger.info("Copying %d file(s) to %s", len(files), output_folder)

    # Prepare source/destination pairs
    file_pairs = []
    for src in files:
        collapsed = rp.collapse_filename(src, base_folder, collapse_delimiter)
        dst = os.path.join(output_folder, collapsed)
        dst = _ensure_unique_path(dst)
        file_pairs.append((src, dst))

    if optional_input_search_pattern:
        if search_subfolders:
            logger.error(
                "The use of '**' in --input-search-pattern is not allowed when --optional-input-search-pattern is provided. Please remove '**' from the input pattern."
            )
            return

        logger.info(
            "Optional matching enabled | main pattern: %s | optional pattern: %s",
            input_pattern,
            optional_input_search_pattern,
        )

        patterns = {
            "main": input_pattern,
            "optional": optional_input_search_pattern,
        }
        
        groups = rp.get_grouped_files_to_process(
            patterns,
            search_subfolders=search_subfolders,
        )
        
        logger.info("Total groups found: %d", len(groups))
        logger.debug("Group keys preview: %s", _preview(sorted(groups.keys())))

        # Filter to only groups with both main and optional files
        valid_groups = {
            basename: group
            for basename, group in groups.items()
            if "main" in group and "optional" in group
        }




        logger.info("Valid groups (with both main and optional): %d", len(valid_groups))
        if len(valid_groups) < len(groups):
            incomplete_groups = len(groups) - len(valid_groups)
            logger.debug("Skipped %d incomplete groups", incomplete_groups)

        optional_base_folder = _determine_base_folder(
            optional_input_search_pattern,
            [group["optional"] for group in valid_groups.values()],
        ) if valid_groups else base_folder

        already_selected = {os.path.abspath(src) for src, _ in file_pairs}
        added_optional = 0
        matched_with_optional = 0

        for basename, group in valid_groups.items():

            matched_with_optional += 1
            optional_file = group["optional"]
            src_abs = os.path.abspath(optional_file)

            if src_abs in already_selected:
                logger.debug("Skipping already selected optional file: %s", optional_file)
                continue

            collapsed = rp.collapse_filename(optional_file, optional_base_folder, collapse_delimiter)
            dst = os.path.join(output_folder, collapsed)
            dst = _ensure_unique_path(dst)
            file_pairs.append((optional_file, dst))
            already_selected.add(src_abs)
            added_optional += 1
            logger.debug("Matched optional file for '%s': %s", basename, optional_file)

        unmatched_main = len(files) - matched_with_optional
        if unmatched_main > 0:
            logger.warning(
                "Optional matching incomplete: matched %d/%d main files with optional files; %d main files unmatched",
                matched_with_optional,
                len(files),
                unmatched_main,
            )
        else:
            logger.info("Optional matching complete: matched all main files (%d)", len(files))

        if not valid_groups:
            logger.warning(
                "No valid groups found (groups with both main and optional files). Check patterns: main=%s, optional=%s",
                input_pattern,
                optional_input_search_pattern,
            )

        logger.info(
            "Optional matching summary: added %d file(s) from pattern %s",
            added_optional,
            optional_input_search_pattern,
        )

    if dry_run:
        print(f"[DRY RUN] Would copy {len(file_pairs)} files")
        print(f"[DRY RUN] Output folder: {output_folder}")
        for src, dst in file_pairs:
            print(f"[DRY RUN] {src} -> {dst}")
        return

    # Copy files
    for src, dst in file_pairs:
        shutil.copy2(src, dst)
        logger.debug("Copied %s -> %s", src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy files matching a search pattern into a single folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Copy masks into one folder
  environment: uv@3.11:copy-files
  commands:
  - python
  - '%REPO%/standard_code/python/copy_files_with_extension.py'
  - --input-search-pattern: '%YAML%/masks/**/*.tif'
  - --output-folder: '%YAML%/merged_masks'
  - --collapse-delimiter: '__'

- name: Copy failed masks and matching raw files
    environment: uv@3.11:copy-files
    commands:
    - python
    - '%REPO%/standard_code/python/copy_files_with_extension.py'
    - --input-search-pattern: '%YAML%/masks/**/*_failed*.tif'
    - --optional-input-search-pattern: '%YAML%/raw/**/*.tif'
    - --output-folder: '%YAML%/failed_with_raw'
    - --collapse-delimiter: '__'
        """,
    )

    parser.add_argument(
        "--input-search-pattern",
        type=str,
        required=True,
        help="Input file pattern (supports wildcards, use '**' for recursive search)",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Output folder (default: base_folder + '_copied')",
    )
    parser.add_argument(
        "--collapse-delimiter",
        type=str,
        default="__",
        help="Delimiter used when flattening subfolder structure",
    )
    parser.add_argument(
        "--optional-input-search-pattern",
        type=str,
        default=None,
        help=(
            "Optional second pattern (e.g. raw images). "
            "Only files whose basename matches one of the failed basenames "
            "from --input-search-pattern will be copied"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copy actions without executing",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.version:
        version_file = Path(__file__).parent.parent.parent / "VERSION"
        try:
            version = version_file.read_text(encoding="utf-8").strip()
        except Exception:
            version = "unknown"
        print(f"copy_files_with_extension.py version: {version}")
        return

    process_files(
        input_pattern=args.input_search_pattern,
        output_folder=args.output_folder,
        collapse_delimiter=args.collapse_delimiter,
        optional_input_search_pattern=args.optional_input_search_pattern,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()


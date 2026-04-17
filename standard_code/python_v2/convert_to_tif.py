"""
Minimalistic image format converter using shared batch orchestration.

This v2 CLI keeps the existing single-file conversion behavior while delegating
file discovery, task building, dry-run preview, and batch execution to
bioimage_pipeline_utils.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

LEGACY_PYTHON_DIR = Path(__file__).resolve().parent.parent / "python"
if str(LEGACY_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_PYTHON_DIR))

import bioimage_pipeline_utils as rp
from convert_to_tif import convert_single_file


logger = logging.getLogger(__name__)


def _normalize_output_format(output_format: str) -> str:
    """Normalize output format with fallback when rp helper is unavailable."""
    if hasattr(rp, "normalize_output_format"):
        return rp.normalize_output_format(output_format)

    fmt = str(output_format).strip().lower()
    aliases = {
        "ome.tif": "ome.tif",
        "tif": "tif",
        "tiff": "tif",
        "npy": "npy",
        "ilastik-h5": "ilastik-h5",
        "h5": "ilastik-h5",
    }
    if fmt not in aliases:
        raise ValueError(f"Unsupported output format: {output_format}")
    return aliases[fmt]


def _output_extension_for_format(output_format: str, tiff_extension: str = ".ome.tif") -> str:
    """Resolve output extension with fallback when rp helper is unavailable."""
    if hasattr(rp, "output_extension_for_format"):
        return rp.output_extension_for_format(output_format, tiff_extension=tiff_extension)

    fmt = _normalize_output_format(output_format)
    if fmt in {"tif", "ome.tif"}:
        ext = tiff_extension
    elif fmt == "npy":
        ext = ".npy"
    else:
        ext = ".h5"
    if ext and not ext.startswith("."):
        ext = f".{ext}"
    return ext


def _resolve_default_output_folder(input_pattern: str, files: list[str]) -> str:
    """Match the current convert_to_tif default output folder behavior."""
    if '**' in input_pattern:
        base_folder = input_pattern.split('**')[0].rstrip('/\\')
        if not base_folder:
            base_folder = os.getcwd()
        base_folder = os.path.abspath(base_folder)
    else:
        base_folder = str(Path(files[0]).parent)
    return base_folder + "_tif"


def process_files(
    input_pattern: str,
    output_folder: Optional[str] = None,
    output_format: str = "tif",
    projection_method: Optional[str] = None,
    collapse_delimiter: str = "__",
    maxcores: Optional[int] = None,
    no_parallel: bool = False,
    save_metadata: bool = True,
    output_extension: str = "",
    dry_run: bool = False,
    standard_tif: bool = False,
    split: bool = False,
    scene_filter: str = "largest",
    scene_filter_strings: Optional[list[str]] = None,
    scene_merge_channel: bool = False,
) -> None:
    """Process multiple files matching a pattern."""
    if hasattr(rp, "discover_input_files"):
        files = rp.discover_input_files(input_pattern)
    else:
        files = rp.get_files_to_process2(input_pattern, search_subfolders='**' in input_pattern)
    if not files:
        logger.error("No files found matching pattern: %s", input_pattern)
        return

    logger.info("Found %s file(s) to process", len(files))

    if output_folder is None:
        output_folder = _resolve_default_output_folder(input_pattern, files)

    if split and _normalize_output_format(output_format) != "tif":
        raise ValueError("--split only supports --output-format tif")

    if hasattr(rp, "build_batch_tasks"):
        plan = rp.build_batch_tasks(
            input_pattern=input_pattern,
            output_folder=output_folder,
            output_extension=_output_extension_for_format(output_format, tiff_extension=".ome.tif"),
            output_suffix=output_extension,
            default_output_folder_suffix="_tif",
            collapse_delimiter=collapse_delimiter,
        )
    else:
        tasks = []
        output_ext = _output_extension_for_format(output_format, tiff_extension=".ome.tif")
        if '**' in input_pattern:
            base_folder = input_pattern.split('**')[0].rstrip('/\\')
            if not base_folder:
                base_folder = os.getcwd()
            base_folder = os.path.abspath(base_folder)
        else:
            base_folder = str(Path(files[0]).parent)
        for src in files:
            collapsed = rp.collapse_filename(src, base_folder, collapse_delimiter)
            out_name = os.path.basename(rp.resolve_output_path(collapsed, extension=output_ext, suffix=output_extension))
            tasks.append(SimpleNamespace(input_path=src, output_path=os.path.join(output_folder, out_name)))
        plan = SimpleNamespace(tasks=tasks, output_folder=output_folder)
    logger.info("Output folder: %s", plan.output_folder)

    if dry_run:
        dry_run_lines = []
        if projection_method:
            dry_run_lines.append(f"Projection method: {projection_method}")
        dry_run_lines.append(f"Scene filter: {scene_filter} (strings={scene_filter_strings})")
        dry_run_lines.append(f"Scene merge channel: {scene_merge_channel}")
        if hasattr(rp, "print_dry_run_preview"):
            rp.print_dry_run_preview(plan, extra_lines=dry_run_lines)
        else:
            print(f"[DRY RUN] Would process {len(plan.tasks)} files")
            print(f"[DRY RUN] Output folder: {plan.output_folder}")
            for line in dry_run_lines:
                print(f"[DRY RUN] {line}")
            for task in plan.tasks:
                print(f"[DRY RUN] {task.input_path} -> {task.output_path}")
        return

    process_kwargs = {
        "output_format": output_format,
        "projection_method": projection_method,
        "save_metadata": save_metadata,
        "standard_tif": standard_tif,
        "split": split,
        "scene_filter": scene_filter,
        "scene_filter_strings": scene_filter_strings,
        "scene_merge_channel": scene_merge_channel,
    }
    if hasattr(rp, "run_batch_tasks") and hasattr(rp, "summarize_batch_results"):
        results = rp.run_batch_tasks(
            tasks=plan.tasks,
            process_func=convert_single_file,
            process_kwargs=process_kwargs,
            maxcores=maxcores,
            no_parallel=no_parallel,
            skip_existing=False,
            force=False,
        )
        summary = rp.summarize_batch_results(results)
    else:
        file_pairs = [(task.input_path, task.output_path) for task in plan.tasks]
        failed = 0
        if no_parallel or len(file_pairs) == 1:
            for src, dst in file_pairs:
                if not convert_single_file(src, dst, **process_kwargs):
                    failed += 1
                    logger.error("Failed: %s", src)
        else:
            max_workers = rp.resolve_maxcores(maxcores, len(file_pairs))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(convert_single_file, src, dst, **process_kwargs): (src, dst)
                    for src, dst in file_pairs
                }
                for future in as_completed(futures):
                    src, _ = futures[future]
                    try:
                        if not future.result():
                            failed += 1
                            logger.error("Failed: %s", src)
                    except Exception as exc:
                        failed += 1
                        logger.error("Failed: %s (%s)", src, exc)

        summary = {
            "succeeded": len(file_pairs) - failed,
            "failed": failed,
        }
        results = []

    for result in results:
        if not result.success:
            logger.error("Failed: %s%s", result.input_path, f" ({result.error})" if result.error else "")

    logger.info(
        "Done: %s succeeded, %s failed",
        summary["succeeded"],
        summary["failed"],
    )


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Minimalistic image converter to OME-TIFF with optional Z-projection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Convert ND2 files to OME-TIFF (keep largest scene only)
  environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python_v2/convert_to_tif.py'
  - --input-search-pattern: '%YAML%/input/**/*.nd2'
  - --output-folder: '%YAML%/output'
  - --log-level: INFO

- name: Convert OBF files, keep only MLE scenes
  environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python_v2/convert_to_tif.py'
  - --input-search-pattern: '%YAML%/input/**/*.obf'
  - --output-folder: '%YAML%/output'
  - --scene-filter: includes
  - --scene-filter-strings: /MLE
  - --scene-merge-channel
  - --log-level: INFO
""",
    )

    parser.add_argument("--input-search-pattern", type=str, required=True, help="Input file pattern (supports wildcards, use '**' for recursive search)")
    parser.add_argument("--output-folder", type=str, default=None, help="Output folder (default: input_folder + '_tif')")
    parser.add_argument("--projection-method", type=str, default=None, choices=["max", "sum", "mean", "median", "min", "std"], help="Z-projection method (default: no projection)")
    parser.add_argument("--collapse-delimiter", type=str, default="__", help="Delimiter for collapsing subfolder paths (default: '__')")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing (process files sequentially)")
    parser.add_argument("--maxcores", type=int, default=None, help="Maximum CPU cores to use for parallel processing (default: all available CPU cores minus 1). Ignored if --no-parallel is set.")
    parser.add_argument("--no-metadata", action="store_true", help="Skip saving metadata YAML sidecars")
    parser.add_argument("--output-suffix", type=str, default="", help="Additional suffix to add before .ome.tif")
    parser.add_argument("--output-format", type=str, choices=["tif", "npy", "ilastik-h5"], default="tif", help="Output format (default: tif). Note: --split requires tif.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without executing")
    parser.add_argument("--standard-tif", action="store_true", help="Save as standard TIFF instead of OME-TIFF (better NIS-Elements compatibility)")
    parser.add_argument("--split", action="store_true", help="Save each T, C, Z slice as individual file in a folder (maximum compatibility)")
    parser.add_argument("--scene-filter", type=str, default="largest", choices=["all", "largest", "smallest", "includes", "excludes"], help="Scene selection strategy for multi-scene files (default: largest).")
    parser.add_argument("--scene-filter-strings", type=str, nargs="+", default=None, metavar="STRING", help="One or more substrings used with --scene-filter includes/excludes.")
    parser.add_argument("--scene-merge-channel", action="store_true", help="Group filtered scenes by HH:MM:SS timestamp in scene name and merge each timestamp-group into the channel dimension")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument("--log-level", type=str, default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level (default: WARNING)")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s - %(levelname)s - %(message)s')

    if args.version:
        version_file = Path(__file__).parent.parent.parent / "VERSION"
        try:
            version = version_file.read_text(encoding='utf-8').strip()
        except Exception:
            version = "unknown"
        print(f"convert_to_tif.py version: {version}")
        return

    process_files(
        input_pattern=args.input_search_pattern,
        output_folder=args.output_folder,
        output_format=args.output_format,
        projection_method=args.projection_method,
        collapse_delimiter=args.collapse_delimiter,
        maxcores=args.maxcores,
        no_parallel=args.no_parallel,
        save_metadata=not args.no_metadata,
        output_extension=args.output_suffix,
        dry_run=args.dry_run,
        standard_tif=args.standard_tif,
        split=args.split,
        scene_filter=args.scene_filter,
        scene_filter_strings=args.scene_filter_strings,
        scene_merge_channel=args.scene_merge_channel,
    )


if __name__ == "__main__":
    main()
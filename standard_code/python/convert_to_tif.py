import numpy as np
import os
import argparse
import yaml
import logging
from typing import Optional, Any, Tuple
from joblib import Parallel, delayed
import joblib
from tqdm import tqdm
from contextlib import contextmanager
import io
import time
from datetime import datetime
from bioio.writers import OmeTiffWriter  # type: ignore[import]
from bioio import BioImage

# Local imports
import run_pipeline_helper_functions as rp
from extract_metadata import get_all_metadata
from drift_correction import (
    drift_correct_xy_parallel as dc_cpu,
    drift_correct_xy_pygpureg as dc_gpu,
    drift_correct_xy_cupy as dc_cupy,
)

# Local tqdm-joblib integration: advance progress on job completion
@contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            try:
                tqdm_object.update(n=self.batch_size)
            finally:
                return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

# Minimal timing helper (currently unused; retained for future use)
@contextmanager
def _timed_step(verbose: bool, label: str):
    yield

def process_multipoint_file(input_file_path: str, output_tif_file_path: str, drift_correct_channel: int = -1, use_parallel: bool = True, projection_method: Optional[str] = None, args: Optional[argparse.Namespace] = None) -> None:
    """
    Process a multipoint file and convert it to TIF format with optional drift correction.
    """
    img = rp.load_bioio(input_file_path)
    if img is None:
        logging.error(f"Could not load image from file {input_file_path}. Skipping this file.")
        return

    for i, scene in enumerate(img.scenes):
        img.set_scene(scene)
        filename_base = os.path.splitext(os.path.basename(input_file_path))[0]
        scene_output_tif_file_path = os.path.join(os.path.dirname(output_tif_file_path), f"{filename_base}_S{i:02}.tif")

        # Process the scene
        process_file(
            img=img,
            input_file_path=input_file_path,
            output_tif_file_path=scene_output_tif_file_path,
            drift_correct_channel=drift_correct_channel,
            projection_method=projection_method,
            args=args,
        )
    
def process_file(img:BioImage, input_file_path: str, output_tif_file_path: str, drift_correct_channel: int = -1, projection_method: Optional[str] = None, args: Optional[argparse.Namespace] = None) -> None:
    # --- Helper functions (factored units) ---

    def get_physical_pixel_sizes_safe(image: BioImage, src_path: str, out_path: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        try:
            pps = image.physical_pixel_sizes if image.physical_pixel_sizes is not None else (None, None, None)
        except Exception as e:
            logging.warning(f"Error retrieving physical pixel sizes: {e} for file {src_path}. Using None.")
            with open(os.path.splitext(out_path)[0] + "_error.txt", 'w') as f:
                f.write(f"Error retrieving physical pixel sizes: {e} for file {src_path}. Using None.\n")
            pps = (None, None, None)
        return pps  # type: ignore[return-value]

    def load_or_derive_metadata(src_path: str, out_path: str) -> dict:
        input_metadata_file_path: str = os.path.splitext(src_path)[0] + "_metadata.yaml"
        metadata: dict[str, Any] = {}
        if os.path.exists(input_metadata_file_path):
            with open(input_metadata_file_path, 'r') as f:
                loaded_md = yaml.safe_load(f)
            if isinstance(loaded_md, dict):
                metadata = loaded_md
        else:
            try:
                loaded_md = get_all_metadata(src_path)
                if isinstance(loaded_md, dict):
                    metadata = loaded_md
            except Exception as e:
                logging.warning(f"Error retrieving metadata: {e} for file {src_path}. Using None.")
                metadata = {"Error": f"Error retrieving metadata: {e} for file {src_path}. Using None."}
                with open(os.path.splitext(out_path)[0] + "_metadata_error.txt", 'w') as f:
                    f.write(f"Error retrieving metadata: {e} for file {src_path}. Using None.\n")
        return metadata

    def project_image(image: BioImage, proj_method: Optional[str], initial_dtype: Any) -> np.ndarray:
        def _project_eager(base_np: np.ndarray) -> np.ndarray:
            if proj_method == "max":
                return base_np.max(axis=2, keepdims=True)
            elif proj_method == "sum":
                logging.warning("Using sum projection. If the sum exceeds the original dtype, the result will be cast to a higher dtype to avoid saturation.")
                up_dtype = None
                if initial_dtype == np.uint8:
                    up_dtype = np.uint16
                elif initial_dtype == np.uint16:
                    up_dtype = np.uint32
                elif initial_dtype == np.uint32:
                    up_dtype = np.uint64
                work = base_np.astype(up_dtype) if up_dtype is not None else base_np
                out = work.sum(axis=2, keepdims=True)
                # Downcast if safe / match previous behavior
                if initial_dtype == np.uint8:
                    if np.any(out > 255):
                        print(f"Sum exceeds uint8 range. Upcasting to uint16. Number of saturated pixels (would be): {np.sum(out > 255)}")
                        return out.astype(np.uint16)
                    return out.astype(np.uint8)
                elif initial_dtype == np.uint16:
                    if np.any(out > 65535):
                        print(f"Sum exceeds uint16 range. Upcasting to uint32. Number of saturated pixels (would be): {np.sum(out > 65535)}")
                        return out.astype(np.uint32)
                    return out.astype(np.uint16)
                elif initial_dtype == np.uint32:
                    if np.any(out > 4294967295):
                        print(f"Sum exceeds uint32 range. Upcasting to float64. Number of saturated pixels (would be): {np.sum(out > 4294967295)}")
                        return out.astype(np.float64)
                    return out.astype(np.uint32)
                elif initial_dtype == np.float32 or initial_dtype == np.float64:
                    return out.astype(initial_dtype)
                else:
                    raise ValueError(f"Unsupported dtype: {initial_dtype}")
            elif proj_method == "mean":
                return base_np.mean(axis=2, keepdims=True)
            elif proj_method == "median":
                return np.median(base_np, axis=2, keepdims=True)
            elif proj_method == "min":
                return base_np.min(axis=2, keepdims=True)
            elif proj_method == "std":
                out = base_np.std(axis=2, keepdims=False)
                return np.expand_dims(out, axis=2)
            else:
                return base_np

        # Load full array into RAM (fastest path when memory allows)
        base = image.data
        compute_fn = getattr(base, "compute", None)
        base_np: np.ndarray
        if callable(compute_fn):
            base_np = np.asarray(compute_fn())
        else:
            base_np = np.asarray(base)
        img_data_local = _project_eager(base_np)
        return img_data_local

    def apply_drift_correction_if_needed(
        img_data: np.ndarray,
        drift_channel: int,
        drift_threads_val: int,
        method: str,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[str]]:
        if drift_channel <= -1:
            return img_data, None, None

        m = (method or "gpu").lower()
        if m == "cpu":
            output_img_local, tmats_local = dc_cpu(
                img_data, drift_correct_channel=drift_channel, transform_threads=drift_threads_val
            )
            return output_img_local, tmats_local, "StackReg"
        elif m == "cupy":
            # Force max-accuracy settings for CuPy in this tool
            output_img_local, shifts_local = dc_cupy(
                img_data,
                drift_correct_channel=drift_channel,
                downsample=1,
                refine_upsample=50,
                use_running_ref=True,
                highpass_pix=None,
                lowpass_pix=None,
            )
            if shifts_local is None:
                raise RuntimeError("CuPy drift correction unavailable or failed; aborting because fallback not requested.")
            return output_img_local, shifts_local, "CuPy"
        elif m == "gpu":
            output_img_local, shifts_local = dc_gpu(img_data, drift_correct_channel=drift_channel)
            if shifts_local is None:
                raise RuntimeError("GPU (pyGPUreg) drift correction unavailable or failed; aborting because fallback not requested.")
            return output_img_local, shifts_local, "pyGPUreg"
        elif m == "auto":
            # Try CuPy (defaults) → pyGPUreg → CPU
            out, shifts = dc_cupy(img_data, drift_correct_channel=drift_channel)
            if shifts is not None:
                return out, shifts, "CuPy"
            out, shifts = dc_gpu(img_data, drift_correct_channel=drift_channel)
            if shifts is not None:
                return out, shifts, "pyGPUreg"
            out, tmats = dc_cpu(img_data, drift_correct_channel=drift_channel, transform_threads=drift_threads_val)
            return out, tmats, "StackReg"
        else:
            raise ValueError("drift method must be one of: 'cpu', 'gpu', 'cupy', 'auto'")

    def save_outputs(image_to_save: np.ndarray, shifts: Optional[np.ndarray], phys_px_sizes: Tuple[Optional[float], Optional[float], Optional[float]],
                     out_tif_path: str, out_shifts_path: str, metadata_dict: dict, proj_method: Optional[str], drift_channel: int, out_md_path: str, drift_method: Optional[str]) -> None:
        OmeTiffWriter.save(image_to_save, out_tif_path, dim_order="TCZYX", physical_pixel_sizes=phys_px_sizes)
        if shifts is not None:
            np.save(out_shifts_path, shifts)

        convert_info: dict[str, Any] = {
            "Projection": {"Method": proj_method}
        }
        if shifts is not None:
            convert_info["Drift correction"] = {
                "Method": drift_method,
                "Drift_correct_channel": drift_channel,
                "Shifts": os.path.basename(out_shifts_path),
            }
        metadata_dict["Convert to tif"] = convert_info
        with open(out_md_path, 'w') as f:
            yaml.dump(metadata_dict, f)

    # --- Begin refactored process flow ---
    input_metadata_file_path: str = os.path.splitext(input_file_path)[0] + "_metadata.yaml"
    output_metadata_file_path: str = os.path.splitext(output_tif_file_path)[0] + "_metadata.yaml"
    output_shifts_file_path: str = os.path.splitext(output_tif_file_path)[0] + "_shifts.npy"

    # SAFEGUARD: Prevent writing to the same file as input
    if os.path.abspath(input_file_path) == os.path.abspath(output_tif_file_path):
        logging.error(f"Output file path matches input file path! Aborting to prevent overwriting original file: {input_file_path}")
        return
    if os.path.abspath(input_metadata_file_path) == os.path.abspath(output_metadata_file_path):
        logging.error(f"Output metadata file path matches input metadata file path! Aborting to prevent overwriting original metadata: {input_metadata_file_path}")
        return

    # Determine dtype early
    try:
        initial_dtype = img.data.dtype
    except Exception as e:
        print(f"Error: The image data does not have a 'dtype' attribute {e}.\n Skipping this file: {input_file_path}")
        return

    physical_pixel_sizes = get_physical_pixel_sizes_safe(img, input_file_path, output_tif_file_path)

    metadata = load_or_derive_metadata(input_file_path, output_tif_file_path)

    # Fixed default threads for CPU StackReg transforms (kept simple since --drift-threads is removed)
    drift_threads_val = 4

    img_data = project_image(img, projection_method, initial_dtype)

    drift_method = str(getattr(args, "drift_correct_method", "gpu")) if args is not None else "gpu"
    output_img, shifts, drift_method = apply_drift_correction_if_needed(
        img_data, drift_correct_channel, drift_threads_val, drift_method
    )

    save_outputs(output_img, shifts, physical_pixel_sizes, output_tif_file_path, output_shifts_file_path, metadata, projection_method, drift_correct_channel, output_metadata_file_path, drift_method)

def process_folder(args: argparse.Namespace, parallel: bool = True) -> None:
    # Determine if input is a glob pattern or a directory
    is_glob = any(x in args.input_search_pattern for x in ["*", "?", "["])
    if is_glob:
        files_to_process = rp.get_files_to_process2(
            args.input_search_pattern,
            getattr(args, 'search_subfolders', False) or False
        )
        base_folder = os.path.dirname(args.input_search_pattern) or "."
    else:
        # Folder mode: rely entirely on search pattern semantics; include all files by default
        recursive = getattr(args, 'search_subfolders', False) or False
        pattern = os.path.join(args.input_search_pattern, "**", "*") if recursive else os.path.join(args.input_search_pattern, "*")
        files_to_process = rp.get_files_to_process2(pattern, recursive)
        base_folder = args.input_search_pattern

    # Allow overriding destination folder via CLI; fall back to existing base_folder + "_tif"
    destination_folder = args.output_folder if getattr(args, "output_folder", None) else base_folder + "_tif"
    os.makedirs(destination_folder, exist_ok=True)

    # Safety: Bio-Formats/JVM is not friendly with multi-proc + threaded Dask. If IMS present, run sequentially.
    if parallel and any(str(f).lower().endswith(".ims") for f in files_to_process):
        logging.warning("IMS detected; disabling parallel processing to avoid JVM/threading crashes.")
        parallel = False

    def process_single_file(input_file_path):
        output_tif_file_path: str = rp.collapse_filename(input_file_path, base_folder, args.collapse_delimiter)
        output_tif_file_path: str = os.path.splitext(output_tif_file_path)[0] + args.output_file_name_extension + ".tif"
        output_tif_file_path: str = os.path.join(destination_folder, output_tif_file_path)

        def _run():
            if args.multipoint_files:
                logging.info(f"Processing multipoint file: {input_file_path}.")
                process_multipoint_file(
                    input_file_path,
                    output_tif_file_path,
                    drift_correct_channel=args.drift_correct_channel,
                    use_parallel=True,
                    projection_method=args.projection_method,
                    args=args,
                )
            else:
                img = rp.load_bioio(input_file_path)
                process_file(
                    img=img,
                    input_file_path=input_file_path,
                    output_tif_file_path=output_tif_file_path,
                    drift_correct_channel=args.drift_correct_channel,
                    projection_method=args.projection_method,
                    args=args
                )
        _run()

    if parallel:
        # Advance progress when each job completes
        with tqdm_joblib(tqdm(total=len(files_to_process), desc="Processing files", unit="file")):
            Parallel(n_jobs=-1)(delayed(process_single_file)(file) for file in files_to_process)
    else:
        # Update only after each file has finished processing
        with tqdm(total=len(files_to_process), desc="Processing files", unit="file") as pbar:
            for input_file_path in files_to_process:
                process_single_file(input_file_path)
                pbar.update(1)


def main(parsed_args: argparse.Namespace, parallel: bool = True) -> None:
    inp = parsed_args.input_search_pattern
    if os.path.isfile(inp):
        logging.info(f"Processing single file: {inp}")
        # If an explicit output folder is provided, save into that folder; otherwise keep current behavior
        if getattr(parsed_args, "output_folder", None):
            os.makedirs(parsed_args.output_folder, exist_ok=True)
            output_filename = os.path.basename(os.path.splitext(inp)[0]) + parsed_args.output_file_name_extension + ".tif"
            output_tif_file_path = os.path.join(parsed_args.output_folder, output_filename)
        else:
            output_tif_file_path = os.path.splitext(inp)[0] + parsed_args.output_file_name_extension + ".tif"
        
        if parsed_args.multipoint_files:
            logging.info("Processing multipoint file.")
            process_multipoint_file(
                inp,
                output_tif_file_path,
                drift_correct_channel=parsed_args.drift_correct_channel,
                use_parallel=parallel,
                projection_method=parsed_args.projection_method,
                args=parsed_args,
            )
        else:
            img = rp.load_bioio(inp)
            process_file(
                img=img,
                input_file_path=inp,
                output_tif_file_path=output_tif_file_path,
                drift_correct_channel=parsed_args.drift_correct_channel,
                projection_method=parsed_args.projection_method,
                args=parsed_args
            )
    else:
        # Treat as folder or glob and process batch
        if os.path.isdir(inp):
            logging.info(f"Processing folder: {inp}")
        else:
            logging.info(f"Processing pattern: {inp}")
        process_folder(parsed_args, parallel=parallel)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Process a folder (or file) of images and convert them to TIF format with optional drift correction. "
            "Use --drift-correct-channel <idx> to enable drift correction and --drift-correct-method {cpu,gpu,cupy,auto} to choose the algorithm."
        )
    )
    parser.add_argument("--input-search-pattern", type=str, required=True, help="Glob pattern to search for input files (e.g. './data/*.tif')")
    parser.add_argument("--search-subfolders", action="store_true", help="Enable recursive search when using a glob pattern")
    parser.add_argument("--collapse-delimiter", type=str, default="__", help="Delimiter to collapse file paths (Only applicable for directory input)")

    parser.add_argument("--projection-method", type=str, default=None, help="Method for projection (options: max, sum, mean, median, min, std)")
    parser.add_argument("--multipoint-files", action="store_true", help="Not implemented yet: Store the output in a multipoint file /series as separate_files (default: False)")

    parser.add_argument("--drift-correct-channel", type=int, default=-1, help="Channel for drift correction (default: -1, no correction)")
    parser.add_argument(
        "--drift-correct-method",
        type=str,
        default="cupy",
        choices=["cpu", "gpu", "cupy", "auto"],
        help="Drift correction algorithm: cpu=StackReg, gpu=pyGPUreg (default), cupy=CuPy phase correlation, auto=try cupy→gpu→cpu.",
    )

    parser.add_argument("--no-parallel", action="store_true", help="Do not use parallel processing")

    parser.add_argument("--output-file-name-extension", type=str, default="", help="Output file name extension (e.g., '_max', do not include '.tif')")
    parser.add_argument("--output-folder", type=str, required=False, help="Path to save the processed files")

    parsed_args = parser.parse_args()

    # Default to warnings-only logging to keep normal runs quiet
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    parallel = not parsed_args.no_parallel

    main(parsed_args, parallel=parallel)

import os
import argparse
import tempfile
from typing import Optional
import numpy as np
import tifffile
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
import trackpy as tp
import logging

# Local imports
import bioimage_pipeline_utils as rp


def _to_py_int(val):
    """Convert a possibly-numpy scalar to a plain Python int safely."""
    try:
        return int(val.item())
    except Exception:
        return int(val)


def track_maxima(
    input_image: rp.BioImage,
    search_range: int = 10,
    min_distance: int = 10,
    threshold_abs: Optional[float] = None,
    threshold_rel: Optional[float] = None,
    footprint_size: Optional[int] = None,
    gaussian_sigma: float = 1.0,
    exclude_border: bool = True,
    num_peaks: Optional[int] = None,
    output_mask_path: Optional[str] = None,
    output_format: str = "tif",
):
    dims = input_image.dims
    T, C, Z, Y, X = int(dims.T), int(dims.C), int(dims.Z), int(dims.Y), int(dims.X)

    records = []
    footprint = None if footprint_size is None else np.ones((footprint_size, footprint_size), dtype=bool)
    np_num_peaks = np.inf if num_peaks is None else int(num_peaks)

    estimated_output_gib = (T * Z * Y * X * np.dtype(np.uint16).itemsize) / (1024 ** 3)
    normalized_format = str(output_format).lower()
    write_direct_tiff = output_mask_path is not None and normalized_format in {"tif", "tiff", "ome.tif"}

    temp_mask_path = None
    new_masks = None
    if write_direct_tiff and output_mask_path is not None:
        if os.path.exists(output_mask_path):
            os.remove(output_mask_path)
        logging.info(
            "Creating disk-backed labeled TIFF at %s (~%.2f GiB logical size).",
            output_mask_path,
            estimated_output_gib,
        )
        new_masks = tifffile.memmap(
            output_mask_path,
            shape=(T, 1, Z, Y, X),
            dtype=np.uint16,
            bigtiff=True,
            metadata={"axes": "TCZYX"},
        )
        new_masks[:] = 0
    elif output_mask_path is not None:
        fd, temp_mask_path = tempfile.mkstemp(prefix="track_maxima_", suffix=".dat", dir=tempfile.gettempdir())
        os.close(fd)
        new_masks = np.memmap(temp_mask_path, dtype=np.uint16, mode="w+", shape=(T, 1, Z, Y, X))
        new_masks[:] = 0

    lazy_data = getattr(input_image, "dask_data", None)

    for t in range(T):
        for z in range(Z):
            if lazy_data is not None:
                block = lazy_data[t : t + 1, :, z : z + 1, :, :].compute()
                block_np = np.asarray(block)
                mask2d = block_np[0, 0, 0]
            else:
                mask2d = np.asarray(input_image.get_image_data("YX", T=t, C=0, Z=z))

            filtered = gaussian_filter(mask2d, sigma=gaussian_sigma)
            maxima = peak_local_max(
                filtered,
                min_distance=min_distance,
                threshold_abs=threshold_abs,
                threshold_rel=threshold_rel,
                footprint=footprint,
                exclude_border=exclude_border,
                num_peaks=np_num_peaks,
            )
            for maxima_idx in maxima:
                y_coord = _to_py_int(maxima_idx[0])
                x_coord = _to_py_int(maxima_idx[1])
                records.append({
                    "frame": t,
                    "z": z,
                    "y": y_coord,
                    "x": x_coord,
                    "label": len(records) + 1,
                    "t_local": t * Z + z,
                })

    df = pd.DataFrame(records)

    if df.empty:
        try:
            if new_masks is not None:
                del new_masks
            if temp_mask_path:
                os.remove(temp_mask_path)
        except Exception:
            pass
        raise ValueError("No maxima found in the selected mask for tracking.")

    logging.getLogger("trackpy").setLevel(logging.WARNING)
    df_tracked = tp.link_df(df, search_range=search_range, pos_columns=["x", "y"], t_column="t_local")

    for row in df_tracked.itertuples():
        t_idx = _to_py_int(row.frame)
        z_idx = _to_py_int(row.z)
        y_coord = _to_py_int(row.y)
        x_coord = _to_py_int(row.x)
        label_val = _to_py_int(row.label)
        if new_masks is not None:
            new_masks[t_idx, 0, z_idx, y_coord, x_coord] = label_val

    if output_mask_path:
        csv_path = os.path.splitext(output_mask_path)[0] + ".csv"
        df_tracked.to_csv(csv_path, index=False)

        if new_masks is not None:
            new_masks.flush()
            if not write_direct_tiff:
                save_kwargs = {"dim_order": "TCZYX"}
                if normalized_format in {"tif", "tiff", "ome.tif"}:
                    save_kwargs["compression"] = "zlib"
                rp.save_with_output_format(new_masks, output_mask_path, output_format, **save_kwargs)
            del new_masks
            try:
                if temp_mask_path is not None:
                    os.remove(temp_mask_path)
            except OSError:
                pass

        return df_tracked, None

    return df_tracked, new_masks


def process_file(
    input_file_path: str,
    output_file_path: str,
    search_range: int,
    min_distance: int,
    threshold_abs: float,
    threshold_rel: float,
    footprint_size: int,
    gaussian_sigma: float,
    exclude_border: bool,
    num_peaks: int,
    output_format: str,
):
    img = rp.load_tczyx_image(input_file_path)
    df, new_mask = track_maxima(
        img,
        search_range=search_range,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        footprint_size=footprint_size,
        gaussian_sigma=gaussian_sigma,
        exclude_border=exclude_border,
        num_peaks=num_peaks,
        output_mask_path=output_file_path,
        output_format=output_format,
    )
    return df


def process_folder_or_file(args: argparse.Namespace):
    recursive = getattr(args, 'search_subfolders', False) or False
    if os.path.isdir(args.input_search_pattern):
        ext = getattr(args, 'extension', '') or '.tif'
        pattern = os.path.join(args.input_search_pattern, '**', f'*{ext}') if recursive else os.path.join(args.input_search_pattern, f'*{ext}')
        base_folder = args.input_search_pattern
    else:
        pattern = args.input_search_pattern
        base_folder = os.path.dirname(pattern) or '.'
    input_files = rp.get_files_to_process2(pattern, recursive)

    if not input_files:
        print("No files found to process.")
        return

    destination_folder = args.output_folder or (
        (args.input_search_pattern.rstrip(os.sep) if os.path.isdir(args.input_search_pattern) else base_folder.rstrip(os.sep))
        + "_tracked"
    )
    os.makedirs(destination_folder, exist_ok=True)

    def process_single_file(input_file_path):
        output_file_name = rp.collapse_filename(input_file_path, base_folder, args.collapse_delimiter)
        output_ext = rp.output_extension_for_format(args.output_format, tiff_extension=".tif")
        output_file_name = os.path.basename(
            rp.resolve_output_path(output_file_name, extension=output_ext, suffix=args.output_suffix)
        )
        output_file_path = os.path.join(destination_folder, output_file_name)
        process_file(
            input_file_path,
            output_file_path,
            search_range=args.search_range,
            min_distance=args.min_distance,
            threshold_abs=args.threshold_abs,
            threshold_rel=args.threshold_rel,
            footprint_size=args.footprint_size,
            gaussian_sigma=args.gaussian_sigma,
            exclude_border=args.exclude_border,
            num_peaks=args.num_peaks,
            output_format=args.output_format,
        )

    use_parallel = not args.no_parallel
    if use_parallel and input_files:
        try:
            sample_img = rp.load_tczyx_image(input_files[0])
            sample_dims = sample_img.dims
            estimated_output_gib = (
                int(sample_dims.T) * int(sample_dims.Z) * int(sample_dims.Y) * int(sample_dims.X) * 2
            ) / (1024 ** 3)
            if estimated_output_gib > 0.5:
                logging.warning(
                    "Large sparse output detected (~%.2f GiB dense per file). Switching to sequential mode to avoid disk and memory exhaustion.",
                    estimated_output_gib,
                )
                use_parallel = False
        except Exception as exc:
            logging.warning("Could not estimate output size for parallel safety check: %s", exc)

    if use_parallel:
        from contextlib import contextmanager
        from joblib import Parallel, delayed

        @contextmanager
        def _tqdm_joblib(tqdm_object):
            import joblib
            class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
                def __call__(self, *args, **kwargs):
                    tqdm_object.update(n=self.batch_size)
                    return super().__call__(*args, **kwargs)
            old_batch_callback = joblib.parallel.BatchCompletionCallBack
            joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
            try:
                yield tqdm_object
            finally:
                joblib.parallel.BatchCompletionCallBack = old_batch_callback
                tqdm_object.close()

        with tqdm(total=len(input_files), desc="Tracking maxima", unit="file") as pbar:
            with _tqdm_joblib(pbar):
                Parallel(n_jobs=rp.resolve_maxcores(args.maxcores, len(input_files)))(delayed(process_single_file)(file) for file in input_files)
    else:
        for input_file_path in tqdm(input_files, desc="Tracking maxima", unit="file"):
            process_single_file(input_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track local maxima across time using TrackPy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Track maxima over time
  environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/track_maxima.py'
  - --input-search-pattern: '%YAML%/masks/**/*_mask.tif'
  - --output-folder: '%YAML%/tracked_maxima'

- name: Track maxima recursively with search range 20
  environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/track_maxima.py'
  - --input-search-pattern: '%YAML%/masks/**/*.tif'
  - --output-folder: '%YAML%/tracked_maxima'
  - --search-range: 20
  - --search-subfolders
        """
    )
    parser.add_argument("--input-search-pattern", type=str, required=True, help="Glob pattern to search for input files (e.g. './data/*.tif')")
    parser.add_argument("--extension", type=str, default=".tif", help="File extension to search for.")
    parser.add_argument("--search-subfolders", action="store_true", help="Search recursively in subfolders.")
    parser.add_argument("--collapse-delimiter", type=str, default="__", help="Delimiter for flattening output filenames.")
    parser.add_argument("--search-range", type=int, default=10, help="Distance between maxima to consider them as the same (default: 10).")
    parser.add_argument("--min-distance", type=int, default=10, help="Minimum distance between peaks for peak_local_max (pixels).")
    parser.add_argument("--gaussian-sigma", type=float, default=1.0, help="Gaussian sigma to smooth input before peak detection (default: 1.0).")
    parser.add_argument("--threshold-abs", type=float, default=None, help="Absolute intensity threshold for peak detection (default: None).")
    parser.add_argument("--threshold-rel", type=float, default=None, help="Relative intensity threshold (fraction of max) for peak detection (default: None).")
    parser.add_argument("--footprint-size", type=int, default=None, help="Square footprint side length for local peak detection (default: None=auto).")
    parser.add_argument("--no-exclude-border", dest="exclude_border", action="store_false", default=True, help="Do not exclude peaks touching the image border.")
    parser.add_argument("--num-peaks", type=int, default=None, help="Maximum number of peaks to return per slice (default: unlimited).")
    parser.add_argument("--output-folder", type=str, default=None, help="Output folder for tracked mask files (default: <input_root>_tracked).")
    parser.add_argument("--output-suffix", type=str, default="_tracked", help="Suffix to append to output file name (default: '_tracked').")
    parser.add_argument("--output-format", type=str, choices=["tif", "npy", "ilastik-h5"], default="tif", help="Output format (default: tif)")
    parser.add_argument("--no-parallel", action="store_true", help="Do not use parallel processing.")
    parser.add_argument("--maxcores", type=int, default=None, help="Maximum CPU cores to use for parallel processing (default: all available CPU cores minus 1). Ignored if --no-parallel is set.")
    parser.add_argument("--log-level", type=str, default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level (default: WARNING)")

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    process_folder_or_file(args)
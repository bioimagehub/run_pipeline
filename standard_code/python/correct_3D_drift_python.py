from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from skimage.registration import phase_cross_correlation

import bioimage_pipeline_utils as rp


"""
This is an attempt to copy the functionality of the ImageJ/Fiji Correct 3D Drift plugin in pure Python, without any Java dependencies.

All credit goes to the original authors 
https://github.com/fiji/Correct_3D_Drift/blob/master/src/main/resources/scripts/Plugins/Registration/Correct_3D_drift.py
###
# #%L
# Script to register time frames (stacks) to each other.
# %%
# Copyright (C) 2010 - 2024 Fiji developers.
# %%
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with this program.  If not, see
# <http://www.gnu.org/licenses/gpl-3.0.html>.
# #L%
###


"""


logger = logging.getLogger(__name__)

@dataclass
class ShiftVec:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "ShiftVec") -> "ShiftVec":
        return ShiftVec(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "ShiftVec") -> "ShiftVec":
        return ShiftVec(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "ShiftVec":
        return ShiftVec(self.x * scalar, self.y * scalar, self.z * scalar)

    def to_int(self) -> tuple[int, int, int]:
        # Match Java rounding semantics from original IJ path.
        return (
            int(math.floor(self.x + 0.5)),
            int(math.floor(self.y + 0.5)),
            int(math.floor(self.z + 0.5)),
        )


def _extract_2d_frame(
    frame_getter: Callable[[int], np.ndarray],
    t: int,
    z_min: int,
    z_max: int,
    background: float,
    correct_only_xy: bool,
) -> np.ndarray:
    frame = frame_getter(t)
    z0 = z_min - 1
    z1 = z_max
    vol = frame[z0:z1, :, :]

    if background > 0:
        vol = vol.astype(np.float32) - background
        vol = np.clip(vol, 0, None)

    nz = vol.shape[0]
    if nz == 1 or correct_only_xy:
        return vol.mean(axis=0)
    return vol


def _clamp(val: float, limit: float) -> float:
    return max(-limit, min(limit, val))


def _compute_shift_pair(
    ref: np.ndarray,
    moving: np.ndarray,
    max_shifts: tuple[float, float, float],
) -> ShiftVec:
    ref_f = ref.astype(np.float32)
    mov_f = moving.astype(np.float32)

    shift, _, _ = phase_cross_correlation(
        ref_f,
        mov_f,
        normalization="phase",
        upsample_factor=1,
    )

    if ref.ndim == 2:
        dy, dx = float(shift[0]), float(shift[1])
        dz = 0.0
    else:
        dz, dy, dx = float(shift[0]), float(shift[1]), float(shift[2])

    dx = _clamp(dx, max_shifts[0])
    dy = _clamp(dy, max_shifts[1])
    dz = _clamp(dz, max_shifts[2])

    return ShiftVec(dx, dy, dz)


def compute_all_shifts(
    frame_getter: Callable[[int], np.ndarray],
    nt: int,
    z_min: int,
    z_max: int,
    background: float,
    correct_only_xy: bool,
    max_shifts: tuple[float, float, float],
    multi_time_scale: bool,
    verbose: bool = True,
) -> list[ShiftVec]:
    shifts: list[ShiftVec] = [ShiftVec() for _ in range(nt)]

    def _run_dt(dt: int, current_shifts: list[ShiftVec]) -> list[ShiftVec]:
        if verbose:
            logger.info(f"Computing drift at dt={dt} ...")
        for t in range(dt, nt + dt, dt):
            if t > nt - 1:
                t = nt - 1
            t_prev = t - dt
            if verbose:
                logger.info(f"  Between frames {t_prev + 1} and {t + 1}")

            # Keep argument order consistent with IJ plugin semantics.
            ref = _extract_2d_frame(
                frame_getter, t, z_min, z_max, background, correct_only_xy
            )
            mov = _extract_2d_frame(
                frame_getter, t_prev, z_min, z_max, background, correct_only_xy
            )

            local_new_shift = _compute_shift_pair(ref, mov, max_shifts)
            local_shift = current_shifts[t] - current_shifts[t_prev]
            add_shift = local_new_shift - local_shift

            for i, tt in enumerate(range(t_prev, nt)):
                factor = i / dt
                current_shifts[tt].x += factor * add_shift.x
                current_shifts[tt].y += factor * add_shift.y
                current_shifts[tt].z += factor * add_shift.z

        return current_shifts

    if verbose:
        logger.info("Computing drift at frame shifts of 1")
    shifts = _run_dt(1, shifts)

    if multi_time_scale:
        dt_max = nt - 1
        for dt in [3, 9, 27, 81, 243, 729, dt_max]:
            if dt >= dt_max:
                if verbose:
                    logger.info(f"Computing drift at frame shifts of {dt_max}")
                shifts = _run_dt(dt_max, shifts)
                break
            if verbose:
                logger.info(f"Computing drift at frame shifts of {dt}")
            shifts = _run_dt(dt, shifts)

    return shifts


def invert_shifts(shifts: list[ShiftVec]) -> list[ShiftVec]:
    return [ShiftVec(-s.x, -s.y, -s.z) for s in shifts]


def shifts_to_int(shifts: list[ShiftVec]) -> list[tuple[int, int, int]]:
    return [s.to_int() for s in shifts]


def apply_integer_shifts(
    frame_getter: Callable[[int], np.ndarray],
    nt: int,
    nc: int,
    nz: int,
    nh: int,
    nw: int,
    dtype: np.dtype,
    shifts: list[tuple[int, int, int]],
) -> np.memmap:
    if len(shifts) != nt:
        raise ValueError(
            f"Number of shifts ({len(shifts)}) must match number of frames ({nt})."
        )

    xs = [s[0] for s in shifts]
    ys = [s[1] for s in shifts]
    zs = [s[2] for s in shifts]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    new_w = nw + max_x - min_x
    new_h = nh + max_y - min_y
    new_z = nz + max_z - min_z

    tmp_file = tempfile.NamedTemporaryFile(prefix="drift_corrected_", suffix=".mmap", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()
    out = np.memmap(tmp_path, mode="w+", dtype=dtype, shape=(nt, nc, new_z, new_h, new_w))
    out[:] = 0

    for t in range(nt):
        dx, dy, dz = shifts[t]
        ox = dx - min_x
        oy = dy - min_y
        oz = dz - min_z
        out[t, :, oz : oz + nz, oy : oy + nh, ox : ox + nw] = frame_getter(t)

    out.flush()

    return out


def translate_subpixel(
    data: np.ndarray,
    shifts: list[tuple[float, float, float]],
) -> np.ndarray:
    raise NotImplementedError(
        "Sub-pixel correction is not yet implemented. "
        "Run without --subpixel to use integer-pixel correction."
    )


@dataclass
class Options:
    channel: int = 1
    correct_only_xy: bool = False
    multi_time_scale: bool = False
    subpixel: bool = False
    edge_enhance: bool = False
    background: float = 0.0
    z_min: int = 1
    z_max: int | None = None
    max_shift_x: float = 10.0
    max_shift_y: float = 10.0
    max_shift_z: float = 10.0
    only_compute: bool = False
    shifts_file: str | None = None
    use_lazy_loading: bool = False
    verbose: bool = True


def load_shifts_txt(path: str | Path) -> list[tuple[int, int, int]]:
    shifts = []
    with open(path) as fh:
        lines = fh.readlines()

    in_shifts = False
    for line in lines:
        stripped = line.strip()
        if stripped == "Shifts":
            in_shifts = True
            continue
        if stripped == "dx\tdy\tdz":
            continue
        if in_shifts and stripped:
            parts = stripped.split("\t")
            shifts.append(
                (
                    int(round(float(parts[0]))),
                    int(round(float(parts[1]))),
                    int(round(float(parts[2]))),
                )
            )
    return shifts


def save_shifts_txt(
    path: str | Path,
    shifts: list[tuple[int, int, int]],
    data_shape: tuple,
) -> None:
    _, _, nz, nh, nw = data_shape
    lines = []
    lines.append("ROI zero-based")
    lines.append("\nx_min\ty_min\tz_min\tx_max\ty_max\tz_max")
    lines.append(f"\n0\t0\t0\t{nw - 1}\t{nh - 1}\t{nz - 1}")
    lines.append("\nShifts")
    lines.append("\ndx\tdy\tdz")
    for dx, dy, dz in shifts:
        lines.append(f"\n{dx}\t{dy}\t{dz}")
    with open(path, "w") as fh:
        fh.writelines(lines)


def _build_channel_frame_getter(
    img,
    channel_index: int,
    use_lazy_loading: bool,
) -> Callable[[int], np.ndarray]:
    if use_lazy_loading and hasattr(img, "dask_data") and img.dask_data is not None:
        dask_data = img.dask_data

        def _get_frame_lazy(t: int) -> np.ndarray:
            return np.asarray(dask_data[t, channel_index, :, :, :].compute())

        return _get_frame_lazy

    np_data = np.asarray(img.data)

    def _get_frame_eager(t: int) -> np.ndarray:
        return np.asarray(np_data[t, channel_index, :, :, :])

    return _get_frame_eager


def _build_full_frame_getter(
    img,
    use_lazy_loading: bool,
) -> Callable[[int], np.ndarray]:
    if use_lazy_loading and hasattr(img, "dask_data") and img.dask_data is not None:
        dask_data = img.dask_data

        def _get_full_frame_lazy(t: int) -> np.ndarray:
            return np.asarray(dask_data[t, :, :, :, :].compute())

        return _get_full_frame_lazy

    np_data = np.asarray(img.data)

    def _get_full_frame_eager(t: int) -> np.ndarray:
        return np.asarray(np_data[t, :, :, :, :])

    return _get_full_frame_eager


def run_single_file(
    input_path: str | Path,
    output_path: str | Path | None,
    options: Options,
) -> bool:
    input_path = Path(input_path)

    if options.verbose:
        logger.info(f"Reading: {input_path}")

    img = rp.load_tczyx_image(str(input_path))
 
    
    scenes = list(img.scenes)
    n_scenes = len(scenes)
    pad_width = 1 if n_scenes <= 9 else len(str(n_scenes))

    def _with_scene_suffix(path_in: Path, scene_index_1based: int) -> Path:
        # For multi-scene inputs, always append scene suffix before extension.
        if n_scenes <= 1:
            return path_in
        suffix = f"_{scene_index_1based:0{pad_width}d}"
        if path_in.name.lower().endswith(".shifts.txt"):
            stem = path_in.name[:-11]
            return path_in.with_name(f"{stem}{suffix}.shifts.txt")
        if path_in.name.lower().endswith(".ome.tiff"):
            stem = path_in.name[:-9]
            return path_in.with_name(f"{stem}{suffix}.ome.tiff")
        if path_in.name.lower().endswith(".ome.tif"):
            stem = path_in.name[:-8]
            return path_in.with_name(f"{stem}{suffix}.ome.tif")
        return path_in.with_name(f"{path_in.stem}{suffix}{path_in.suffix}")

    for scene_idx, scene_id in enumerate(scenes, start=1):
        img.set_scene(scene_id)
        nt, nc, nz, nh, nw = img.shape

        if options.verbose:
            logger.info(
                f"Scene {scene_idx}/{n_scenes} '{scene_id}': "
                f"T={nt} C={nc} Z={nz} H={nh} W={nw}"
            )

        z_max = options.z_max if options.z_max is not None else nz
        z_min = options.z_min

        if z_min < 1 or z_max > nz:
            raise ValueError(
                f"z_min={z_min} / z_max={z_max} out of range for {nz} z-planes."
            )

        ch = options.channel
        if ch < 0 or ch >= nc:
            raise ValueError(f"Channel {options.channel} out of range (file has {nc} channels).")

        channel_frame_getter = _build_channel_frame_getter(
            img=img,
            channel_index=ch,
            use_lazy_loading=options.use_lazy_loading,
        )
        full_frame_getter = _build_full_frame_getter(
            img=img,
            use_lazy_loading=options.use_lazy_loading,
        )

        max_shifts = (options.max_shift_x, options.max_shift_y, options.max_shift_z)

        if options.shifts_file:
            if options.verbose:
                logger.info(f"Loading shifts from: {options.shifts_file}")
            int_shifts = load_shifts_txt(options.shifts_file)
            if len(int_shifts) != nt:
                raise ValueError(
                    f"Shifts file has {len(int_shifts)} entries but data has {nt} frames."
                )
        else:
            if options.verbose:
                logger.info("Computing drift...")

            if options.edge_enhance:
                logger.warning("--edge-enhance is accepted but not yet implemented; ignored.")

            float_shifts = compute_all_shifts(
                frame_getter=channel_frame_getter,
                nt=nt,
                z_min=z_min,
                z_max=z_max,
                background=options.background,
                correct_only_xy=options.correct_only_xy,
                max_shifts=max_shifts,
                multi_time_scale=options.multi_time_scale,
                verbose=options.verbose,
            )

            correction_shifts = invert_shifts(float_shifts)
            int_shifts = shifts_to_int(correction_shifts)

        if output_path is None:
            base_output_path = input_path.with_name(input_path.stem + "_corrected.ome.tif")
        else:
            base_output_path = Path(output_path)
        scene_output_path = _with_scene_suffix(base_output_path, scene_idx)

        if options.only_compute:
            if options.verbose:
                logger.info(f"Saving shifts to: {scene_output_path}")
            save_shifts_txt(scene_output_path, int_shifts, (nt, nc, nz, nh, nw))
            continue

        if options.verbose:
            logger.info("Applying shifts...")
            xs = [s[0] for s in int_shifts]
            ys = [s[1] for s in int_shifts]
            min_x, min_y = min(xs), min(ys)
            for t, (dx, dy, dz) in enumerate(int_shifts):
                logger.info(f"  Frame {t + 1} correcting drift {dx - min_x},{dy - min_y},{dz}")

        if options.subpixel:
            raise NotImplementedError(
                "Sub-pixel correction is not yet implemented with lazy frame loading. "
                "Run without --subpixel to use integer-pixel correction."
            )

        result = apply_integer_shifts(
            frame_getter=full_frame_getter,
            nt=nt,
            nc=nc,
            nz=nz,
            nh=nh,
            nw=nw,
            dtype=np.asarray(full_frame_getter(0)).dtype,
            shifts=int_shifts,
        )

        if options.verbose:
            logger.info(f"Writing: {scene_output_path}")

        img.set_scene(scene_id)
        physical_pixel_sizes = getattr(img, 'physical_pixel_sizes', None)
        rp.save_tczyx_image(result, str(scene_output_path), physical_pixel_sizes=physical_pixel_sizes)

        mmap_path = getattr(result, "filename", None)
        if mmap_path and isinstance(mmap_path, str) and os.path.exists(mmap_path):
            try:
                os.remove(mmap_path)
            except OSError:
                logger.warning(f"Could not remove temporary file: {mmap_path}")

    if options.verbose:
        logger.info("Done.")

    return True


def process_files(
    input_pattern: str,
    output_folder: Optional[str],
    collapse_delimiter: str,
    output_extension: str,
    no_parallel: bool,
    options: Options,
) -> int:
    search_subfolders = "**" in input_pattern
    files = rp.get_files_to_process2(input_pattern, search_subfolders=search_subfolders)

    if not files:
        logger.error(f"No files found matching pattern: {input_pattern}")
        return 1

    logger.info(f"Found {len(files)} file(s) to process")

    if "**" in input_pattern:
        base_folder = input_pattern.split("**")[0].rstrip("/\\")
        if not base_folder:
            base_folder = "."
        base_folder = os.path.abspath(base_folder)
    else:
        base_folder = str(Path(files[0]).parent)

    if output_folder is None:
        output_folder = base_folder + "_drift_corrected"

    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"Output folder: {output_folder}")

    file_pairs: list[tuple[str, str]] = []
    for src in files:
        collapsed = rp.collapse_filename(src, base_folder, collapse_delimiter)
        src_stem = Path(collapsed).stem
        if options.only_compute:
            out_name = src_stem + output_extension + ".shifts.txt"
        else:
            out_name = src_stem + output_extension + ".ome.tif"
        out_path = os.path.join(output_folder, out_name)
        file_pairs.append((src, out_path))

    failed = 0

    if no_parallel or len(file_pairs) == 1:
        for src, dst in file_pairs:
            try:
                run_single_file(src, dst, options)
            except Exception:
                failed += 1
                logger.exception(f"Failed drift correction for {src}")
    else:
        max_workers = min(os.cpu_count() or 4, len(file_pairs))
        logger.info(f"Processing with {max_workers} workers")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_single_file, src, dst, options): (src, dst)
                for src, dst in file_pairs
            }
            for future in as_completed(futures):
                src, _ = futures[future]
                try:
                    future.result()
                except Exception:
                    failed += 1
                    logger.exception(f"Failed drift correction for {src}")

    if failed > 0:
        logger.error(f"Completed with {failed} failed file(s)")
        return 2

    logger.info("All files processed successfully")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="correct_3D_drift_python",
        description=(
            "Pure-Python port of ImageJ/Fiji Correct 3D Drift plugin. "
            "Reads image files via bioimage pipeline utilities and writes drift-corrected OME-TIFF."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Correct 3D drift for ND2 files (parallel)
  environment: uv@3.11:drift-correction
  commands:
  - python
  - '%REPO%/standard_code/python/correct_3D_drift_python.py'
  - --input-search-pattern: '%YAML%/input/**/*.nd2'
  - --output-folder: '%YAML%/output'
    - --channel: 0
  - --multi-time-scale
  - --log-level: INFO

- name: Correct 3D drift (single worker for large files)
  environment: uv@3.11:drift-correction
  commands:
  - python
  - '%REPO%/standard_code/python/correct_3D_drift_python.py'
  - --input-search-pattern: '%YAML%/input/**/*.ome.tif'
  - --output-folder: '%YAML%/output'
  - --no-parallel
  - --use-lazy-loading
  - --log-level: INFO
        """,
    )

    p.add_argument(
        "--input-search-pattern",
        type=str,
        required=True,
        help="Input file pattern (supports wildcards, use '**' for recursive search)",
    )
    p.add_argument(
        "--output-folder",
        default=None,
        type=str,
        help=(
            "Output folder (default: input_folder + '_drift_corrected'). "
            "When --only-compute is set, shifts files are written there."
        ),
    )

    p.add_argument(
        "--collapse-delimiter",
        type=str,
        default="__",
        help="Delimiter for collapsing subfolder paths (default: '__')",
    )

    p.add_argument(
        "--output-file-name-extension",
        type=str,
        default="",
        help="Additional extension to add before output suffix",
    )

    p.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing (process files sequentially)",
    )

    p.add_argument(
        "--channel",
        type=int,
        default=0,
        metavar="N",
        help="Zero-based channel index for drift detection (0=first channel, 1=second channel)",
    )
    p.add_argument("--correct-only-xy", action="store_true", help="Restrict correction to X/Y.")
    p.add_argument("--multi-time-scale", action="store_true", help="Use dt = 1,3,9,... drift refinement.")
    p.add_argument("--edge-enhance", action="store_true", help="Accepted but not implemented.")
    p.add_argument("--background", type=float, default=0.0, metavar="VALUE", help="Background subtraction.")
    p.add_argument("--z-min", type=int, default=1, metavar="N", help="First z-plane (1-based).")
    p.add_argument("--z-max", type=int, default=None, metavar="N", help="Last z-plane (1-based).")
    p.add_argument("--max-shift-x", type=float, default=10.0, metavar="PX", help="Max drift per step in X.")
    p.add_argument("--max-shift-y", type=float, default=10.0, metavar="PX", help="Max drift per step in Y.")
    p.add_argument("--max-shift-z", type=float, default=10.0, metavar="PX", help="Max drift per step in Z.")

    p.add_argument("--subpixel", action="store_true", help="Raises NotImplementedError for now.")

    p.add_argument("--only-compute", action="store_true", help="Only compute and save shifts.")
    p.add_argument("--use-shifts", default=None, metavar="FILE", help="Use precomputed shifts file.")
    p.add_argument(
        "--use-lazy-loading",
        action="store_true",
        help="Use lazy frame loading for drift estimation and disk-backed output for lower RAM usage",
    )

    p.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING)",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    options = Options(
        channel=args.channel,
        correct_only_xy=args.correct_only_xy,
        multi_time_scale=args.multi_time_scale,
        subpixel=args.subpixel,
        edge_enhance=args.edge_enhance,
        background=args.background,
        z_min=args.z_min,
        z_max=args.z_max,
        max_shift_x=args.max_shift_x,
        max_shift_y=args.max_shift_y,
        max_shift_z=args.max_shift_z,
        only_compute=args.only_compute,
        shifts_file=args.use_shifts,
        use_lazy_loading=args.use_lazy_loading,
        verbose=True,
    )

    try:
        return process_files(
            input_pattern=args.input_search_pattern,
            output_folder=args.output_folder,
            collapse_delimiter=args.collapse_delimiter,
            output_extension=args.output_file_name_extension,
            no_parallel=args.no_parallel,
            options=options,
        )
    except NotImplementedError as exc:
        logger.error(str(exc))
        return 2
    except Exception as exc:
        logger.error(str(exc))
        raise

    return 0


if __name__ == "__main__":
    sys.exit(main())

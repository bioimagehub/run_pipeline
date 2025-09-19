import os
import argparse
import glob
import numpy as np
import csv
from typing import Tuple

from skimage import io
from bioio import BioImage


def _find_pair(folder: str) -> Tuple[str, str]:
    # Match the naming from Simulate_ER_images_script.py: "{i}_IN_snr*.png" and "{i}_GT*.png"
    ins = sorted(glob.glob(os.path.join(folder, "*_IN_snr*.png")))
    gts = sorted(glob.glob(os.path.join(folder, "*_GT*.png")))
    # Build mapping by prefix index before first underscore
    idx_to_in = {}
    for p in ins:
        base = os.path.basename(p)
        idx = base.split("_")[0]
        idx_to_in[idx] = p
    idx_to_gt = {}
    for p in gts:
        base = os.path.basename(p)
        idx = base.split("_")[0]
        idx_to_gt[idx] = p
    matches = []
    for k in sorted(idx_to_in.keys()):
        if k in idx_to_gt:
            matches.append((idx_to_in[k], idx_to_gt[k]))
    if not matches:
        raise FileNotFoundError("No IN/GT pairs found in folder")
    # Return first pair; pipeline likely ran on a single image
    return matches[0]


def _binarize_mask_ome(ome_path: str) -> np.ndarray:
    bio = BioImage(ome_path)
    arr = np.asarray(bio.data)  # TCZYX
    if arr.ndim != 5:
        raise ValueError(f"Expected 5D TCZYX mask, got {arr.shape}")
    # C should be 1, collapse to ZYX by taking T=0, C=0 and max over Z
    bin_zyx = (arr[0, 0] > 0).astype(np.uint8)  # Z,Y,X
    return bin_zyx


def _resolve_path_maybe_glob(p: str) -> str:
    if any(ch in p for ch in "*?["):
        matches = sorted(glob.glob(p))
        if not matches:
            raise FileNotFoundError(f"No files matched pattern: {p}")
        return matches[0]
    return p


def _binarize_mask_png(png_path: str, mask_class: str) -> np.ndarray:
    arr = io.imread(png_path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D class-coded PNG, got {arr.shape}")
    if mask_class == "any":
        return (arr != 0).astype(np.uint8)
    mapping = {"tubule": 255, "sheet": 85, "sbt": 170}
    if mask_class not in mapping:
        raise ValueError(f"Unknown mask class: {mask_class}")
    return (arr == mapping[mask_class]).astype(np.uint8)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = np.sum((a & b) > 0)
    s = a.sum() + b.sum()
    return 2.0 * inter / s if s > 0 else 1.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare ERnet OME mask to synthetic ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--synthetic-folder", required=True, help="Folder containing *_IN_snr*.png and *_GT*.png")
    parser.add_argument("--ernet-mask", required=True, help="Path or glob to ERnet output (OME-TIFF mask or class-coded PNG)")
    parser.add_argument("--mask-class", choices=["any", "tubule", "sheet", "sbt"], default="any", help="Class to binarize if PNG output is provided")
    parser.add_argument("--out-csv", default=None, help="Optional CSV path for metrics; defaults next to mask")
    args = parser.parse_args(argv)

    in_path, gt_path = _find_pair(args.synthetic_folder)
    gt = io.imread(gt_path)
    if gt.ndim != 2:
        raise ValueError(f"Expected 2D GT image, got {gt.shape}")

    # Normalize GT to binary: generator wrote img_as_ubyte(I1), where foreground is bright
    gt_bin = (gt > 127).astype(np.uint8)

    mask_path = _resolve_path_maybe_glob(args.ernet_mask)
    if mask_path.lower().endswith(('.tif', '.tiff')):
        mask_zyx = _binarize_mask_ome(mask_path)
        if mask_zyx.ndim != 3:
            raise ValueError(f"Expected ZYX mask, got {mask_zyx.shape}")
        mask_2d = (mask_zyx.max(axis=0) > 0).astype(np.uint8)
    else:
        mask_2d = _binarize_mask_png(mask_path, args.mask_class)

    # Align sizes if tiling introduced border differences
    h = min(gt_bin.shape[0], mask_2d.shape[0])
    w = min(gt_bin.shape[1], mask_2d.shape[1])
    gt_c = gt_bin[:h, :w]
    mk_c = mask_2d[:h, :w]

    d = _dice(gt_c, mk_c)
    iou = (np.logical_and(gt_c, mk_c).sum()) / (np.logical_or(gt_c, mk_c).sum() + 1e-9)

    out_csv = args.out_csv or os.path.splitext(mask_path)[0] + "_compare.csv"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    f = open(out_csv, "w", newline="")
    try:
        wri = csv.writer(f)
        wri.writerow(["in_png", "gt_png", "mask", "dice", "iou", "height", "width"])
        wri.writerow([os.path.basename(in_path), os.path.basename(gt_path), os.path.basename(mask_path), f"{d:.4f}", f"{iou:.4f}", h, w])
    finally:
        f.close()
    print(f"Wrote comparison: {out_csv} (Dice={d:.4f}, IoU={iou:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

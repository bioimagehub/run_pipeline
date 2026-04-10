from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import tifffile


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_tiff_outputs.py <left.tif> <right.tif>")

    left_path = Path(sys.argv[1])
    right_path = Path(sys.argv[2])
    left = tifffile.imread(left_path)
    right = tifffile.imread(right_path)

    result = {
        "left": str(left_path),
        "right": str(right_path),
        "shape_equal": left.shape == right.shape,
        "dtype_equal": str(left.dtype) == str(right.dtype),
        "equal": bool(np.array_equal(left, right)),
        "left_sum": int(left.astype(np.int64).sum()),
        "right_sum": int(right.astype(np.int64).sum()),
        "max_abs_diff": int(np.max(np.abs(left.astype(np.int64) - right.astype(np.int64)))),
    }

    output_path = left_path.with_name(left_path.stem + "_comparison.json")
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
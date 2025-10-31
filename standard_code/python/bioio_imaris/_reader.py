from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Iterable

import numpy as np
import dask.array as da

try:
    from bioio_base import Reader as BaseReader  # type: ignore
    from bioio_base import types, constants  # type: ignore
except Exception as e:  # fallback if bioio_base not exposed directly
    # bioio exposes base under bioio.types/constants in some versions
    from bioio import types, constants  # type: ignore
    from bioio.readers.reader import Reader as BaseReader  # type: ignore


class Reader(BaseReader):
    """Pure-Python Imaris .ims reader for bioio (TCZYX)."""

    _ims: Any

    @classmethod
    def supports_extension(cls, ext: str) -> bool:
        return ext.lower() == ".ims"

    def _read_delayed(self) -> types.ArrayLike:
        # Import lazily to keep dependency optional
        from imaris_ims_file_reader.ims import ims as ImarisIMS  # type: ignore
        # The library exposes a 5D array interface with axes (T, C, Z, Y, X) at highest resolution (level 0)   
        self._ims = ImarisIMS(self._image)
        # Build a Dask view to keep everything lazy / chunked
        arr = da.from_array(self._ims, chunks=getattr(self._ims, "chunks", None))
        # Ensure dtype is a NumPy dtype
        return arr.astype(self._ims.dtype, copy=False)

    def _get_shape(self) -> Tuple[int, int, int, int, int]:
        shp = getattr(self._ims, "shape", None)
        if shp is None:
            # Some versions expose dims via properties
            t = int(getattr(self._ims, "TimePoints", 1))
            c = int(getattr(self._ims, "Channels", [None]).__len__())
            z = int(getattr(self._ims, "SizeZ", 1))
            y = int(getattr(self._ims, "SizeY", 1))
            x = int(getattr(self._ims, "SizeX", 1))
            return (t, c, z, y, x)
        return tuple(int(x) for x in shp)

    def _get_dtype(self) -> np.dtype:
        return np.dtype(getattr(self._ims, "dtype", np.uint16))

    def _get_channel_names(self) -> Optional[Iterable[str]]:
        try:
            chans = getattr(self._ims, "Channels", None)
            if chans is None:
                return None
            names = []
            for idx, ch in enumerate(chans):
                name = None
                try:
                    name = ch.get("Name", None)
                except Exception:
                    pass
                names.append(name if name else f"C{idx}")
            return names
        except Exception:
            return None

    def _get_physical_pixel_sizes(self) -> types.PhysicalPixelSizes:
        # Try to pull voxel size from metadata; fall back to 1.0 µm
        try:
            params = getattr(self._ims, "Parameters", {})
            ext = None
            if isinstance(params, dict):
                ext = params.get("Extents", {}).get("Spacing", None)
            if ext is None:
                # alternate key shapes sometimes occur
                ext = params.get("Spacing", None) if isinstance(params, dict) else None
            if ext is not None and len(ext) >= 3:
                return types.PhysicalPixelSizes(z=float(ext[0]), y=float(ext[1]), x=float(ext[2]))
        except Exception:
            pass
        return types.PhysicalPixelSizes(1.0, 1.0, 1.0)

    def _get_dims(self) -> str:
        return constants.Dimensions.DefaultOrder  # "TCZYX"

    def _get_metadata(self) -> Dict[str, Any]:
        md: Dict[str, Any] = {}
        try:
            params = getattr(self._ims, "Parameters", None)
            if params is not None:
                try:
                    md["imaris_parameters"] = dict(params)
                except Exception:
                    md["imaris_parameters"] = params
            rl = getattr(self._ims, "ResolutionLevels", None)
            if rl is not None:
                md["resolution_levels"] = int(rl)
            tp = getattr(self._ims, "TimePoints", None)
            if tp is not None:
                md["time_points"] = int(tp)
        except Exception:
            pass
        return md

from __future__ import annotations
from typing import Any, Optional, Tuple, Iterable, List

import h5py
import numpy as np
import dask.array as da

from bioio_base.reader import Reader as BaseReader
from bioio_base import types
from bioio_base import dimensions


class IlastikH5Reader(BaseReader):
    """
    Bioio reader for Ilastik HDF5 files saved in TZYXC format.
    Converts TCZYX to TCZYX for compatibility with bioio standard.
    """
    
    _cached_data: Optional[np.ndarray] = None
    _cached_axes: Optional[list] = None
    _h5_path: Optional[str] = None

    def __init__(self, image: Any, **kwargs: Any):
        """Initialize the reader with the image path."""
        # Store path before calling super().__init__()
        if isinstance(image, (str, bytes)):
            self._h5_path = str(image)
        super().__init__(image, **kwargs)

    @classmethod
    def _is_supported_image(cls, image: Any, **kwargs: Any) -> bool:
        """Check if this is a valid Ilastik H5 file."""
        try:
            if isinstance(image, (str, bytes)):
                with h5py.File(image, 'r') as f:
                    return 'exported_data' in f
            return False
        except Exception:
            return False

    @property
    def scenes(self) -> Tuple[str, ...]:
        """Return available scenes (always single scene for H5)."""
        return ("Image",)
    
    @classmethod
    def supports_extension(cls, ext: str) -> bool:
        return ext.lower() in (".h5", ".hdf5")

    def _read_delayed(self) -> types.ArrayLike:
        """Load data lazily using Dask."""
        import xarray as xr
        
        img_path = getattr(self, '_image', None) or self._h5_path
        with h5py.File(img_path, 'r') as f:
            data = f['exported_data'][:]
            axes = self._get_axes(f)
            
        # Convert TZYXC to TCZYX for BioImage
        axis_map = {ax: i for i, ax in enumerate(axes)}
        perm = [axis_map[a] for a in 'TCZYX']
        tczyx_data = np.transpose(data, perm)
        
        # Cache for other methods
        self._cached_data = tczyx_data
        self._cached_axes = ['T', 'C', 'Z', 'Y', 'X']
        
        # Return as xarray.DataArray with dimension labels
        # BaseReader will handle converting to dask if needed
        xarr = xr.DataArray(
            tczyx_data,  # Use numpy array, not dask
            dims=['T', 'C', 'Z', 'Y', 'X'],
            coords={
                'T': np.arange(tczyx_data.shape[0]),
                'C': np.arange(tczyx_data.shape[1]),
                'Z': np.arange(tczyx_data.shape[2]),
                'Y': np.arange(tczyx_data.shape[3]),
                'X': np.arange(tczyx_data.shape[4]),
            }
        )
        
        return xarr

    def _read_immediate(self) -> types.ArrayLike:
        """Load data immediately (same as delayed for H5)."""
        return self._read_delayed()

    def _get_axes(self, f: h5py.File) -> list[str]:
        """Extract axis order from HDF5 file attributes."""
        if 'axistags' in f['exported_data'].attrs:
            import json
            axistags = f['exported_data'].attrs['axistags']
            if isinstance(axistags, bytes):
                axistags = axistags.decode('utf-8')
            axes_json = json.loads(axistags)
            return [a['key'].upper() for a in axes_json['axes']]
        return ['T', 'Z', 'Y', 'X', 'C']

    def _get_shape(self) -> Tuple[int, int, int, int, int]:
        """Return shape in TCZYX order."""
        if self._cached_data is not None:
            return tuple(self._cached_data.shape)
        
        # Read temporarily to get shape
        img_path = getattr(self, '_image', None) or self._h5_path
        with h5py.File(img_path, 'r') as f:
            data_shape = f['exported_data'].shape
            axes = self._get_axes(f)
        
        # Convert to TCZYX order
        axis_map = {ax: i for i, ax in enumerate(axes)}
        return tuple(data_shape[axis_map[a]] for a in 'TCZYX')

    def _get_dtype(self) -> np.dtype:
        """Return data type."""
        img_path = getattr(self, '_image', None) or self._h5_path
        with h5py.File(img_path, 'r') as f:
            return np.dtype(f['exported_data'].dtype)

    def _get_dims(self) -> str:
        """Return dimension order string."""
        return dimensions.DEFAULT_DIMENSION_ORDER  # "TCZYX"

    def _get_physical_pixel_sizes(self) -> types.PhysicalPixelSizes:
        """Return pixel sizes (default to 1.0 if not available)."""
        return types.PhysicalPixelSizes(Z=1.0, Y=1.0, X=1.0)

    def _get_channel_names(self) -> Optional[Iterable[str]]:
        """Return channel names if available."""
        img_path = getattr(self, '_image', None) or self._h5_path
        with h5py.File(img_path, 'r') as f:
            if 'channel_names' in f.attrs:
                return list(f.attrs['channel_names'])
        return None

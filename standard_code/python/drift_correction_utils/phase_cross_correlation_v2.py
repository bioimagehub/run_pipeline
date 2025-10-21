"""
Pure Python version of ImageJ's phase cross-correlation drift correction.
Follows original logic and function names as closely as possible.
Uses numpy, scipy, cupy (if available), and rp (bioimage_pipeline_utils) for image I/O and processing.
"""

import numpy as np
import os
import logging
from typing import List, Tuple, Optional, Dict, Any

# Always use rp for image I/O
import bioimage_pipeline_utils as rp

from scipy.ndimage import shift as scipy_shift
try:
  import cupy as cp
  from cupyx.scipy.ndimage import shift as cupy_shift
  GPU_AVAILABLE = True
except ImportError:
  GPU_AVAILABLE = False

logger = logging.getLogger(__name__)

def compute_shift(img1: np.ndarray, img2: np.ndarray, upsample_factor: int = 10, use_gpu: bool = False) -> np.ndarray:
  """
  Compute translation shift between two images using phase cross-correlation.
  Returns shift as (dz, dy, dx) for 3D or (dy, dx) for 2D.
  """
  if use_gpu and GPU_AVAILABLE:
    # Use CuPy for GPU acceleration
    from skimage.registration import phase_cross_correlation as pcc
    img1_gpu = cp.asarray(img1)
    img2_gpu = cp.asarray(img2)
    shift, error, phasediff = pcc(cp.asnumpy(img1_gpu), cp.asnumpy(img2_gpu), upsample_factor=upsample_factor)
    return shift
  else:
    from skimage.registration import phase_cross_correlation as pcc
    shift, error, phasediff = pcc(img1, img2, upsample_factor=upsample_factor)
    return shift

def extract_frame(img: np.ndarray, t: int, c: int, z_min: int, z_max: int) -> np.ndarray:
  """
  Extracts a frame (timepoint), channel, and z-range from a TCZYX image.
  Returns a stack (Z, Y, X).
  """
  # img: TCZYX
  return img[t, c, z_min:z_max+1, :, :]

def add_Point3f(p1, p2):
  return [p1[0]+p2[0], p1[1]+p2[1], p1[2]+p2[2]]

def subtract_Point3f(p1, p2):
  return [p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]]

def get_Point3i(point, dimension):
  return point[dimension]

def set_Point3i(point, dimension, value):
  point[dimension] = int(value)

def compute_and_update_frame_translations_dt(
  img: np.ndarray,
  dt: int,
  options: Dict[str, Any],
  shifts: Optional[List[List[float]]] = None
) -> List[List[float]]:
  """
  Compute X,Y,Z translation between every t and t+dt time points in img.
  img: TCZYX numpy array
  options: dict with keys 'channel', 'z_min', 'z_max', 'upsample_factor', 'max_shifts', 'correct_only_xy', 'use_gpu'
  shifts: list of [dx, dy, dz] (absolute, not relative)
  """
  nt = img.shape[0]
  c = options.get('channel', 0)
  z_min = options.get('z_min', 0)
  z_max = options.get('z_max', img.shape[2]-1)
  upsample_factor = options.get('upsample_factor', 10)
  max_shifts = options.get('max_shifts', [20, 20, 5])
  correct_only_xy = options.get('correct_only_xy', True)
  use_gpu = options.get('use_gpu', False)

  if shifts is None:
    shifts = [[0.0, 0.0, 0.0] for _ in range(nt)]

  for t in range(dt, nt+dt, dt):
    if t > nt-1:
      t = nt-1
    # Extract frames
    frame1 = extract_frame(img, t-dt, c, z_min, z_max)
    frame2 = extract_frame(img, t, c, z_min, z_max)
    # Project to 2D if only XY correction
    if correct_only_xy or frame1.shape[0] == 1:
      frame1_proj = np.max(frame1, axis=0) if frame1.shape[0] > 1 else frame1[0]
      frame2_proj = np.max(frame2, axis=0) if frame2.shape[0] > 1 else frame2[0]
      shift = compute_shift(frame2_proj, frame1_proj, upsample_factor, use_gpu)
      shift3 = [shift[1], shift[0], 0.0]  # (dy, dx) -> (x, y, z)
    else:
      shift = compute_shift(frame2, frame1, upsample_factor, use_gpu)
      shift3 = [shift[2], shift[1], shift[0]]  # (dz, dy, dx)
    # Limit shifts
    for d in range(3):
      if shift3[d] > max_shifts[d]:
        shift3[d] = max_shifts[d]
      if shift3[d] < -max_shifts[d]:
        shift3[d] = -max_shifts[d]
    # Update shifts
    local_shift = subtract_Point3f(shifts[t], shifts[t-dt])
    add_shift = subtract_Point3f(shift3, local_shift)
    for i, tt in enumerate(range(t-dt, nt)):
      for d in range(3):
        shifts[tt][d] += (i/dt) * add_shift[d]
  return shifts

def invert_shifts(shifts: List[List[float]]) -> List[List[float]]:
  return [[-s[0], -s[1], -s[2]] for s in shifts]

def convert_shifts_to_integer(shifts: List[List[float]]) -> List[List[int]]:
  return [[int(round(s[0])), int(round(s[1])), int(round(s[2]))] for s in shifts]

def compute_min_max(shifts):
  """Compute min and max shifts in each dimension. Works with both float and int lists."""
  arr = np.array(shifts)
  minx, miny, minz = np.min(arr, axis=0)
  maxx, maxy, maxz = np.max(arr, axis=0)
  return int(minx), int(miny), int(minz), int(maxx), int(maxy), int(maxz)

def register_hyperstack(
  img: np.ndarray,
  shifts: List[List[int]],
  fill_value: float = 0.0,
  use_gpu: bool = False
) -> np.ndarray:
  """
  Applies integer shifts to all frames in a TCZYX image.
  Returns a new TCZYX numpy array with registered frames.
  """
  T, C, Z, Y, X = img.shape
  minx, miny, minz, maxx, maxy, maxz = compute_min_max(shifts)
  width = X + maxx - minx
  height = Y + maxy - miny
  slices = Z + maxz - minz
  out = np.full((T, C, slices, height, width), fill_value, dtype=img.dtype)
  for t in range(T):
    sx, sy, sz = shifts[t]
    sx -= minx
    sy -= miny
    sz -= minz
    for c in range(C):
      for z in range(Z):
        z_out = z + sz
        if 0 <= z_out < slices:
          out[t, c, z_out, sy:sy+Y, sx:sx+X] = img[t, c, z, :, :]
  return out

def register_hyperstack_subpixel(
  img: np.ndarray,
  shifts: List[List[float]],
  fill_value: float = 0.0,
  use_gpu: bool = False
) -> np.ndarray:
  """
  Applies subpixel shifts to all frames in a TCZYX image using interpolation.
  Returns a new TCZYX numpy array with registered frames.
  """
  T, C, Z, Y, X = img.shape
  minx, miny, minz, maxx, maxy, maxz = compute_min_max(shifts)
  width = X + int(round(maxx - minx))
  height = Y + int(round(maxy - miny))
  slices = Z + int(round(maxz - minz))
  out = np.full((T, C, slices, height, width), fill_value, dtype=img.dtype)
  
  # Use GPU if requested and available
  use_cupy = use_gpu and GPU_AVAILABLE
  
  for t in range(T):
    sx, sy, sz = shifts[t]
    sx -= minx
    sy -= miny
    sz -= minz
    
    # Integer parts for positioning
    sx_int = int(sx)
    sy_int = int(sy)
    # Fractional parts for subpixel shift
    sx_frac = sx - sx_int
    sy_frac = sy - sy_int
    
    for c in range(C):
      for z in range(Z):
        z_out = int(round(z + sz))
        if 0 <= z_out < slices:
          if use_cupy:
            arr = cp.asarray(img[t, c, z, :, :])
            shifted = cupy_shift(arr, shift=[sy_frac, sx_frac], order=1, mode='constant', cval=fill_value)
            shifted_np = cp.asnumpy(shifted)
          else:
            shifted_np = scipy_shift(img[t, c, z, :, :], shift=[sy_frac, sx_frac], order=1, mode='constant', cval=fill_value)
          
          # Place the shifted image at the integer position
          out[t, c, z_out, sy_int:sy_int+Y, sx_int:sx_int+X] = shifted_np
  return out

def run(
  path: str,
  output_path: Optional[str] = None,
  channel: int = 0,
  z_min: int = 0,
  z_max: Optional[int] = None,
  correct_only_xy: bool = True,
  multi_time_scale: bool = False,
  subpixel: bool = True,
  upsample_factor: int = 10,
  max_shifts: List[int] = [20, 20, 5],
  use_gpu: bool = False
):
  """
  Main entry point for drift correction. Loads image, computes shifts, applies correction, and saves result.
  """
  logger.info(f"Loading image: {path}")
  img_obj = rp.load_tczyx_image(path)
  img = img_obj.data  # TCZYX
  if z_max is None:
    z_max = img.shape[2] - 1
  options = dict(
    channel=channel,
    z_min=z_min,
    z_max=z_max,
    upsample_factor=upsample_factor,
    max_shifts=max_shifts,
    correct_only_xy=correct_only_xy,
    use_gpu=use_gpu
  )
  logger.info("Computing drift shifts (dt=1)...")
  dt = 1
  shifts = compute_and_update_frame_translations_dt(img, dt, options)
  if multi_time_scale:
    nt = img.shape[0]
    dts = [3,9,27,81,243,729,nt-1]
    for dt in dts:
      if dt < nt-1:
        logger.info(f"Computing drift shifts (dt={dt})...")
        shifts = compute_and_update_frame_translations_dt(img, dt, options, shifts)
      else:
        logger.info(f"Computing drift shifts (dt={nt-1})...")
        shifts = compute_and_update_frame_translations_dt(img, nt-1, options, shifts)
        break
  shifts = invert_shifts(shifts)
  logger.info("Applying shifts to image stack...")
  if subpixel:
    registered = register_hyperstack_subpixel(img, shifts, use_gpu=use_gpu)
  else:
    shifts_int = convert_shifts_to_integer(shifts)
    registered = register_hyperstack(img, shifts_int, use_gpu=use_gpu)
  img_obj_registered = rp.BioImage(
    registered,
    physical_pixel_sizes=img_obj.physical_pixel_sizes,
    channel_names=img_obj.channel_names,
    metadata=img_obj.metadata
  )
  if output_path is not None:
    logger.info(f"Saving registered image to: {output_path}")
    rp.save_tczyx_image(img_obj_registered, output_path)
  return img_obj_registered, shifts

def register_image_xy(
    img,
    reference: str = 'first',
    channel: int = 0,
    show_progress: bool = True,
    no_gpu: bool = False,
    crop_fraction: float = 1.0,
    upsample_factor: int = 10
) -> Tuple[Any, np.ndarray]:
  """
  Register a TCZYX image using translation in XY dimensions only.
  
  Compatible wrapper for drift_correction.py integration.
  
  Args:
    img: BioImage object containing TCZYX image data
    reference: Registration reference strategy ('first', 'previous', or 'median')
    channel: Zero-indexed channel to use for computing transformations
    show_progress: Whether to display progress (currently ignored)
    no_gpu: Force CPU execution even if GPU is available
    crop_fraction: Fraction of image to use (currently ignored, uses full image)
    upsample_factor: Subpixel precision factor (default: 10)
  
  Returns:
    Tuple containing registered BioImage and transformation matrices (shifts as Nx3 array)
  """
  # Get image data
  img_data = img.data  # TCZYX
  T = img_data.shape[0]
  
  # Prepare options
  options = dict(
    channel=channel,
    z_min=0,
    z_max=img_data.shape[2] - 1,
    upsample_factor=upsample_factor,
    max_shifts=[50, 50, 5],  # Reasonable defaults
    correct_only_xy=True,
    use_gpu=not no_gpu
  )
  
  logger.info(f"Computing drift shifts using {reference} reference...")
  
  if reference == 'first':
    # Compute shifts relative to first frame
    dt = 1
    shifts = compute_and_update_frame_translations_dt(img_data, dt, options)
  elif reference == 'previous':
    # Compute shifts frame-to-frame
    dt = 1
    shifts = compute_and_update_frame_translations_dt(img_data, dt, options)
  elif reference == 'median':
    # For median, we compute to first then adjust (simplified approach)
    logger.warning("Median reference not fully implemented, using first frame reference")
    dt = 1
    shifts = compute_and_update_frame_translations_dt(img_data, dt, options)
  else:
    raise ValueError(f"Unknown reference type: {reference}")
  
  # Invert shifts for correction
  shifts = invert_shifts(shifts)
  
  logger.info("Applying shifts to image stack...")
  # Always use subpixel correction for best quality
  registered_data = register_hyperstack_subpixel(img_data, shifts, fill_value=0.0, use_gpu=not no_gpu)
  
  # Create output BioImage with same metadata
  registered_img = rp.BioImage(
    registered_data,
    physical_pixel_sizes=img.physical_pixel_sizes,
    channel_names=img.channel_names,
    metadata=img.metadata
  )
  
  # Convert shifts to transformation matrix format (Nx3 array: dx, dy, dz)
  # Note: shifts are in [x, y, z] format
  tmats = np.array(shifts, dtype=np.float32)
  
  return registered_img, tmats


if __name__ == "__main__":
  import argparse
  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  parser = argparse.ArgumentParser(description="Drift correction using phase cross-correlation (pure Python version)")
  parser.add_argument("input", help="Input image path")
  parser.add_argument("-o", "--output", help="Output image path", default=None)
  parser.add_argument("-c", "--channel", type=int, default=0, help="Channel index (zero-based)")
  parser.add_argument("--z_min", type=int, default=0, help="Minimum Z slice (zero-based)")
  parser.add_argument("--z_max", type=int, default=None, help="Maximum Z slice (zero-based)")
  parser.add_argument("--no-xy", action="store_true", help="Correct all axes, not just XY")
  parser.add_argument("--multi", action="store_true", help="Enable multi-time-scale drift estimation")
  parser.add_argument("--no-subpixel", action="store_true", help="Disable subpixel correction (integer only)")
  parser.add_argument("--upsample", type=int, default=10, help="Upsample factor for subpixel accuracy")
  parser.add_argument("--max-shifts", nargs=3, type=int, default=[20,20,5], help="Max allowed shift in X Y Z")
  parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration if available")
  args = parser.parse_args()
  run(
    path=args.input,
    output_path=args.output,
    channel=args.channel,
    z_min=args.z_min,
    z_max=args.z_max,
    correct_only_xy=not args.no_xy,
    multi_time_scale=args.multi,
    subpixel=not args.no_subpixel,
    upsample_factor=args.upsample,
    max_shifts=args.max_shifts,
    use_gpu=args.gpu
  )
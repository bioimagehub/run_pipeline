import numpy as np
import shutil
import zarr
from skimage.registration import phase_cross_correlation
import argparse
import os
from tqdm import tqdm
from bioio import BioImage
import bioio_bioformats


"""
Modified from Dominik Schienstock
by Oeyvind Oedegaard Fougner 20250411

"""

class DimUtils:
    def __init__(self, bioio_dims):
        """Initialize the DimUtils class with a Dimensions object."""
        # Store the shape and order of dimensions
        self.shape = (
            bioio_dims.T,
            bioio_dims.C,
            bioio_dims.Z,
            bioio_dims.Y,
            bioio_dims.X,
        )
        self.order = bioio_dims.order  # e.g., "TCZYX"
        
        # Creating a mapping of dimension names to their respective indices
        self.dim_map = {dim: idx for idx, dim in enumerate(self.order)}

    def dim_idx(self, dim_name):
        """Return the index for the provided dimension name."""
        return self.dim_map.get(dim_name, None)
    
    def dim_val(self, dim_name):
        """Return the size of the provided dimension name."""
        idx = self.dim_idx(dim_name)
        if idx is not None:
            return self.shape[idx]
        raise ValueError(f"Dimension '{dim_name}' not found in the dimension mapping.")


"""
Get drift correction shift
"""
def drift_correction_shifts(
  image_array, phase_shift_channel, dim_utils,
  timepoints = None, upsample_factor = 100):
  # get shifts
  shifts = list()

  # get image dimension information
  channel_idx = dim_utils.dim_idx('C')
  time_idx = dim_utils.dim_idx('T')

  # create slices for corrections
  slices = [slice(None) for x in range(len(image_array.shape))]
  slices[channel_idx] = slice(phase_shift_channel, phase_shift_channel + 1, 1)

  # get all timepoints if not specified
  if timepoints is None:
    timepoints = range(1, dim_utils.dim_val('T'))

  # go through timepoints
  for x in timepoints:
    if x % 10 == 0:
      print(x)

    # define timepoints
    slices_a = slices.copy()
    slices_b = slices.copy()
    slices_a[time_idx] = slice(x - 1, x, 1)
    slices_b[time_idx] = slice(x, x + 1, 1)

    # convert to tuple
    slices_a = tuple(slices_a)
    slices_b = tuple(slices_b)

    # (sub)pixel precision
    shift, error, diffphase = phase_cross_correlation(
      np.squeeze(image_array[slices_a]),
      np.squeeze(image_array[slices_b]),
      upsample_factor = upsample_factor
    )

    # add shift
    shifts.append(shift)

  # convert to array
  return np.vstack(shifts)

"""
get max, min and sum shift
"""
def drift_shifts_summary(shifts):
  max_shifts = np.zeros((3))
  min_shifts = np.zeros((3))
  cur_shifts = np.zeros((3))

  # get maximum shifts for left and right
  for x in shifts:
    # get new shifts
    cur_shifts = cur_shifts + x

    # set max and min shifts
    max_shifts = np.maximum(cur_shifts, max_shifts)
    min_shifts = np.minimum(cur_shifts, min_shifts)

  min_shifts = abs(min_shifts)
  sum_shifts = max_shifts + min_shifts

  return {
    'max': max_shifts,
    'min': min_shifts,
    'sum': sum_shifts
    }

"""
get new image dimensions for drift correction
"""
def drift_correction_im_shape(image_array, dim_utils, shifts_summary):
  # get new shape
  new_shape = list(image_array.shape)

  # assuming shifts are Z, Y, X
  new_shape[dim_utils.dim_idx('Z')] += abs(shifts_summary['sum'][0])
  new_shape[dim_utils.dim_idx('Y')] += abs(shifts_summary['sum'][1])
  new_shape[dim_utils.dim_idx('X')] += abs(shifts_summary['sum'][2])

  # round new shape for new array
  new_shape_round = tuple([
    round(x) for x in new_shape
  ])

  return new_shape, new_shape_round

"""
drift correct image
"""
def drift_correct_im(
  input_array, dim_utils, phase_shift_channel,
  timepoints = None, drift_corrected_path = None,
  upsample_factor = 100, shifts = None
  ):
  # get all timepoints if not specified
  if timepoints is None:
    timepoints = range(dim_utils.dim_val('T'))

  # get shifts
  if shifts is None:
    print('>> Get shifts')

    shifts = drift_correction_shifts(
      input_array, phase_shift_channel, dim_utils,
      # check that the first timepoint is not used
      timepoints = timepoints[1:],
      upsample_factor = upsample_factor
      )

  # get shifts summary
  shifts_summary = drift_shifts_summary(shifts)

  # get new image dimensions
  drift_im_shape, drift_im_shape_round = drift_correction_im_shape(
    input_array, dim_utils, shifts_summary
  )

  # get first image position
  first_im_pos = drift_correction_first_im_pos(
    drift_im_shape, dim_utils, shifts_summary
  )

  # use new shape for chunking
  # TODO !! this assumes smaller images as we usually have for 2P
  chunk_size = list(input_array.chunksize)
  for x in ('Y', 'X'):
    chunk_size[dim_utils.dim_idx(x)] = drift_im_shape_round[dim_utils.dim_idx(x)]
  chunk_size = tuple(chunk_size)

  # remove previous folder
  if drift_corrected_path is not None:
    shutil.rmtree(drift_corrected_path)

  # create array
  drift_correction_zarr = zarr.create(
    drift_im_shape_round,
    dtype = input_array.dtype,
    chunks = chunk_size,
    store = drift_corrected_path
  )

  # use first position for slice
  slices = first_im_pos

  # get timepoint shape
  tp_slice = [slice(None) for x in range(len(drift_im_shape_round))]
  tp_slice[dim_utils.dim_idx('T')] = slice(0, 1, 1)
  tp_slice = tuple(tp_slice)
  tp_shape = drift_correction_zarr[tp_slice].shape

  print('>> Apply shifts')

  # go through timepoints and add images
  for i in timepoints:
    # create slice
    if i > 0:
      new_slices = list()

      # adjust slices
      for j, y in enumerate(slices):
        new_slices.append(slice(
          # subtract '1' because there is no
          # shift for the first frame
          y.start + shifts[i - 1, j],
          y.stop + shifts[i - 1, j],
          1
        ))

      # push back
      slices = new_slices

    # round for slicing
    new_slices = [slice(None) for x in range(len(drift_im_shape_round))]
    im_slices = [slice(None) for x in range(len(drift_im_shape_round))]

    # set Z, X, Y for new slices
    for j, y in enumerate(('Z', 'Y', 'X')):
      new_slices[dim_utils.dim_idx(y)] =  slice(round(slices[j].start), round(slices[j].stop), 1)

    # set time for image slice
    im_slices[dim_utils.dim_idx('T')] = slice(i, i + 1, 1)

    # convert to tuple
    new_slices = tuple(new_slices)
    im_slices = tuple(im_slices)

    if i % 10 == 0:
      print(i)

    # add to image list
    new_image = np.zeros(tp_shape)

    # check that slices match dimension
    if new_image[new_slices].shape != input_array[im_slices].shape:
      # get wrong dimensions
      dif_dim = [x - y for x, y in zip(
        new_image[new_slices].shape,
        input_array[im_slices].shape
      )]

      # adjust dimensions
      new_slices = list(new_slices)

      for j, y in enumerate(dif_dim):
        if y > 0:
          # add?
          new_slices[j] = slice(
              new_slices[j].start + y,
              new_slices[j].stop, 1)

        if y < 0:
          # add?
          if new_slices[j].start - y >= 0:
            new_slices[j] = slice(
              new_slices[j].start + y,
              new_slices[j].stop, 1)

          # subtract?
          elif new_slices[j].stop + y < drift_correction_zarr.shape[j]:
            new_slices[j] = slice(
              new_slices[j].start,
              new_slices[j].stop + y, 1)

      new_slices = tuple(new_slices)

    # copy to 'zero' image
    new_image[new_slices] = input_array[im_slices]

    # push to zarr
    drift_correction_zarr[im_slices] = new_image

  # return
  return drift_correction_zarr

"""
get position of first image for drift correction
"""
def drift_correction_first_im_pos(drift_im_shape, dim_utils, shifts_summary):
  # get new position
  new_pos = np.take(
    drift_im_shape,
    [dim_utils.dim_idx('Z'), dim_utils.dim_idx('Y'), dim_utils.dim_idx('X')]
    )

  # place the first image
  first_pos = tuple(
    [slice(shifts_summary['min'][i],
           new_pos[i] - shifts_summary['max'][i],
           1) for i in range(3)]
  )

  return first_pos



# TODO
# I need to merge the code below




def process_file(args):
        file_path, output_path = args.input_file, args.output_file
        print(file_path, output_path)

        # Open image
        img = BioImage(file_path, reader=bioio_bioformats.Reader)
        print(img.shape)  # Should output (T, C, Z, Y, X)
        

    
        # Get the image dimensions and data
        dim_utils = DimUtils(img.dims)  # Dimensions object
        img_data = img.data  # returns 5D TCZYX dask array

        
        # TODO alow argument for the phase shift channel default is 0
        phase_shift_channel = 0  # Assuming the first channel is the one to correct
        drift_corrected_image = drift_correct_im(
            img_data,
            dim_utils,
            phase_shift_channel,
            upsample_factor=5,  # TODO allow argument
            drift_corrected_path=output_path,
        )
        crasr
        
 
def collapse_filename(file_path, base_folder, delimiter):
    """
    Collapse the file path into a single filename, preserving directory info.
    """
    rel_path = os.path.relpath(file_path, start=base_folder)
    collapsed = delimiter.join(rel_path.split(os.sep))
    return collapsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, help="Path to the file to be processed")
    parser.add_argument("-o", "--output_file", type=str, help="Path for the output tif file")

    args = parser.parse_args()

    # Ensure the user provides a valid folder path
    if args.input_file is None or not os.path.isfile(args.folder_path):
        args.input_file = r"C:\Users\oodegard\Desktop\collection_of_different_file_formats\input_nd2\1.nd2"
        args.output_file = r"C:\Users\oodegard\Desktop\collection_of_different_file_formats\input_nd2\1_drift_corrected.tif"
        

    process_file(args)



if __name__ == "__main__":
    main()


# Image Drift Correction and TIF Conversion Script

This script processes a folder of multi-dimensional bioimaging data, applies drift correction, and converts the processed data into the TIF format. The user can specify various options for drift correction and projection methods to tailor the processing to their needs.
You can also skip the drift correction to just convert all your images to OME-TIF. It can also search subfolders and flatten out your folder structure. The output will then have double _ instead of / to move all files into a single folder but keep folder info.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Input Parameters](#input-parameters)
- [Output Files](#output-files)
- [Functionality](#functionality)
- [Requirements](#requirements)

## Installation
To use this script, ensure you have anaconda/miniconda installed, and a compatible conda environment see [Requirements](#requirements).
You can download the file manually, but you will need both `run_pipeline_helper_functions.py` and `convert_to_tif.py` (in the same folder).

## Usage

To run the script, use the following command line syntax:

```bash
python convert_to_tif.py -p <input_folder> [-e <file_extension>] [-R] [--collapse_delimiter <delimiter>] [-drift_ch <channel>] [-pmt <projection_method>] [-o <output_file_extension>]
```
or run it using the run_pipeline.exe pipeline manager with this config file
```bash
run:
- name: Convert to tif and collapse folder structure
  environment: drift
  commands:
  - python
  - ./standard_code/convert_to_tif.py # Paths that start with ./ and ends with .py are relative to the root folder of run_python_pipeline.exe
  - --folder_path: ./input # This can be the ./ in  ./input means that its relative to the config file. You can also use full paths
  - --extension: .nd2 # must include the dot
  - --search_subfolders # This is a flag, so it can be omitted to only search the top folder
  - --collapse_delimiter: __ # If you are searching in subfolders / will be replaced with double underscores as a default so folder/subfolder/image1.nd2 will become folder__subfolder__image1.tif in the output folder
  - --drift_correct_channel: 0 # zero means ch 1 [-1 means no drift correction (can also be omitted)]
```
Then you can simply run `run_python_pipeline.exe path/to/your/config_file.yaml`

### Arguments
- `-p` or `--input_folder`: Path to the folder containing images to be processed (required).
- `-e` or `--extension`: Specify a file extension to filter files (optional).
- `-R` or `--search_subfolders`: Flag to search for files in subfolders (optional).
- `--collapse_delimiter`: Delimiter to be used when collapsing file paths (default: `__`).
- `-drift_ch` or `--drift_correct_channel`: Channel to be used for drift correction (default: `-1`, which means no correction).
- `-pmt` or `--projection_method`: Method for projection (options: `max`, `sum`, `mean`, `median`, `min`, `std`). If not specified, no projection is applied.
- `-o` or `--output_file_name_extension`: Customize the output file name extension (default: `None`).

## Input Parameters
The script processes any valid multi-dimensional imaging files found in the specified input folder. The bioio package will open the files as a TCZYT numpy array.
- Time (T)
- Channel (C)
- Z-axis (Z)
- Y-axis (Y)
- X-axis (X)
It has been tested on OME-TIF, ImageJ tif, and .nd2 files. But any file format that can be opened with bioformats should work. Let me know if you have problems, or if it works on more file formats (so that i can expand the list)

## Output Files
For each processed input file, the script generates the following outputs:
- **OME-TIF file**: The converted image saved in TIF format, which contains drift-corrected (if applicable) and/or projected data.
- **Metadata YAML file**: A YAML file containing metadata regarding the processing steps, including information on drift correction and projection methods used.
- **Shifts Numpy file**: A NumPy file containing shift data for drift correction (only generated if drift correction is applied).

All output files will be saved in a newly created folder named `input_folder_tif` where the original input folder is located.

## Functionality

### Drift Correction
The script employs the `StackReg` algorithm to correct drift in the specified channel. It determines the shifts needed by performing a rigid body registration on a maximum projection across the Z-dimension. (3D registration is not implemented yet, but please request this if you need it)

### Projection Methods
Users have the option to project the data across the Z-dimension using:
- **Max**: Maximum projection
- **Sum**: Sum projection
- **Mean**: Average intensity projection
- **Median**: Median intensity projection
- **Min**: Minimum projection
- **Std**: Standard deviation projection

If no projection method is specified, the original data is used.

## Requirements
This script requires Python 3.x but is tested on python 3.12. You also need `standard_code/run_pipeline_helper_functions.py`
```
name: drift
channels:
  - conda-forge
  - anaconda
dependencies:
  - python=3.12
  - pystackreg
  - numpy
  - matplotlib
  - scyjava
  - tqdm
  - joblib
  - pip
  - pip:
      - bioio
      - bioio-ome-tiff
      - bioio-nd2
      - bioio-bioformats
```

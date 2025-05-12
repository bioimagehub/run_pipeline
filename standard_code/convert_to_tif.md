```markdown
# Image Drift Correction and TIF Conversion Script

This script processes a folder of multi-dimensional bioimaging data, applies drift correction, and converts the processed data into the TIF format. The user can specify various options for drift correction and projection methods to tailor the processing to their needs.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Input Parameters](#input-parameters)
- [Output Files](#output-files)
- [Functionality](#functionality)
- [Requirements](#requirements)

## Installation

To use this script, ensure you have the necessary Python packages installed. You can install them using pip:

```bash
pip install numpy pystackreg tqdm joblib pyyaml bioio
```

## Usage

To run the script, use the following command line syntax:

```bash
python script_name.py -p <input_folder> [-e <file_extension>] [-R] [--collapse_delimiter <delimiter>] [-drift_ch <channel>] [-pmt <projection_method>] [-o <output_file_extension>]
```

### Arguments
- `-p` or `--input_folder`: Path to the folder containing images to be processed (required).
- `-e` or `--extension`: Specify a file extension to filter files (optional).
- `-R` or `--search_subfolders`: Flag to search for files in subfolders (optional).
- `--collapse_delimiter`: Delimiter to be used when collapsing file paths (default: `__`).
- `-drift_ch` or `--drift_correct_channel`: Channel to be used for drift correction (default: `-1`, which means no correction).
- `-pmt` or `--projection_method`: Method for projection (options: `max`, `sum`, `mean`, `median`, `min`, `std`). If not specified, no projection is applied.
- `-o` or `--output_file_name_extension`: Customize the output file name extension (default: `None`).

## Input Parameters

The script processes any valid multi-dimensional imaging files found in the specified input folder. Each image must contain at least the following dimensions:
- Time (T)
- Channel (C)
- Z-axis (Z)
- Y-axis (Y)
- X-axis (X)

The input images can be in a variety of formats that the bioio library supports.

## Output Files

For each processed input file, the script generates the following outputs:
- **TIF file**: The converted image saved in TIF format, which contains drift-corrected (if applicable) and/or projected data.
- **Metadata YAML file**: A YAML file containing metadata regarding the processing steps, including information on drift correction and projection methods used.
- **Shifts Numpy file**: A NumPy file containing shift data for drift correction (only generated if drift correction is applied).

All output files will be saved in a newly created folder named `input_folder_tif` where the original input folder is located.

## Functionality

### Drift Correction
The script employs the `StackReg` algorithm to correct drift in the specified channel. It determines the shifts needed by performing a rigid body registration on a maximum projection across the Z-dimension.

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

This script requires Python 3.x and the following libraries:
- `numpy`
- `pystackreg`
- `tqdm`
- `joblib`
- `pyyaml`
- `bioio`

Ensure that you have all dependencies installed in your Python environment before running the script.

---

Enjoy working with your biological image datasets and feel free to modify the script to suit your specific needs!
```

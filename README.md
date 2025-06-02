# Python Pipeline Runner
This program allows users to execute a series of shell commands (e.g. python) defined in a YAML configuration file. The application can activate different Python environments and handle input and output directories based on the specified commands.
The aim is to standardise workflows and to keep a detailed record of what type of processing have been done.

## Features
- Load YAML configuration files to define tasks and their parameters.
- Automatically resolve paths for Python scripts and data inputs based on their context.
- Activate specified Python environments using Anaconda.
- 
## Requirements
- Tested on Windows, and Anaconda 
- You need to have the anaconda environments that can run your code already installed.
- Python scripts or command line scripts that accepts input arguments

## Installation

1. **Clone this repository:**

   ```bash
   git clone [https://github.com/username/repo-name.git](https://github.com/bioimagehub/run_pipeline.git)
   ```

2. Make or copy a configuration file to the toplevel of your image folder. the configuration file could look something like this
```yaml
run:
- name: Convert to tif and collapse folder structure
  environment: convert_to_tif # your python environment ( must already be installed)
  commands:
  - python
  - ./standard_code/python/convert_to_tif.py # strings that start with ./ and ends with .py will use the location of the run_pipeline folder and is usefull for executing code in e.g. ./run_pipeline/standard_code/python
  - --input-file-or-folder: ./input # strings that start with ./ with no endings are mapped relative to the yaml file. 
  - --extension: .nd2
  - --search-subfolders
  - --collapse-delimiter: __
  - --drift-correct-channel: -1
  - --projection-method: max
  last_processed: "2025-05-26" # If this line is present this part will be skipped. If you want to run just this part delete this line if you want to run everything use the  --force_reprocessing tag
- name: Convert multipoint images to tif and collapse folder structure
  environment: convert_to_tif
  commands:
  - python
  - ./standard_code/python/convert_to_tif.py
  - --input-file-or-folder: ./input_multipoint
  - --extension: .nd2
  - --search-subfolders
  - --collapse-delimiter: __
  - --drift-correct-channel: -1
  - --projection-method: max
  - --multipoint-files
  last_processed: "2025-05-26"
```
save this as a .yaml file in your experiment forder next to your folder with your images like this:

```bash
Experiment Folder
│   pipeline_config.yaml
│
└───input
        Image_1.nd2
        Image_2.nd2
        Image_3.nd2
        Image_4.nd2
```

3. Make the python environemnts that you have defined in your configuration file or use `base`. For the example above you need to have an environment called `drift` that have the libraries to run `convert_to_tif.py`, and `segment` to run `segment_threshold.py`.

4. open a command line tool E.g. Anaconda prompt or cmd

5. **Run the application:**

   ```bash
   go run main.go path/to/config.yaml
   ```

   If no argument is provided, the application will prompt you to input the path.

2. **Follow the prompts** in the terminal to see the output and any error messages.

## Licensing

This project is licensed under the MIT License. See the LICENSE file for more details.

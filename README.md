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
- You need to have the anaconda environments that can run your code already installed
- Python scripts that accepts input arguments

## Installation

1. **Clone this repository:**

   ```bash
   git clone [https://github.com/username/repo-name.git](https://github.com/bioimagehub/run_pipeline.git)
   ```

2. Make or copy a configuration file to the toplevel of your image folder. the configuration file could look something like this
```yaml
run:
- name: Collapse folder structure and save as .tif # A name that describes this part of the code
  environment: drift # name of the python environment you want to call
  commands: # This is a list of commands that you want to execute
  - python
  - ./standard_code/convert_to_tif.py " # You can add the full path to your code. things that start with ./ and ends with .py will be relative to this git repo
  - --input_folder: ./input # Things that start with ./ will be relative to the yaml file
  - --extension: .ims # remember the dot before extension
  - --search_subfolders
  - --collapse_delimiter: __
  - --projection_method: sum
- name: run find nuclei with threshold # Once the first run comand has been executed this will run.
  environment: segment
  commands:
  - python
  - ./standard_code/segment_threshold.py
  - --input_folder: ./input_tif
  - --output_folder: ./output_nuc_mask
  - --channels: 3
  - --median_filter_size: 15
  - --method: li
  - --min_size: 10000
  - --max_size: 55000
  - --watershed_large_labels
  - --remove_xy_edges
```
save this 
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

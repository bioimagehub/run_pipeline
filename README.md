# Python Pipeline Runner
This Go-based application allows users to execute a series of Python commands defined in a YAML configuration file. The application can activate different Python environments and handle input and output directories based on the specified commands.

## Features
- Load YAML configuration files to define tasks and their parameters.
- Automatically resolve paths for Python scripts and data inputs based on their context.
- Activate specified Python environments using Anaconda.
- Maintain state by skipping already processed segments and logging their last processed date.

## Requirements

- Go 1.16 or later
- Anaconda installed on your system
- Python scripts in the specified formats and locations

## Installation

1. **Clone this repository:**

   ```bash
   git clone https://github.com/username/repo-name.git
   cd repo-name
   ```

2. **Make sure you have Go installed.** You can download it from [golang.org](https://golang.org/dl/). You can also just use the .exe file if you are using windows

3. **Install dependencies (if applicable):**

   Run the following command to install any required packages:

   ```bash
   go mod tidy
   ```

4. **Set up the `.env` file** to configure your Anaconda path:

   ```plaintext
   ANACONDA_PATH=/path/to/your/anaconda
   ```

## Configuration

The configuration is defined in a YAML file. Here’s an example structure:

```yaml
run:
  - name: Convert to tif, flatten folder structure, extract core metadata
    environment: bioio # name of env as it is called with conda activate ... If you write "base" conda activate is skipped 
    commands: # This is just like you would use command line 
      - python # Must be called for python scripts
      - ./standard_code/bfconvert_folder.py # strings that beguin with ./ and end with .py are defined as relative to run_pipeline
      - --folder_path: ./Input # strings that beguin with ./ but do not end with .py are defined as relative to .yaml paths
      - --extension: .ims # You can define arg names like this
      - --search_subfolders # like this if its e.g. action="store_true"
      - --collapse_delimiter # Or over two lines. it is handled
      - __ # in the same way
  - name: The name of the second "segment" to be run
  ... same setup as abvove
```

- **name**: Descriptive name for the segment of work.
- **environment**: The Anaconda environment to activate.
- **commands**: List of commands to execute, which can include flags and file paths.

### Important Notes on Paths
- Python scripts indicated by paths starting with `./` and ending with `.py` are resolved relative to the main program directory.
- Paths starting with `./` (but not ending with `.py`) are resolved relative to the directory of the YAML file.

## Usage

1. **Run the application:**

   You can specify the path to the YAML configuration file as a command-line argument:

   ```bash
   go run main.go path/to/config.yaml
   ```

   If no argument is provided, the application will prompt you to input the path.

2. **Follow the prompts** in the terminal to see the output and any error messages.

## Licensing

This project is licensed under the MIT License. See the LICENSE file for more details.

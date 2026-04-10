# BIPHUB Pipeline Manager

## Overview

The BIPHUB Pipeline Manager is a robust, reproducible pipeline orchestration system developed for the [BIPHUB (Bioimage Informatics Hub)](https://www.uio.no/tjenester/it/forskning/kompetansehuber/biphub/) at the University of Oslo. This system provides a standardized framework for bioimage analysis workflows, emphasizing reliability, reproducibility, and accessibility for researchers worldwide.

## Architecture

### Core Components

1. **Go Pipeline Orchestrator** (`run_pipeline.go`, or `run_pipeline.exe`)
   - Universal command line orchestrator built in Go for cross-platform compatibility
   - Language-agnostic: can execute any command line program or script
   - Manages environment isolation using UV package manager and Conda environments
   - Provides reproducible execution with comprehensive logging and error handling
   - YAML-based configuration system using a top-level `run:` list of segments
   - Supports segment control flow with `type: normal`, `type: pause`, `type: stop`, and `type: force`
   - When making YAML configs for run_pipeline, do a quick run of `run_pipeline.exe -h` to get the latest commands and best practices

2. **Standard Code Collection** (`standard_code/`)
   - Curated Python modules for common bioimage analysis tasks
   - Type-hinted functions following Python best practices
   - Modular design for maximum reusability and minimal error propagation
   - Comprehensive test coverage for reliability

3. **Pipeline Configurations** (`pipeline_configs/`)
   - YAML configuration files defining specific analysis workflows
   - Parameterized pipelines for different imaging modalities
   - Example datasets and expected outputs for validation

### Key Features

- **Universal Orchestration**: Go orchestrator can execute any command line program, not limited to Python
- **Reproducible Environments**: Primarily uses UV for fast, reliable Python environment management, with Conda as backup for complex dependencies
- **Type Safety**: Comprehensive type hints throughout Python codebase for better IDE support and error prevention
- **Cross-Platform**: Go-based orchestrator ensures consistent behavior across Windows, macOS, and Linux
- **Modular Design**: Pluggable components allow easy extension and customization
- **Error Handling**: Robust error handling and logging for production-ready deployments

### Environment Management Status
- **Primary**: UV dependency groups are the standard way to run Python pipeline segments.
- **Default Environment**: Use `uv@3.11:default` for general-purpose Python CLIs unless a specialized environment is required.
- **Specialized Environments**: Keep dedicated groups only where there is a real technical reason, such as ERNet, TensorFlow-based denoising, ImageJ, Ilastik, or GPU-specific workflows.
- **Legacy Support**: Conda environment specs are still kept for compatibility and older workflows.
- **Virtual Environments**: UV creates environment folders such as `.venv_default_py3_11/` alongside other group-specific environments.
- **Development Policy**: New pipelines and modules should target UV unless a concrete dependency constraint requires something else.

## Supported Analysis Types

- **Segmentation**: ERnet, Cellpose, threshold-based, and Ilastik integration
- **Image Processing**: Format conversion, metadata extraction, channel merging
- **Measurement**: Mask analysis, distance calculations, edge detection
- **Tracking**: Time-series analysis with indexed mask tracking
- **Visualization & QC**: ROI generation for ImageJ integration, mask quality control, tracking validation, segmentation contour overlays, distance heatmaps, group-wise summary plots

## Usage

### Quick Start

1. Build the pipeline executable:
   ```bash
   go build -o run_pipeline.exe
   ```

2. Run a pipeline:
   ```bash
   ./run_pipeline.exe pipeline_configs/convert_to_tif.yaml
   ```

3. View the built-in YAML help and step-type reference:
   ```bash
   ./run_pipeline.exe -h
   ```

### Pipeline Configuration

Pipelines are defined as a top-level `run:` list. Each segment can either run commands or control pipeline flow.

Supported segment fields:
- `name`: Human-readable segment name.
- `environment`: Environment selector such as `uv@3.11:default`.
- `env`: Optional environment-variable map applied only to that segment.
- `use-linux-distro`: Optional WSL routing selector such as `Default` or a specific distro name.
- `commands`: Command list to execute.
- `type`: Optional control type. Supported values are `normal`, `pause`, `stop`, and `force`.
- `message`: Optional message used by control segments.

Common `env` use cases:
- Tune GPU batching per segment without changing code.
- Pass tool-specific configuration only to one pipeline step.
- Keep benchmark settings in YAML so they participate in status hashing and reprocessing.

Path tokens supported by the runner:
- `%REPO%/path`: Resolves relative to the repository/program root.
- `%YAML%/path`: Resolves relative to the YAML file location.
- `%VARNAME%/path`: Resolves using a variable loaded from `.env`.
- `./path`: Supported for backward compatibility, but deprecated in favor of `%REPO%` or `%YAML%`.

Example configuration structure:
```yaml
run:
- name: Convert ND2 to OME-TIFF
  environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/convert_to_tif.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.nd2'
  - --output-folder: '%YAML%/output_data'

- name: Fill holes on Linux GPU with custom batching
  environment: uv@3.11:fill-holes-gpu
  use-linux-distro: Default
  env:
    RP_GPU_KERNEL_TARGET_FREE_FRACTION: '0.85'
    RP_GPU_KERNEL_MAX_BATCH_WINDOWS: '16384'
    RP_GPU_KERNEL_WINDOW_BYTES_FACTOR: '6'
  commands:
  - python
  - '%REPO%/standard_code/python/fill_greyscale_holes.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data'
  - --kernel-size: 40
  - --kernel-overlap: half

- name: Enable force mode for later segments
  type: force
  message: 'Reprocessing all subsequent steps.'

- name: Pause for inspection
  type: pause
  message: 'Paused for user inspection.'

- name: Stop intentionally
  type: stop
  message: 'Pipeline stopped intentionally.'
```

## Development Guidelines

### Code Standards

- **Type Hints**: All Python functions must include comprehensive type hints
- **Documentation**: Every module and function requires docstrings following Google style
- **Error Handling**: Use specific exception types with meaningful error messages via logging not printing
- **Testing**: Unit tests required for all new functionality

### Contributing

1. Fork the repository
2. Create feature branch following naming convention: `feature/description`
3. Ensure all tests pass and type checking succeeds
4. Submit pull request with detailed description

## License and Attribution

```
MIT License

Copyright (c) 2024 BIPHUB - Bioimage Informatics Hub, University of Oslo
Written by Øyvind Fiksdahl Østerås for BIPHUB

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Project Structure

```
├── run_pipeline.go            # Main Go orchestrator
├── run_pipeline.exe           # Compiled Windows executable
├── .venv_*_py3_11/            # UV-managed per-group virtual environments
├── uv.lock                    # UV dependency lock file
├── go.sum                     # Go module checksums
├── pyproject.toml             # Python project configuration
├── run_pipeline_from_anywhere.bat  # Convenience launcher for Windows
├── TODO.txt                   # Active development task list
├── VERSION                    # Version tracking
├── standard_code/             # Reusable modules
│   ├── python/               # Python analysis functions
│   │   ├── plots/           # Visualization and QC modules
│   │   ├── drift_correction_utils/  # Drift correction algorithms
│   │   ├── bioio_imaris/    # Imaris file format support
│   │   └── Test_code/       # Development test scripts
│   └── NIS_Elemens_macros/  # NIS-Elements automation scripts
├── pipeline_configs/          # YAML pipeline definitions
├── pipeline-designer/         # Wails-based visual pipeline designer
├── conda_envs/               # Environment specifications (legacy/backup)
├── external/                 # Third-party tools (UV bundled)
│   └── UV/                  # UV package manager executables
├── go/                       # Go utility modules
│   └── find_anaconda_path/  # Anaconda detection utility
├── segmentation_strategy/    # Segmentation SOPs and workflow notes
├── logs/                     # Runtime logs and error outputs
├── output/                   # Generated outputs during local runs
├── temp/                     # Temporary workspace data
└── assets/                   # Documentation and logos
```

## Dependencies

- **Go 1.19+**: For pipeline orchestrator compilation
- **Python 3.11+**: For analysis modules
- **UV Package Manager**: Primary Python environment management (preferred)
- **Conda/Miniconda**: Backup for complex dependency management and legacy support
- **ImageJ/Fiji**: For ROI visualization and analysis

## Support and Documentation

- **Issues**: Submit bug reports and feature requests via GitHub Issues
- **Documentation**: Comprehensive guides available in project wiki
- **BIPHUB Support**: Contact BIPHUB team for institutional support

## AI Agent Instructions

## AI Personality

AI agents working with this codebase should adapt their behavior based on user requests and context. Let me know when you switch modes. There are three primary operational modes:

### Teacher Mode 🎓

**Trigger phrases:** "teach me", "teacher mode", "explain how to", "I want to learn", "guide me through"

**Behavior:**
- **DO NOT execute any commands or tools automatically**
- Focus on explanation and step-by-step guidance
- Break complex topics into digestible pieces
- Use conversational style, not overwhelming documentation dumps
- Wait for user confirmation before proceeding to next steps
- Ask clarifying questions to understand the user's experience level
- Provide practical, hands-on examples they can try themselves

**Example approach:**
```
"Let me explain how UV dependency management works. First, can you tell me - 
have you worked with Python package managers before like pip or conda? 
This will help me tailor the explanation to your level."
```

### Autonomous Work Mode 🤖

**Trigger phrases:** "work overnight", "work until done", "complete the TODOs", "work while I'm away", "work for [time period]"

**Behavior:**
- **Work continuously without asking for feedback or confirmation**
- Execute commands, make edits, and solve problems independently
- Use comprehensive error handling and graceful recovery
- Document all actions taken in detailed commit-like summaries
- Only stop when explicitly told to or when all tasks are complete
- Make reasonable assumptions when encountering ambiguous situations
- Prioritize robustness and safety over speed
- Ensure that you have the nessesary approvals before running a command

**Example approach:**
```
"I'll work on the TODO list continuously. I'll implement each item systematically, 
test as I go, and provide you with a comprehensive summary when complete or 
when I encounter any blocking issues that require your decision."
```

### Balanced Mode ⚖️ (Default)

**Trigger phrases:** No specific trigger - this is the default behavior

**Behavior:**
- **Execute commands and make changes proactively**
- Ask for clarification when requirements are ambiguous
- Confirm before making significant structural changes
- Provide explanations alongside actions
- Balance efficiency with transparency
- Stop and ask if encountering unexpected errors or edge cases

**Example approach:**
```
"I'll implement the drift correction function. I notice there are several 
algorithm options - I will start with GPU acceleration add the CPU compatibility to the TODO list with instructions? 
I'll start with the GPU version since I see CUDA dependencies already configured."
```

### Mode Detection Guidelines

- **Teacher Mode**: User explicitly asks to learn or understand something
- **Autonomous Mode**: User gives time-bounded work requests or indicates they'll be away
- **Balanced Mode**: All other scenarios, including general development requests

### Communication Style

**Teacher Mode**: Socratic questioning, step-by-step guidance, conversational
**Autonomous Mode**: Concise progress updates, detailed final summaries
**Balanced Mode**: Clear explanations of actions taken, questions when needed


### Image I/O and Helper Functions

**ALWAYS** use the following import for helper functions:

```python
import bioimage_pipeline_utils as rp
```

For all image reading, **ALWAYS** use:

```python
img = rp.load_tczyx_image(path)
img.data  # returns 5D TCZYX numpy array
img.dask_data  # returns 5D TCZYX dask array
img.dims  # returns a Dimensions object
img.dims.order  # returns string "TCZYX"
img.dims.X  # returns size of X dimension
img.shape  # returns tuple of dimension sizes in TCZYX order
img.get_image_data("CZYX", T=0)  # returns 4D CZYX numpy array
```

This ensures that all images are loaded as 5D arrays (TCZYX), even if the original image has fewer dimensions. This standardization makes looping over T, C, Z, Y, X safe and predictable (e.g., T=1 for single timepoint images).

For saving images, **ALWAYS** use:

```python
rp.save_tczyx_image(img, path)
```

This will ensure consistent output and metadata handling across the pipeline.

Never use other image I/O methods directly; always go through these helper functions for both reading and writing.

### CLI Example Format

### Standard CLI Arguments

For user-facing Python CLIs in `standard_code/python/`, standardize the core batch-processing arguments so pipeline YAML files are predictable across modules.

Required argument names for batch file-processing CLIs:

- `--input-search-pattern`
   - Glob pattern for input files.
   - Use this exact name instead of alternatives such as `--input`, `--input-file`, or `--input-folder` when the CLI processes one or more files from disk.

- `--output-folder`
   - Destination folder for outputs.
   - Use this exact name instead of alternatives such as `--output`, `--output-dir`, or `--save-folder`.

- `--output-suffix`
   - Suffix appended to the input stem before the file extension.
   - Use this exact name instead of `--suffix`, `--output-file-name-extension`, or other variants.
   - Default values should be explicit and module-specific, for example `_filled`, `_tracked`, or `_gaussian`.

- `--no-parallel`
   - Disable parallel processing.
   - If a CLI can process multiple files, it should expose this flag even if parallel execution is enabled by default.

- `--maxcores`
  - Maximum number of CPU cores to use for parallel processing.
  - Use this exact name instead of alternatives such as `--n-jobs`, `--workers`, `--max-workers`, or `--cores`.
  - Default behavior should be all available CPU cores minus 1.
  - If a CLI can process multiple files, it should expose this flag even if parallel execution is enabled by default.
  - `--maxcores` should be ignored when `--no-parallel` is set.

- `--log-level`
  - Logging verbosity for the CLI.
  - Use this exact name for user-facing Python CLIs so pipeline runs can control verbosity consistently.
  - Standard definition:
    `parser.add_argument('--log-level', type=str, default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level (default: WARNING)')`

Implementation rules:

- Prefer these exact five names in both argparse definitions and YAML examples.
- New CLI modules should also define a module-level logger with `logger = logging.getLogger(__name__)` and configure logging in `main()` after parsing arguments with `logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s - %(levelname)s - %(message)s')`.
- If a script currently uses legacy names such as `--suffix`, `--output-file-name-extension`, `--n-jobs`, `--workers`, or `--max-workers`, rename them to the standardized form.
- If a module is a true single-file utility and cannot reasonably support one or more of these arguments, document the reason clearly in the CLI help text and keep the deviation intentional rather than accidental.
- New modules should be reviewed against this standard before they are added to `standard_code/python/`.

When creating or documenting CLI tools (Python scripts in `standard_code/python/`), **ALWAYS** provide examples in YAML config format for `run_pipeline.exe`, not as bash/shell commands.

Environment selection for YAML examples:

- Use `environment: uv@3.11:default` as the standard environment for general-purpose Python CLIs that work with the shared default dependency group.
- Only use a module-specific or specialized environment when there is a real technical reason, for example Torch/ERNet, TensorFlow/Noise2Void, ImageJ, Ilastik, NIS-Elements, or GPU-specific drift correction.
- If a CLI works with the shared default environment, prefer documenting it with `uv@3.11:default` even if older YAML examples used a legacy per-module environment name.

**CORRECT format** (in argparse epilog):
```python
epilog="""
Example YAML config for run_pipeline.exe:
---
run:
- name: Process images (default settings)
  environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/module_name.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data'
  
- name: Process images (custom parameters)
  environment: uv@3.11:default
  commands:
  - python
  - '%REPO%/standard_code/python/module_name.py'
  - --input-search-pattern: '%YAML%/input_data/**/*.tif'
  - --output-folder: '%YAML%/output_data'
  - --parameter-name: value
  - --dry-run

- name: Pause for manual inspection
  type: pause
  message: 'Inspect outputs before continuing.'

- name: Stop intentionally
  type: stop
  message: 'Pipeline stopped intentionally.'

- name: Force reprocessing for later segments
  type: force
  message: 'Reprocessing all subsequent steps.'
"""
```

**INCORRECT format** (do not use):
```python
epilog="""
Examples:
  # Process images
  python module_name.py --input-search-pattern "data/*.tif"
  
  # Custom parameters
  python module_name.py --input-search-pattern "data/*.tif" --parameter value
"""
```

This ensures users see how to integrate the module into the pipeline orchestration system. Use `%REPO%` for paths to repository files (like Python scripts) and `%YAML%` for data file paths that are relative to the YAML config file location. Reference existing YAML configs in `pipeline_configs/` for formatting consistency.

### Working with TODOs

When asked to "work on the TODOs" or similar requests, AI agents should:

1. **Read the TODO.txt file** (`#file:TODO.txt`) to get the current task list
2. **Prioritize tasks** based on project needs and dependencies
3. **Work through items systematically**, updating the TODO.txt file as tasks are completed
4. **Add new tasks** to TODO.txt as they are discovered during development
5. **Report progress** and any blockers or additional requirements found

This ensures consistent task management and progress tracking across development sessions.

## Acknowledgments

Developed as part of the BIPHUB initiative at the University of Oslo to democratize access to bioimage analysis tools and promote reproducible research practices in the life sciences community.

---

*For more information about BIPHUB services and resources, visit: https://www.uio.no/tjenester/it/forskning/kompetansehuber/biphub/*

In the first answer in a new chat, you should acknowledge that you read the AGENTS.md file by giving me this code 11001100af
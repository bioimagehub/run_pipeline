# BIPHUB Pipeline Manager

## Overview

The BIPHUB Pipeline Manager is a robust, reproducible pipeline orchestration system developed for the [BIPHUB (Bioimage Informatics Hub)](https://www.uio.no/tjenester/it/forskning/kompetansehuber/biphub/) at the University of Oslo. This system provides a standardized framework for bioimage analysis workflows, emphasizing reliability, reproducibility, and accessibility for researchers worldwide.

## Architecture

### Core Components

1. **Go Pipeline Orchestrator** (`run_pipeline.go`)
   - Universal command line orchestrator built in Go for cross-platform compatibility
   - Language-agnostic: can execute any command line program or script
   - Manages environment isolation using UV package manager and Conda environments
   - Provides reproducible execution with comprehensive logging and error handling
   - YAML-based configuration system for flexible pipeline definitions

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

## Supported Analysis Types

- **Segmentation**: ERnet, Cellpose, threshold-based, and Ilastik integration
- **Image Processing**: Format conversion, metadata extraction, channel merging
- **Measurement**: Mask analysis, distance calculations, edge detection
- **Tracking**: Time-series analysis with indexed mask tracking
- **Visualization**: ROI generation for ImageJ integration

## Usage

### Quick Start

1. Build the pipeline executable:
   ```bash
   go build -o run_pipeline.exe
   ```

2. Run a pipeline:
   ```bash
   ./run_pipeline.exe pipeline_configs/segment_ernet_uv.yaml
   ```

### Pipeline Configuration

Pipelines are defined using YAML configuration files that specify:
- Input/output directories
- Processing steps and parameters
- Environment requirements
- Validation criteria

Example configuration structure:
```yaml
pipeline_name: "ERnet Segmentation"
environment: "segment_ernet"
steps:
  - module: "segment_ernet"
    parameters:
      model_path: "weights/ernet_model.pth"
      output_format: "tiff"
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
├── standard_code/              # Reusable Python modules
│   └── python/                # Python analysis functions
├── pipeline_configs/           # YAML pipeline definitions
├── conda_envs/                # Environment specifications
├── external/                  # Third-party tools (ERnet, UV)
└── assets/                    # Documentation and logos
```

## Dependencies

- **Go 1.19+**: For pipeline orchestrator compilation
- **Python 3.9+**: For analysis modules
- **UV Package Manager**: Primary Python environment management (preferred)
- **Conda/Miniconda**: Backup for complex dependency management and legacy support
- **ImageJ/Fiji**: For ROI visualization and analysis

## Support and Documentation

- **Issues**: Submit bug reports and feature requests via GitHub Issues
- **Documentation**: Comprehensive guides available in project wiki
- **BIPHUB Support**: Contact BIPHUB team for institutional support

## AI Agent Instructions


### Image I/O and Helper Functions

**ALWAYS** use the following import for helper functions:

```python
import run_pipeline_helper_functions as rp
```

For all image reading, **ALWAYS** use:

```python
img = rp.load_bioio(path)
```

This ensures that all images are loaded as 5D arrays (TCZYX), even if the original image has fewer dimensions. This standardization makes looping over T, C, Z, Y, X safe and predictable (e.g., T=1 for single timepoint images).

For saving images, **ALWAYS** use:

```python
rp.save_bioio(img, path)
```

This will ensure consistent output and metadata handling. (If `save_bioio` is not yet implemented, it should be created to mirror the behavior of `load_bioio`.)

Never use other image I/O methods directly; always go through these helper functions for both reading and writing.

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

In the first anwser in a new chat, you should aknowledge that you read the AGENTS.md file by giving me this code 11001100af
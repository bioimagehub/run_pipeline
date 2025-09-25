# run_pipeline (Python / ImageJ workflow orchestrator)

<p align="center">
  <img src="assets/logo_concept_A.svg" alt="run_pipeline logo - flow nodes to pixel" width="200" />
  <br/>
  <sub><em>Deterministic multi-step image analysis pipeline orchestrator</em></sub>
</p>

Execute multi-step image analysis workflows described in a YAML file. Each step (segment) can run a Python script in a specified Conda environment or an ImageJ macro/command, and the pipeline keeps a provenance trail by updating the same YAML file with when and with which code version each segment was processed.

## Key Features
* YAML-driven workflow (ordered list of segments).
* Segment types: normal (default), pause, stop, force.
* Force reprocessing control via CLI flag, on-demand segment type, or auto-escalation when an unprocessed step is encountered.
* Auto environment activation (Conda) or ImageJ headless execution (environment: `imageJ`).
* Path resolution rules for convenience and reproducibility.
* Self-updating YAML: adds/updates `last_processed` (YYYY-MM-DD) and `code_version` after each successful segment.
* Git / VERSION / ldflags based version stamping.
* Interactive discovery of Anaconda base path and ImageJ executable (stored in `.env` as `CONDA_PATH` and `IMAGEJ_PATH`).
* Windows focused (tested on Windows + Anaconda). Go single static binary possible.
* Mix Python scripts, executables, and ImageJ macros in one pipeline.

## Segment Types
| Type | Behavior |
|------|----------|
| (omitted / normal) | Executes commands normally. |
| pause | Prints optional `message`, waits for Enter, then continues. |
| stop  | Prints optional `message` and terminates the pipeline. |
| force | Activates force mode for all subsequent segments (reprocess even if previously done). Optional `message`. |

## Reprocessing Logic
There are three ways force mode can be triggered:
1. CLI flag: `--force_reprocessing` (or `-f`) — all segments re-run regardless of `last_processed`.
2. Segment type `force` inside the YAML — activates force mode for everything after that point.
3. Automatic escalation: during a normal run, the moment the pipeline encounters the first segment missing `last_processed`, it automatically enables force mode for all following segments (ensures downstream consistency).

To re-run a single segment without full force mode, remove its `last_processed` line and run normally (no `--force_reprocessing`). Everything after it will also re-run due to the automatic escalation rule. If you want only that one segment, temporarily comment out or add `last_processed` to the later ones.

## Path Resolution Rules
Inside a segment's `commands` list:
* Strings beginning with `./` and ending with `.py`, `.ijm`, or `.exe` are resolved relative to the pipeline program directory (root of this repository / binary location).
* Other strings beginning with `./` are resolved relative to the directory containing the YAML configuration file (your experiment folder).
* Map style entries (`--flag: value`) will have their value path-resolved using the same rules.

## Auto-updating Metadata
After a segment succeeds:
* `last_processed` is set to the current date.
* `code_version` is written using (priority): ldflags-injected version/commit → `VERSION` file → Git describe/commit → `unknown`.

## ImageJ Steps
Set `environment: imageJ` and provide commands/macros. The program will:
* Look for `IMAGEJ_PATH` in `.env` (prompt once if absent).
* Run headless with `--ij2 --headless --console`.
* Each string command with `./... .ijm` is passed using `--run`.
* Map style arguments become `"key='value'"` parameters to ImageJ.

## Example YAML (mixed features)
```yaml
run:
- name: Convert raw ND2 to TIFF
  environment: convert_to_tif
  commands:
    - python
    - ./standard_code/python/convert_to_tif.py
    - --input-file-or-folder: ./input
    - --extension: .nd2
    - --search-subfolders
    - --projection-method: max
    - --collapse-delimiter: __

- name: Quick pause to inspect outputs
  type: pause
  message: "Check converted images before segmentation."

- name: Threshold segmentation
  environment: segment_threshold
  commands:
    - python
    - ./standard_code/python/segment_threshold.py
    - --input-folder: ./input_tif
    - --channel: 0

- name: Enable force for downstream benchmark
  type: force
  message: "Re-running remaining steps regardless of prior state."

- name: Measure masks
  environment: mask_measure
  commands:
    - python
    - ./standard_code/python/mask_measure.py
    - --mask-folder: ./masks

- name: Export ROIs with ImageJ macro
  environment: imageJ
  commands:
    - ./standard_code/python/mask2imageJROI.ijm
    - outputDir: ./roi_output

- name: Stop intentionally
  type: stop
  message: "Pipeline finished demonstration."
```

After running, successful segments will have `last_processed` and `code_version` appended/updated automatically.

## Command List Format
`commands` can mix:
* Bare strings (e.g. `python`).
* Script paths (resolved as described above).
* Flag maps: `--flag: value` or standalone flags: `--search-subfolders` (value omitted / null in YAML becomes just the flag).

## Installation / Build
Prerequisites: Go (1.21+ recommended) and Anaconda (on Windows). ImageJ optional.

Clone:
```powershell
git clone https://github.com/bioimagehub/run_pipeline.git
cd run_pipeline
```

Build a versioned binary (optional ldflags):
```powershell
go build -ldflags "-X main.Version=v0.3.0" -o run_pipeline.exe run_pipeline.go
```

Or just run directly:
```powershell
go run run_pipeline.go path\to\your_config.yaml
```

### Conda Environments
Each segment specifies an `environment` name. You must already have that Conda environment (or use `base`). Most common environments used by the bundled scripts can be created from the YAML specs in the `conda_envs/` folder (e.g. `conda env create -f conda_envs/segment_threshold.yml`). You only need to create an environment once; the runner just activates it. Feel free to add your own environment YAMLs there to standardize team setups.

## Usage
```powershell
run_pipeline.exe [options] path\to\config.yaml

Options:
  -f, --force_reprocessing  Re-run all segments regardless of state.
  -h, --help                Show help and YAML template.
```

First run will prompt for Anaconda base and (if used) ImageJ path; stored in `.env` next to the binary.

## Typical Experiment Layout
```
ExperimentFolder/
  pipeline.yaml
  input/                # raw data
  input_tif/            # produced by earlier steps
  masks/                # segmentation results
  roi_output/           # ImageJ ROI exports
```

## Standard Scripts (standard_code/python)
* convert_to_tif.py – Convert raw formats (e.g. ND2/IMS) to TIFF, collapse folder structure, projections, drift correction.
* copy_files_with_extension.py – Copy files filtering by extension.
* delete_folder.py – Remove a folder recursively.
* extract_metadata.py – Extract image acquisition metadata and save sidecar YAML/CSV.
* find_maxima.py – Detect local maxima (spot candidates) in images.
* mask_get_distance_matrix.py – Compute pairwise distances between mask objects.
* mask_get_distance_to_point.py – Distances from each mask/object to a specified coordinate.
* mask_get_edges.py – Derive edges/boundaries or adjacency relationships of masks.
* mask_measure.py – Measure intensity / morphology features for masks.
* mask_substract.py – Subtract one mask set from another (e.g. exclusion). (Filename typo: subtract.)
* mask2imageJROI.py – Convert mask data to ImageJ ROI format.
* merge_channels.py – Merge multiple channel images into composite stacks or reorder channels.
* rename_files.py – Batch rename files with a pattern.
* rename_folder.py – Rename/move a folder.
* run_pipeline_helper_functions.py – Shared helper utilities used by scripts.
* segment_ilastik.py – Run or interface with Ilastik-based segmentation.
* segment_nellie.py – Segmentation using the Nellie model/pipeline.
* _segment_ernet.py – ERnet-v2 segmentation wrapper (imports ERnet repo's Inference module and runs inference; requires separate ERnet checkout and weights).
* segment_threshold.py – Simple threshold-based segmentation.
* track_indexed_mask.py – Track labeled mask objects across frames / timepoints.

## Drift correction in convert_to_tif.py
`standard_code/python/convert_to_tif.py` can optionally correct XY drift on time series (TCZYX).

- Enable drift correction with: `--drift-correct-channel <int>` (0-based channel index). Use `-1` (default) to disable.
- Select algorithm with a single flag: `--drift-correct-method {cpu|gpu|cupy|auto}`
  - `cpu` → StackReg translation (pystackreg)
  - `gpu` → pyGPUreg (GPU)
  - `cupy` → CuPy phase correlation (GPU)
  - `auto` → try `cupy` → `gpu` → `cpu` in that order

Outputs
- The corrected OME-TIFF is written to the chosen output path.
- If drift correction ran, a sidecar `*_shifts.npy` is saved containing either the GPU shifts (T×2) or CPU StackReg matrices (T×3×3).

Prerequisites
- CuPy mode: install CuPy in the `convert_to_tif` Conda env. For NVIDIA CUDA on Windows, either:
  - `conda install -c conda-forge cupy` (matching your CUDA toolchain), or
  - `pip install cupy-cuda11x` (pick the build that matches your CUDA)
- pyGPUreg mode (default when no flag): requires `pyGPUreg` in the env.
- CPU mode: requires `pystackreg`.

Examples (PowerShell)
```powershell
# CuPy GPU drift correction
python .\standard_code\python\convert_to_tif.py \
  --input-search-pattern ".\input_tif\*.tif" \
  --projection-method max \
  --drift-correct-channel 0 \
  --drift-correct-method cupy

# CPU StackReg drift correction
python .\standard_code\python\convert_to_tif.py \
  --input-search-pattern ".\input_tif\*.tif" \
  --projection-method max \
  --drift-correct-channel 0 \
  --drift-correct-method cpu

# GPU pyGPUreg (default)
python .\standard_code\python\convert_to_tif.py \
  --input-search-pattern ".\input_tif\*.tif" \
  --projection-method max \
  --drift-correct-channel 0 \
  --drift-correct-method gpu
```

## command line Scripts
You can also invoke any CLI-capable tool (Python modules, installed packages, standalone `.exe` files) inside a segment. This lets you integrate tools like Cellpose, Ilastik exporters, FFMPEG, or your own compiled utilities directly into the unified provenance-tracked workflow.

### General Patterns
Inside `commands`:
* Stand‑alone flags (no value) → add as a plain string: `-v` or `--verbose`.
* Flags with a value → use a YAML map: `--diameter: 80` (this becomes `--diameter 80`).
* Python module execution → include `python` then `-m` then the module name.
* Executables (`tool.exe`) → put the path (relative with `./` if stored in repo, or absolute) as the first runtime token after environment activation.
* If a path contains spaces, prefer referencing it via a relative path using `./` so the runner resolves it; Windows `cmd` will receive it already tokenized (avoid adding extra quotes unless required by the external tool).

### Example: Cellpose (nuclei)
```yaml
- name: Cellpose nuclei segmentation
  environment: cellpose2   # Conda env containing cellpose
  commands:
    - python                # interpreter
    - -m                    # run module
    - cellpose              # module name
    - --image_path: ./input_tif
    - --output: ./masks_cellpose
    - --pretrained_model: nuclei
    - --channels: 0,0       # (nuclei, cytoplasm) style channel pairing
    - --diameter: 80
    - --verbose
```

### Example: Cellpose (cyto2) using direct `cellpose` entry point
```yaml
- name: Cellpose cyto2 segmentation
  environment: cellpose3
  commands:
    - cellpose              # if the entry point script is on PATH in that env
    - --image_path: ./input_tif
    - --output: ./masks_cellpose_cyto
    - --pretrained_model: cyto2
    - --channels: 0,0
    - --save_tif
```

### Example: Running an external executable
```yaml
- name: Custom feature extractor
  environment: base
  commands:
    - ./bin/feature_extractor.exe
    - --input: ./masks
    - --out: ./features
```

### Example: Chaining a helper Python script and passing a folder
```yaml
- name: Copy selected TIFFs
  environment: convert_to_tif
  commands:
    - python
    - ./standard_code/python/copy_files_with_extension.py
    - --input-folder: ./input_tif
    - --extension: .tif
    - --output-folder: ./subset
```

### Tips
* Want a literal value that starts with `./` but should NOT be path-resolved? Add a harmless prefix (e.g. `././`) and strip later in the called script, or place it inside quotes in a wrapper script; current resolver always interprets leading `./`.
* Mix Python and non-Python steps freely; each segment re-activates the specified environment.
* For very long argument lists consider creating a small wrapper script and just call that in the pipeline for clarity.

### Cellpose Notes
Ensure the selected environment (`cellpose2`, `cellpose3`, etc.) actually has Cellpose installed. Diameter, channels, and model names must match the version you installed. Remove or edit `last_processed` to re-run.

### Real-world Example: Two-pass Cellpose (cytoplasm then nucleus)
Below mirrors a practical workflow: first detect cytoplasm with a cyto2 model variant, then nuclei. Each pass writes outputs to separate folders so later steps (e.g. channel merge, mask measurement) can reference them independently.

```yaml
- name: run cellpose to find cytoplasm
  environment: cellpose3
  commands:
    - cellpose
    - --dir: ./cellpose_input          # Input images (e.g. merged channels or preprocessed TIFFs)
    - --savedir: ./cellpose_cyt        # Output directory for cytoplasm masks
    - --channel_axis: 0                # 0 = first axis is channel (adjust if your data layout differs)
    - --use_gpu
    - --save_tif
    - --chan: 1                        # Primary channel index (cytoplasm)
    - --chan2: 2                       # Secondary channel index (optional, e.g. nucleus assist)
    - --pretrained_model: cyto2_cp3    # Custom / renamed model inside the env
    - --diameter: 300                  # Approximate object size in pixels; 0 lets Cellpose auto-estimate
    - --cellprob_threshold: -4         # Relax threshold for faint objects
    - --exclude_on_edges               # Drop partial objects touching image edge
    - --verbose

- name: run cellpose to find nucleus
  environment: cellpose3
  commands:
    - cellpose
    - --dir: ./cellpose_input
    - --savedir: ./cellpose_nuc
    - --channel_axis: 0
    - --use_gpu
    - --save_tif
    - --save_rois                      # Also export ROI representations if supported
    - --chan: 2                        # Nucleus channel
    - --chan2: 0                       # Set 0 if no secondary channel (or helper structural channel)
    - --exclude_on_edges
    - --pretrained_model: nuclei
    - --diameter: 250
    - --verbose
```

Notes:
* Remove `last_processed` / `code_version` lines if copying directly; they are auto-added after a successful run.
* Adjust `--channel_axis` if data order is (Z, Y, X, C) or uses a different convention.
* Large diameters (>200) may benefit from tiling or GPU memory considerations depending on image size.

### ERnet-v2 Notes
Requires cloning ERnet-v2 repository and obtaining a .pth checkpoint. Point `--ernet-dir` to the repo (or its `Inference` folder) and pass `--weights`.

Quick start (PowerShell):
```powershell
# Create env once
conda env create -f .\conda_envs\segment_ernet.yml

# Run wrapper (CPU example)
conda activate segment_ernet
python .\standard_code\python\_segment_ernet.py `
  --input-search-pattern ".\pipeline_configs\input_tif\*.tif" `
  --ernet-dir ".\external\ERnet-v2" `
  --weights ".\external\ERnet-v2\weights\ernet_swin.pth" `
  --output-dir ".\_tmp_out\ernet" `
  --cpu
```
If `--graph-metrics` is enabled, ensure `sknw` is installed; otherwise omit that flag.



## Adding New Scripts
Place them under `standard_code/python/` and reference them with `./standard_code/python/your_script.py` in the YAML.
Alternatively use the full path to the script in the yaml file

## Troubleshooting
* Segment skipped unexpectedly → It already has `last_processed`; remove that line to re-run.
* Only one step updated → Force mode not triggered; use a `force` type or `-f` flag to run everything in the YAML.
* Path not found → Check whether you intended repo-root resolution (`./... .py`) vs experiment-relative path.

## Versioning
The binary embeds version info via (priority): ldflags → `VERSION` file → Git describe. Each processed segment records `code_version` for provenance.

## License
MIT License (see `LICENSE`).

## Disclaimer
Please use at 
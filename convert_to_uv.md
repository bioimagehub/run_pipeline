# Gradual Migration to uv

## Goals
- Keep existing conda-based pipeline running while introducing uv for selected segments.
- Speed up environment creation and make dependencies reproducible via a single `uv.lock`.
- Allow parallel use of multiple Python versions and GPU stacks without cross-contamination.

## Current Layout
- Conda YAML specs live under `conda_envs/` (e.g. `conda_envs/segment_ernet.yml`).
- Pipeline config files in `pipeline_configs/` reference environments by name (e.g. `environment: ernet-gpu`).
- `run_python_pipeline.go` activates conda environments by calling `conda activate <name>` before running the segment commands.

## uv Structure
- Add one `pyproject.toml` + `uv.lock` at the repo root.
- Use uv dependency groups to mirror each conda environment:
  - `uv add --group segment_ernet ...`
  - `uv add --group convert_to_tif ...`
- Each group can pin a different Python version; set it via `uv add python==3.11 --group segment_ernet` or the `[tool.uv.sources]` section in `pyproject.toml`.
- GPU wheels (e.g. `torch==2.4.0+cu124`) require `--index-url https://download.pytorch.org/whl/cu124`. uv stores this in the lockfile so future installs are deterministic.

## Pilot: segment_ernet
1. Create dependency group from `conda_envs/segment_ernet.yml`:
   ```powershell
   uv add --group segment_ernet torch==2.4.0+cu124 torchvision==0.19.0+cu124 torchaudio==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124
   uv add --group segment_ernet python-dotenv tqdm "numpy>=1.24,<2.0" numba scipy scikit-image opencv-python matplotlib pillow "networkx<3" imagecodecs ipykernel timm einops sknw==0.12 bioio bioio-bioformats bioio-ome-tiff bioio-tifffile bioio-nd2
   uv lock
   ```
2. Sync locally when needed: `uv sync --group segment_ernet` (creates `.venv` by default).
3. Smoke test: `uv run --group segment_ernet python -m standard_code.python._segment_ernet --help`.

## YAML Changes
- Introduce a pilot config, e.g. `pipeline_configs/segment_ernet_uv.yaml`, with:
  ```yaml
  run:
  - name: Segment ER with ERnet-v2 (UV pilot)
    environment: uv:segment_ernet
    commands:
    - python
    - '%REPO%/standard_code/python/_segment_ernet.py'
    - ...
  ```
- `environment: uv:<group>` signals the runner to use uv.
- Maintain existing YAML (`segment_ernet.yaml`) so users can fall back to conda until the uv path is proven.

## Runner Adjustments (`run_python_pipeline.go`)
- Split the current `makePythonCommand` into:
  - `makeCondaPythonCommand` (existing behavior).
  - `makeUvCommand` that builds `cmd /C uv run --group <name> ...`.
- In the main loop, branch on `strings.HasPrefix(strings.ToLower(segment.Environment), "uv:")`:
  - Skip Anaconda discovery for uv segments.
  - Trim the `uv:` prefix to obtain the group name.
  - Optionally set `cmd.Dir` to the repo root so uv finds `pyproject.toml`.
- Leave ImageJ handling untouched.

## Validation & Rollout
- Verify CUDA availability inside the uv-managed environment (e.g. `uv run --group segment_ernet python -c "import torch; print(torch.cuda.is_available())"`).
- Execute the pilot YAML via the Go runner and compare outputs with the conda path.
- Document the workflow: how to install uv, sync groups, and run the pipeline.
- After confidence is built, migrate additional environments by repeating the group creation process.
- Once all segments use `uv:...`, retire the conda YAMLs and related helper scripts.

## Useful Commands
- Install uv (one-time): `pipx install uv` or download the standalone binary.
- Create/update dependencies: `uv add --group <name> <packages>`.
- Sync environment: `uv sync --group <name>`.
- Run pipeline segment manually: `uv run --group <name> python path/to/script.py ...`.
- Clean venv caches if necessary: `uv clean`.

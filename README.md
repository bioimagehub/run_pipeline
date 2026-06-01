# BIPHUB Pipeline Manager

Cross-platform pipeline orchestration for reproducible bioimage workflows. The repository contains source code, pipeline definitions, reusable Python modules, and release automation for Windows and Linux builds.

## Versioning

This project now uses a source-plus-tag release model:

- [VERSION](VERSION) is the editable source-of-truth version used by Python packaging and local development builds.
- Git tags mark released versions in the form `vX.Y.Z`.
- Release binaries embed the version, commit, and build date through Go ldflags.
- The tag and [VERSION](VERSION) must match for a release to publish.

## Release Artifacts

The git repository should contain source code, not built executables. Release binaries for Windows and Linux are produced by GitHub Actions and published through GitHub Releases.

Ignored local artifacts include:

- root build outputs such as `run_pipeline.exe` and `run_pipeline`
- release archives such as `.zip` and `.tar.gz`
- generated runtime folders such as `logs/`, `output/`, and most of `temp/`
- local virtual environments such as `.venv*/`

If a built file is already tracked by git, ignoring it is not enough. Remove it from the index once and keep it locally:

```powershell
git rm --cached run_pipeline.exe
git rm --cached run_pipeline
git commit -m "Stop tracking built artifacts"
```

## Continuous Integration

GitHub Actions runs CI on both Windows and Linux for pushes and pull requests. CI performs:

- `go test ./...`
- a local build with injected version metadata

Release publishing is separate and only runs for version tags.

## Release Workflow

1. Update [VERSION](VERSION) to the new semantic version, for example `0.0.19`.
2. Commit the change.
3. Create an annotated git tag with the matching `v` prefix.
4. Push the branch and the tag.
5. GitHub Actions builds Windows and Linux binaries and publishes them to GitHub Releases.

Example:

```powershell
git add VERSION
git commit -m "Release v0.0.19"
git tag -a v0.0.19 -m "Release v0.0.19"
git push
git push origin v0.0.19
```

## Produced Release Files

Tagged releases generate:

- Windows `.zip` archives
- Linux `.tar.gz` archives
- a `checksums.txt` file

Each archive includes the built binary and supporting files needed for distribution.

## Teaching Assistant (Offline Drive Mode)

Two scripts let you prepare an AI-powered lecture at home and then have a conversation with it during a drive — fully offline, no internet required.

### Prerequisites

Install [Ollama](https://ollama.com) and pull a model once (requires internet):

```bash
ollama pull llama3.2        # ~2 GB, good quality
# or a smaller model:
ollama pull phi3:mini       # ~2 GB, faster on CPU
```

### Step 1 — Prepare the lecture (at home)

Run the pipeline or call the script directly:

```bash
# via pipeline runner
run_pipeline.exe pipeline_configs/teach_prepare.yaml

# or directly (edit --topic as needed)
python standard_code/python/teach_prepare.py \
  --topic "sound waves" \
  --duration 60 \
  --model llama3.2 \
  --output-folder ./lectures
```

This calls Ollama locally, generates a structured 1-hour lecture, and saves it to `lectures/sound_waves.json`. Takes 1-3 minutes.

### Step 2 — Chat during the drive (fully offline)

Open a terminal and run:

```bash
python standard_code/python/teach_chat.py --lecture lectures/sound_waves.json
```

The teacher introduces the topic and guides you through the lecture conversationally.

| Input | Effect |
|-------|--------|
| *(Enter)* | Teacher continues to the next section |
| `hint` | Simpler explanation of the last point |
| `quiz` | Teacher quizzes you on material so far |
| `summary` | Recap of everything covered so far |
| `quit` | End the session |

Ollama must be running (`ollama serve`) but no internet connection is needed once the lecture file and model are downloaded.

---

## Recommended Versioning Policy

Use semantic versioning:

- increment patch for bug fixes
- increment minor for new backwards-compatible features
- increment major for breaking CLI or YAML changes

Current repository version: `0.0.18`
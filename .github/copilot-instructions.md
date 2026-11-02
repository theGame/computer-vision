# Copilot instructions for this repository

## Build, test, and lint commands

This repository is a set of standalone computer-vision demos, not a library with a single build pipeline. Use the repo root as the working directory and keep the same Python virtual environment active in both VS Code and the terminal.

### Environment setup

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linting

The GitHub Actions workflow runs Flake8 on every push/PR and treats the configured checks as required:

```bash
python -m pip install flake8
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --max-complexity=10 --max-line-length=127 --statistics
```

### Testing

There is no active test suite in the repository today. The CI workflow includes a placeholder pytest step that prints `No tests implemented yet.`

For a single test when tests are introduced later, follow pytest conventions such as:

```bash
python -m pytest path/to/test_file.py
python -m pytest path/to/test_file.py -k "test_name"
```

If a test file does not exist yet, do not assume a hidden suite is present; the repo currently behaves like a demo project rather than a package with automated checks.

### Running the projects

Run each project as a module from the repository root:

```bash
python -m image_classification.main
python -m color_capture.main
python -m blurring_face.main
python -m text_detection.main
python -m tensor_flow.main
python -m yolov.main
```

The repo's README explicitly warns to avoid running package modules by file path and to keep the terminal and VS Code on the same Python environment.

## High-level architecture

This repository is organized as a collection of independent demo projects, each with its own folder and local assets. The pattern is consistent across the repo:

- Each project directory (`blurring_face/`, `color_capture/`, `image_classification/`, `text_detection/`, `tensor_flow/`, `yolov/`) contains a `main.py` entrypoint and its own dataset or model assets.
- The sample data and model files live next to the code instead of in a centralized `data/` or `models/` directory.
- Shared or reusable logic is minimal; repo-level utilities are limited to things like `utils/color_helper.py`.
- Dependencies are centralized in the root `requirements.txt` rather than per-project requirements files.
- Most examples use relative paths based on `os.path.dirname(__file__)` to read images, videos, or model files from their local folders.

This means the correct mental model is “many small, self-contained CV experiments” rather than “a single app with a package layout.” When changing or adding code, keep the project-local structure intact and avoid introducing a new monorepo-like abstraction unless the task explicitly requires it.

## Key conventions

- Prefer module execution (`python -m <project>.main`) over direct script paths. The README calls this out as a project convention.
- Keep project-local files in their own folder (`images/`, `model/`, `output/`, `videos/`) and resolve them relative to the script file rather than assuming a global working directory.
- This repo is intentionally demo-oriented and not production-hardened. Code often reads bundled sample assets and is designed for learning rather than deployment.
- Several demos require external runtime resources: webcam access for `color_capture`, a GUI for display windows, and compatibility-sensitive runtime libraries such as TensorFlow.
- The TensorFlow example is a known compatibility edge case: the README states it is currently unavailable on Python 3.14 until TensorFlow supports that version. Do not treat this as a generic Python issue; it is a repo-specific limitation.
- The generated outputs are intentionally stored under project folders (`output/`, `model.pkl`, etc.), so new code should follow that same convention unless the task clearly requires a different location.

## Additional repo context

- The root README is the primary source for environment and usage instructions.
- The CI workflow is the source of truth for lint rules and the repository’s current validation posture.
- There are no dedicated AGENTS/CLAUDE/Cursor instruction files in the repo today; this file should be treated as the main repo guidance for Copilot sessions.

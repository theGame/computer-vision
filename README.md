[![Python package](https://github.com/theGame/computer-vision/actions/workflows/python-package.yml/badge.svg)](https://github.com/theGame/computer-vision/actions/workflows/python-package.yml)

# Computer Vision Projects

Small computer-vision exercises for image classification, webcam color detection, and face blurring.

## Project Layout

```text
blurring_face/
	images/neymar-face.png
	model/blaze_face_short_range.tflite
	output/
	main.py
color_capture/
	main.py
image_classification/
	images/empty/
	images/not_empty/
	main.py
utils/
	color_helper.py
requirements.txt
```

## Requirements

- Python 3.11 is recommended for compatibility with the project's native computer-vision packages.
- A webcam is required only for color capture.
- Install the dependencies listed in `requirements.txt`.

## Setup

From the repository root, create and activate a virtual environment.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run commands from the repository root with the virtual environment active. In VS Code, select that environment's interpreter as well.

## Run a Project

### Image Classification

Train and evaluate an SVM using the images in `image_classification/images/empty` and `image_classification/images/not_empty`:

```bash
python -m image_classification.main
```

The script reports test accuracy and saves the trained classifier as `model.pkl` in the current working directory (the repository root when run as above).

### Webcam Color Capture

```bash
python -m color_capture.main
```

Choose a color in the terminal picker. The webcam window displays detected regions; press `q` to quit.

### Face Blurring

```bash
python -m blurring_face.main
```

The script detects faces in `blurring_face/images/neymar-face.png` using `blurring_face/model/blaze_face_short_range.tflite` and writes `blurring_face/output/blurred_image.jpg`.

## Troubleshooting

- Run from the repository root so Python can import the project modules. The commands above use module execution for this reason.
- If a NumPy/OpenCV ABI error occurs, confirm that VS Code and the terminal use the same Python 3.11 virtual environment. The current `requirements.txt` pins NumPy 2.x and `opencv-python` 4.6, a combination that can cause this error; changing Python versions alone may not resolve it. MediaPipe also installs the contrib OpenCV package, so avoid installing both OpenCV distributions into one environment.
- For webcam errors, check that the camera is connected and available to other applications.

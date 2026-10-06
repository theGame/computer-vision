[![Python package](https://github.com/theGame/computer-vision/actions/workflows/python-package.yml/badge.svg)](https://github.com/theGame/computer-vision/actions/workflows/python-package.yml)

# Computer Vision Projects

A collection of small computer-vision examples for image classification, webcam color detection, face blurring, and text detection.

## Projects

```text
blurring_face/
    images/neymar-face.png
    main.py
    model/blaze_face_short_range.tflite
    output/
color_capture/
    main.py
image_classification/
    images/empty/
    images/not_empty/
    main.py
text_detection/
    images/images.jpg
    main.py
utils/
    color_helper.py
requirements.txt
```

## Requirements

- Python 3.11 is recommended for this project.
- A webcam is required for the color-capture project.
- A GUI is required to display the text-detection image and the webcam windows.
- Install the packages listed in `requirements.txt`.

## Setup

Run these commands from the repository root.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Select the same `.venv` interpreter in VS Code before running the projects.

## Run a Project

All commands below must be run from the repository root with the virtual environment active.

### Image Classification

Train and evaluate an SVM using the images in `image_classification/images/empty` and `image_classification/images/not_empty`:

```powershell
python -m image_classification.main
```

The script prints the test accuracy and best SVM parameters, then saves the trained classifier as `model.pkl` in the repository root.

### Webcam Color Capture

Run the color detector and choose a color from the terminal picker:

```powershell
python -m color_capture.main
```

The webcam window displays detected regions. Press `q` to close the window.

### Face Blurring

Blur faces detected in the bundled image:

```powershell
python -m blurring_face.main
```

The script uses `blurring_face/model/blaze_face_short_range.tflite` and writes the processed image to `blurring_face/output/blurred_image.jpg`.

### Text Detection

Run EasyOCR against the bundled image:

```powershell
python -m text_detection.main
```

The script reads `text_detection/images/images.jpg`, draws detected text boxes, and displays the result with Matplotlib. Change `gpu=True` in `text_detection/main.py` to `gpu=False` when running without a compatible GPU.

## Notes

- The project package folders contain executable `main.py` modules, so use `python -m <package>.main` rather than running a script by path.
- `image_classification.main` processes every image in the two category folders; it does not automatically validate that the dataset is balanced or representative.
- The current requirements include OpenCV, NumPy, MediaPipe, EasyOCR, scikit-learn, scikit-image, and `pick`. Keep the terminal and VS Code on the same Python environment to avoid package-version and binary-compatibility problems.
- If your camera or display is unavailable, the color-capture and text-detection projects will not run correctly.

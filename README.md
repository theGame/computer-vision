[![Python package](https://github.com/theGame/computer-vision/actions/workflows/python-package.yml/badge.svg)](https://github.com/theGame/computer-vision/actions/workflows/python-package.yml)

# Computer Vision Projects

This repository contains small Python computer-vision examples for image classification, webcam color detection, face blurring, text detection, and TensorFlow model inference.

The examples use bundled sample images and a model exported from Google Teachable Machine. They are intended as learning and demonstration projects rather than production-ready applications.

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
tensor_flow/
    images/
    keras_model.h5
    labels.txt
    main.py
    README.md
utils/
    color_helper.py
requirements.txt
```

## Before you start

- Install Python 3.14 for the current project configuration.
- Use the same Python environment in both VS Code and the terminal.
- A webcam is required for the color-capture project.
- A GUI is required to display the webcam and text-detection results.
- The TensorFlow example currently requires a TensorFlow release that officially supports Python 3.14. The currently installed TensorFlow stack does not support Python 3.14, so the TensorFlow module remains unavailable until that release is available.

## Setup

Run the commands below from the repository root.

### Windows PowerShell

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the following command for the current terminal and then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

After installation, select the `.venv` interpreter in VS Code. Verify the selected interpreter with:

```powershell
python --version
python -c "import tensorflow as tf; print(tf.__version__)"
```

> The TensorFlow module is not currently runnable with Python 3.14 until a TensorFlow release officially supports that version.

## Run a project

All module commands below must be run from the repository root with the virtual environment active.

### Image classification

Train and evaluate an SVM using the images in `image_classification/images/empty` and `image_classification/images/not_empty`:

```powershell
python -m image_classification.main
```

The script prints the test accuracy and best SVM parameters, then saves the trained classifier as `model.pkl` in the repository root.

### Webcam color capture

Run the color detector and select a color from the terminal picker:

```powershell
python -m color_capture.main
```

The webcam window displays detected regions. Press `q` to close the window.

### Face blurring

Blur faces detected in the bundled image:

```powershell
python -m blurring_face.main
```

The script uses `blurring_face/model/blaze_face_short_range.tflite` and writes the processed image to `blurring_face/output/blurred_image.jpg`.

### Text detection

Run EasyOCR against the bundled image:

```powershell
python -m text_detection.main
```

The script reads `text_detection/images/images.jpg`, draws detected text boxes, and displays the result with Matplotlib. When running without a compatible GPU, change `gpu=True` to `gpu=False` in `text_detection/main.py`.

### TensorFlow model inference

The TensorFlow example loads `tensor_flow/keras_model.h5`, reads `tensor_flow/labels.txt`, and classifies the bundled test image:

```powershell
python -m tensor_flow.main
```

The model, labels, and images were exported from Google Teachable Machine:

https://teachablemachine.withgoogle.com/

The code has **not been tested with real data**. The available images are generated or synthetic sample data, so the result should not be treated as validation against real-world observations.

The TensorFlow startup output may include an Abseil logging warning and an oneDNN message. These messages are informational; they do not necessarily indicate that the application failed.

To disable oneDNN custom operations temporarily, run:

```powershell
$env:TF_ENABLE_ONEDNN_OPTS = "0"
python -m tensor_flow.main
```

## Notes for newcomers

- Use module execution such as `python -m <package>.main`; do not run a package module by its file path.
- The image-classification script processes every image in the two category folders. It does not automatically verify that the data is balanced or representative.
- Keep the terminal and VS Code on the same Python environment. Different Python versions and native packages can cause import errors or binary incompatibilities.
- The dependency file includes OpenCV, NumPy, MediaPipe, EasyOCR, scikit-learn, scikit-image, Pillow, Keras, and TensorFlow.
- If the camera, display, model, or dependencies are unavailable, the corresponding project will not run correctly.
- See [tensor_flow/README.md](tensor_flow/README.md) for details about the Teachable Machine model and TensorFlow compatibility.

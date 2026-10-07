# TensorFlow model example

## Data provenance

The model, labels, and images used in this project were generated or exported through Teachable Machine:

https://teachablemachine.withgoogle.com/

The model was taken from Teachable Machine and saved as `keras_model.h5`. The corresponding label file and sample images were also obtained from the exported project data.

This code has **not been tested with real data**. The available images are generated or synthetic sample data, so prediction results should not be treated as validation against real observations.

## TensorFlow and Python compatibility

TensorFlow does not currently support Python 3.14. A TensorFlow release that officially supports the selected Python version must be available before this project can be run reliably with Python 3.14.

Until then, use a Python version supported by the installed TensorFlow release. Do not assume that Python 3.14 is compatible merely because the project imports successfully.

> The oneDNN message shown during TensorFlow startup is informational. It does not indicate that TensorFlow failed to start.

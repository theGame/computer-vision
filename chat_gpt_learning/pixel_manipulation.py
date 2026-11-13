import cv2 as cv
import os
import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))

# Load an image from file
img = cv.imread(os.path.join(current_path, "images", "image.png"))

if img is None:
    print("Could not read the image.")
    raise SystemExit

# Get pixel value at (x=100, y=50)
pixel_value = img[50, 100]
print("Pixel value at (100, 50):", pixel_value)
print("Pixel value data type:", type(pixel_value))

# Change pixel value at (x=100, y=50) to red (BGR format)
img[50, 100] = [0, 0, 255]

# Create a black image with the same dimensions
black_img = np.zeros(img.shape, dtype=img.dtype)
# we could also write like: black_img[:] = 0

# Draw a white rectangle on the black image
image_with_rect = cv.rectangle(black_img, (150, 50), (250, 150), (255, 255, 255), -1)  # Filled rectangle
cv.imwrite(os.path.join(current_path, "images", "image_with_rectangle.png"), image_with_rect)
import cv2 as cv
import os

current_path  = os.path.dirname(os.path.abspath(__file__))

# Load an image from file
image = cv.imread(os.path.join(current_path, "images", 'image.png'))

if image is None:
    print("Could not read the image.")
    raise SystemExit

# Convert the image to grayscale
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply a binary threshold to the grayscale image
_, binary_image = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY_INV)

# Save the binary image to file
cv.imwrite(os.path.join(current_path, "images", "binary_image.png"), binary_image)
import cv2 as cv
import os

current_path  = os.path.dirname(os.path.abspath(__file__))

# Load an image from file
image = cv.imread(os.path.join(current_path, "images", 'image.png'))

if image is None:
    print("Could not read the image.")
    raise SystemExit

# Convert the image to grayscale before thresholding
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
_, binary_image = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
contour_image = image.copy()
cv.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Save the image with contours
cv.imwrite(os.path.join(current_path, "images", 'contours.png'), contour_image)
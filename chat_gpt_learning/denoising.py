import cv2 as cv
import numpy as np
import os

def denoise_image(image, method='gaussian'):
    if method == 'gaussian':
        return cv.GaussianBlur(image, (5, 5), 0)
    elif method == 'median':
        return cv.medianBlur(image, 5)
    elif method == 'bilateral':
        return cv.bilateralFilter(image, 9, 75, 75)
    else:
        raise ValueError(f"Unknown denoising method: {method}")

def add_contours_to_image(image, filter_type=None):
    # 1. Binarize (Otsu picks the threshold automatically)
    _, binary = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # 2. Find contours on the binary image
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 3. Draw on a 3-channel copy so the green line is actually green
    contour_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    cv.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    print(f"{len(contours)} contours. Filter type: {filter_type if filter_type else 'None'}")
    return contour_image

def show_image(title, image):
    cv.imshow(title, image)

path = os.path.dirname(os.path.abspath(__file__))

image_path = os.path.join(path, 'images', 'webcam.jpg')
img = cv.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Failed to load image: {image_path}")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

box = cv.blur(gray, (5, 5))
gaussian = denoise_image(gray, method='gaussian')
median = denoise_image(gray, method='median')
bilat = denoise_image(gray, method='bilateral')

contour_gray = add_contours_to_image(gray)
contour_box = add_contours_to_image(box)
contour_gaussian = add_contours_to_image(gaussian, filter_type='gaussian')
contour_median = add_contours_to_image(median, filter_type='median')
contour_bilat = add_contours_to_image(bilat, filter_type='bilateral')

show_image('Contours on Original Gray Image', contour_gray)
show_image('Contours on Box Filtered Image', contour_box)
show_image('Contours on Gaussian Filtered Image', contour_gaussian)
show_image('Contours on Median Filtered Image', contour_median)
show_image('Contours on Bilateral Filtered Image', contour_bilat)

cv.waitKey(0)
cv.destroyAllWindows()


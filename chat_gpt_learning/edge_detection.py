from os import path

import cv2 as cv
import numpy as np
import os

path = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(path, "images", "highway.jpeg")

img = cv.imread(image_path)
if img is None:
    raise FileNotFoundError(f'Could not read image: {image_path}')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), 0)

edges_50_150 = cv.Canny(blur, 50, 150)
edges_100_200 = cv.Canny(blur, 100, 200)
edges_150_250 = cv.Canny(blur, 150, 250)

# Automatic thresholds based on the image's median brightness
def auto_canny(image, sigma=0.33):
    v = np.median(image)
    low = int(max(0, (1.0 - sigma) * v))
    high = int(min(255, (1.0 + sigma) * v))
    return cv.Canny(image, low, high)

edges_auto = auto_canny(blur)

cv.imshow("edges_50_150", edges_50_150)
cv.imshow("edges_100_200", edges_100_200)
cv.imshow("edges_150_250", edges_150_250)
cv.imshow("auto", edges_auto)

contours, _ = cv.findContours(edges_auto, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
with_contours = cv.drawContours(img, contours, -1, (0, 255, 0), 2)

cv.imshow("with_contours", with_contours)

cv.waitKey(0)
cv.destroyAllWindows()
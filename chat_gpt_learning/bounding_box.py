import cv2 as cv
import os

# Load an image from file
current_path  = os.path.dirname(os.path.abspath(__file__))
image = cv.imread(os.path.join(current_path, "images", 'image.png'))

if image is None: 
    print("Could not read the image.")
    raise SystemExit

# Convert the image to grayscale before thresholding
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
_, binary_image = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)   

# Draw bounding boxes around the contours on the original image
bounding_box_image = image.copy()   
for index, contour in enumerate(contours):    
    if cv.contourArea(contour) < 200:  # Filter out small contours
        continue
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(bounding_box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.putText(bounding_box_image, f"Object {index + 1}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Save the image with bounding boxes
cv.imwrite(os.path.join(current_path, "images", 'bounding_boxes.png'), bounding_box_image)

# load library 
import cv2 as cv
import os

# load image
current_path = os.path.dirname(os.path.abspath(__file__))
img = cv.imread(os.path.join(current_path, "images", "image.png"))

if img is None:
    print("Could not read the image.")
    raise SystemExit


print("Image shape: (height, width, channels)", img.shape)
print("Data type:", img.dtype)

# convert to grayscale
cvt = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print("Grayscale image shape:", cvt.shape)

# resize the image
scale = img.shape[1]/img.shape[0]
new_width = 320
new_height = int(new_width / scale)
resized_img = cv.resize(cvt, (new_width, new_height), interpolation=cv.INTER_AREA)
print("Resized image shape:", resized_img.shape)

# save the resized image
cv.imwrite(os.path.join(current_path, "images", "result.png"), resized_img)

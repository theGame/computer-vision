import cv2
import os
import easyocr
import matplotlib.pyplot as plt

# read image
directory = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(directory, 'images/', 'images.jpg')
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f'Could not read image: {image_path}')

# instance text detector
reader = easyocr.Reader(['en'], gpu=True)  # set gpu=True if you have a compatible GPU

# detect text on image
texts = reader.readtext(img)


# draw bbox and text on image
threshold = 0.25  # confidence threshold
for text in texts:
    bbox, text, confidence = text

    # if the confidence is greater than the threshold, draw the bbox and text
    if confidence > threshold:
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 2)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

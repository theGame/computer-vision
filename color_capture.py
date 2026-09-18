import sys
import pick
import cv2 as cv
from color_helper import get_limits


title = "Select an color that you want to detect: "
colors = ["Yellow", "Green", "Blue", "Red"]

selected_color = pick.pick(colors, title=title)[0]

color_map = {
    "Yellow": [0, 255, 255],
    "Green": [0, 255, 0],
    "Blue": [255, 0, 0],
    "Red": [0, 0, 255]
}

cap = cv.VideoCapture(0)

# Check if the video stream is opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    sys.exit(0)

lower1, upper1, lower2, upper2 = get_limits(color=color_map[selected_color])

try:
    while True:
        ret, frame = cap.read()
        if not ret:
                print("Error: Could not read frame.")
                continue
        
        hsv_image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        mask1 = cv.inRange(hsv_image, lower1, upper1)
        mask2 = cv.inRange(hsv_image, lower2, upper2)

        mask = cv.bitwise_or(mask1, mask2)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours and len(contours) > 0:
            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

        cv.imshow("frame", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Releasing resources...")
    cap.release()
    cv.destroyAllWindows()



    


import cv2 as cv
import os
from ultralytics import YOLO

path = os.path.dirname(os.path.abspath(__file__))

def detect_objects_with_yolo(
        conf: float=0.5, 
        resulted_image_name: str="annotated_highway.jpeg", 
        iou: float=0):
    # Load the YOLO model
    model = YOLO(os.path.join(path, "models", "yolo26n.pt"))

    # Run object detection on an image
    results = list(model(os.path.join(path, "images", "highway.jpeg"), conf=conf, iou=iou))

    # Process the results and draw bounding boxes on the image
    for result in results:
        boxes = getattr(result, "boxes", None)

        print(f"IoU: {iou}, " f"Detections: {len(boxes) if boxes is not None else 0}")
        
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Get the bounding box coordinates
            # x1, y1, x2, y2 = map(int, box.xyxy[0])
            # class_name = model.names[class_id]
            # print(f"Detected <{class_name}>; confidence: {confidence:.2f}; at [{x1}, {y1}, {x2}, {y2}]")

    # Draw bounding boxes on the image
    annotated_image = results[0].plot()
    cv.imwrite(os.path.join(path, "images", resulted_image_name), annotated_image)

# Run the function with different confidence thresholds, investigation
# detect_objects_with_yolo(conf=0.9, resulted_image_name="highway_09.jpeg")
# detect_objects_with_yolo(conf=0.5, resulted_image_name="highway_05.jpeg")
# detect_objects_with_yolo(conf=0.2, resulted_image_name="highway_02.jpeg")

# With different iou values, investigation
detect_objects_with_yolo(conf=0.3, resulted_image_name="highway_iou_03.jpeg", iou=0.3)
detect_objects_with_yolo(conf=0.3, resulted_image_name="highway_iou_05.jpeg", iou=0.5)
detect_objects_with_yolo(conf=0.3, resulted_image_name="highway_iou_08.jpeg", iou=0.8)
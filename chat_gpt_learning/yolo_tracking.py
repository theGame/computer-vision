import os
import cv2 as cv
from ultralytics import YOLO

path = os.path.dirname(os.path.abspath(__file__))

model = YOLO(os.path.join(path, "models", "yolo26n.pt"))

video_path = os.path.join(path, "videos", "highway.mp4")

output_path = os.path.join(path, "videos", "highway_output.mp4")

result = model.track(
    source=video_path,  # Path to the input video
    conf=0.3,  # Confidence threshold for object detection
    iou=0.5,  # IoU threshold for non-max suppression
    tracker="bytetrack.yaml",  # Tracker configuration file
    persist=True,  # Keep tracking objects across frames
    stream=True,  # Stream the output video to a window
)

writer = None  # Initialize the video writer to None

for result in result:
    annotated_frame = result.orig_img  # Get the annotated frame from the result
    if result.boxes is not None and result.boxes.id is not None:
        track_ids = result.boxes.id.int().cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        boxes = result.boxes.xyxy.int().cpu().tolist()

        for track_id, class_id, confidence, box in zip(
            track_ids,
            class_ids,
            confidences,
            boxes
        ):
            class_name = model.names[class_id]

            if class_name != "car":
                continue

            x1, y1, x2, y2 = box

            print(
                f"Track ID: {track_id}, "
                f"Class: {class_name}, "
                f"Confidence: {confidence:.2f}, "
                f"Box: [{x1}, {y1}, {x2}, {y2}]"
            )

    annotated_frame = result.plot()

    if writer is None:
        height, width = annotated_frame.shape[:2]
        fps = 30.0

        writer = cv.VideoWriter(
            output_path,
            cv.VideoWriter_fourcc(*"avc1"),
            fps,
            (width, height)
        )

    writer.write(annotated_frame)

if writer is not None:
    writer.release()
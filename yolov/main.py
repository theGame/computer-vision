from ultralytics import YOLO
import os
import cv2

project_dir = os.path.dirname(os.path.abspath(__file__))

# Load yolov8 model
model_path = os.path.join(project_dir, "model", "yolov8n.pt")
model = YOLO(model_path)

# Load video
video_path = os.path.join(project_dir, "videos", "test.mp4")
cap = cv2.VideoCapture(video_path)

ret = True
# Read frames
while ret:
    ret, frame = cap.read()

    # Detect objects and track objects
    results = model.track(frame, persist=True)

    # Plot results
    frame_ = results[0].plot()

    # Visualize results
    cv2.imshow("YOLOv8 Detection and Tracking", frame_)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

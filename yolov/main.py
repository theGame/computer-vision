from ultralytics import YOLO
import os
import cv2

project_dir = os.path.dirname(os.path.abspath(__file__))

# load yolov8 model
model_path = os.path.join(project_dir, "model", "yolov8n.pt")
model = YOLO(model_path)

# load video
video_path = os.path.join(project_dir, "videos", "test.mp4")
cap = cv2.VideoCapture(video_path)

ret = True
# read frames
while ret:
    ret, frame = cap.read()

    # detect objects and track objects
    results = model.track(frame, persist=True)

    # plot results
    frame_ = results[0].plot()

    #visualize results
    cv2.imshow("YOLOv8 Detection and Tracking", frame_)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

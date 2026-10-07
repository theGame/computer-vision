import os
import cv2
import mediapipe as mp

# prepare output directory
project_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(project_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# read image
image_path = os.path.join(project_dir, 'images', 'neymar-face.png')
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f'Could not read image: {image_path}')
ih, iw, _ = img.shape

# Initialize MediaPipe Face Detection
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the image mode:
options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=os.path.join(project_dir, 'model', 'blaze_face_short_range.tflite')
    ),
    running_mode=VisionRunningMode.IMAGE)

print("Detecting faces in the image...")

with FaceDetector.create_from_options(options) as detector:
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    results = detector.detect(mp_image)

    if results.detections is not None:
        for detection in results.detections:
            bboxC = detection.bounding_box
            x = max(0, bboxC.origin_x)
            y = max(0, bboxC.origin_y)
            right = min(iw, bboxC.origin_x + bboxC.width)
            bottom = min(ih, bboxC.origin_y + bboxC.height)
            w, h = right - x, bottom - y

            if w <= 0 or h <= 0:
                continue
            # Blur face
            face_region = img[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
            img[y:y+h, x:x+w] = blurred_face

# Save image
cv2.imwrite(os.path.join(output_dir, 'blurred_image.jpg'), img)
print(f"Blurred image saved to {os.path.join(output_dir, 'blurred_image.jpg')}")

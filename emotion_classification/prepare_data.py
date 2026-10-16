import os
import cv2
from matplotlib import image
# from utils.face_helper import get_face_landmarks

dataset_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images/')
for emotion_index, emotion in enumerate(os.listdir(dataset_directory)):
    emotion_directory = os.path.join(dataset_directory, emotion)
    if not os.path.isdir(emotion_directory):
        continue

    for image_file in os.listdir(emotion_directory):
        image_path = os.path.join(emotion_directory, image_file)

        image = cv2.imread(image_path)        
        if image is None:
            print(f"Error: Could not read image {image_path}. Skipping.")
            continue

        # landmarks = get_face_landmarks(image)
        print(f"Processed {image_path}.")

        # Here you can save the landmarks or process them as needed
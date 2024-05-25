import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Load known faces and their names
known_faces = []
known_faces_names = []

for file in os.listdir("TrainingImage"):
    image = face_recognition.load_image_file(f"TrainingImage/{file}")
    face_encodings = face_recognition.face_encodings(image)

    # Check if there are multiple faces in the image
    if len(face_encodings) > 0:
        for face_encoding in face_encodings:
            known_faces.append(face_encoding)
            known_faces_names.append(os.path.splitext(file)[0])
    else:
        print(f"No face found in {file}")

print(f"known_faces = {known_faces}"
      f"known_faces_names = {known_faces_names}")





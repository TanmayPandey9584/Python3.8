import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Load known faces and their names
known_faces = []
known_faces_names = []

training_image_path = os.path.join(os.getcwd(), "TrainingImage")
for file in os.listdir(training_image_path):
  image = face_recognition.load_image_file(os.path.join(training_image_path, file))
  face_encodings = face_recognition.face_encodings(image)

  # Check if there are multiple faces in the image
  if len(face_encodings) > 0:
      for face_encoding in face_encodings:
          known_faces.append(face_encoding)
          known_faces_names.append(os.path.splitext(file)[0])
  else:
      print(f"No face found in {file}")
 
'''
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
'''

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
students = known_faces_names.copy()

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        # If a match was found in known_faces, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_faces_names[first_match_index]

        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Mark attendance
    for name in face_names:
        if name in students:
            students.remove(name)
            current_date = datetime.now().strftime("%d/%m/%y")
            current_time = datetime.now().strftime("%H:%M:%S")
            with open("attendance.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, current_time, current_date])

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Attendance Marked")
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
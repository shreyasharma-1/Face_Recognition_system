
import csv
import os
import numpy as np
import cv2
import face_recognition
from datetime import datetime


# Initialize webcam
video_capture = cv2.VideoCapture(0) # 0 is the default camera


# Load and encode known faces
ratan_image = face_recognition.load_image_file("photo/ratan.jpg")
ratan_face_encoding = face_recognition.face_encodings(ratan_image)[0]

bill_image = face_recognition.load_image_file("photo/bill.jpg")
bill_face_encoding = face_recognition.face_encodings(bill_image)[0]

mukesh_image = face_recognition.load_image_file("photo/mukesh.jpg")
mukesh_face_encoding = face_recognition.face_encodings(mukesh_image)[0]

known_face_encodings = [ratan_face_encoding, bill_face_encoding, mukesh_face_encoding]
known_face_names = ["Ratan", "bill", "mukesh"]


students = known_face_names.copy()
# Initialize face data
face_locations = []
face_encodings = []
face_names = []
s = True



now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)


# Start real-time video processing
while True:
    _, frame = video_capture.read()

    # Resize and convert frame to RGB
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    rgb_small_frame = small_frame[:,:,::-1]

    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            if name in known_face_names:
                if name not in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])


    cv2.imshow("Attendance System", frame) #display the frame
    if cv2.waitKey(1) & 0xFF == ord("q"): #press q to quit
        break

video_capture.release() #release the webcam
cv2.destroyAllWindows() #close all windows
f.close() #close the csv file

    
    


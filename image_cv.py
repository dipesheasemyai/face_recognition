import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier("/home/easemyai/Downloads/haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier()
# img = cv2.imread("/home/easemyai/Downloads/Sample-image.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# face = face_classifier.detectMultiScale(gray, 1.3, 5)

# if face is  ():
#     print("No face found.")

# for (x, y, w, h) in face:
#     cv2.rectangle(img, (x,y), (x+w, y+h), (127, 0, 255), 2)
#     cv2.imshow('Face Detection', img)
#     cv2.waitKey(0) 
# cv2.destoryAllWindows()

cap = cv2.VideoCapture("/home/easemyai/Downloads/head-pose-face-detection-male.mp4")

   
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
 
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (127, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()

print("hello world")


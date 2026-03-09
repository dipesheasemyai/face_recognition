from ultralytics import YOLO
import cv2

model = YOLO("/home/easemyai/Documents/cv_project/model/best.pt")

cap = cv2.VideoCapture("/home/easemyai/Documents/cv_project/video_folder/20260227113017.ts")

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, verbose=False)
    frame = results[0].plot()
    
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
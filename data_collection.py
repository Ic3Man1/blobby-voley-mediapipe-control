import cv2
import os

label = "gest" 
gesture_path = f"dataset/gesture"
os.makedirs(gesture_path, exist_ok=True)
no_gesture_path = f"dataset/no_gesture"
os.makedirs(no_gesture_path, exist_ok=True)

cap = cv2.VideoCapture(0)
gesture_count = 0
no_gesture_count = 0

while True:
    ret, frame = cap.read()
    cv2.imshow("Collecting data, to save gesture press 's', to save default pose press 'k', to quit press 'q'", frame)
    
    key = cv2.waitKey(1)
    if key == ord('s'):
        img_name = f"{gesture_path}/{gesture_count}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        gesture_count += 1
    elif key == ord('k'):
        img_name = f"{no_gesture_path}/{no_gesture_count}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        no_gesture_count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
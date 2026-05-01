import cv2
import mediapipe as mp
import csv
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

LABEL = "idle"
FILE_NAME = "hand_gestures.csv"

def normalize_landmarks(landmarks):
    temp_list = []
    base_x, base_y = landmarks[0].x, landmarks[0].y
    
    # 1. Translacja do (0,0) względem nadgarstka
    for lm in landmarks:
        temp_list.append(lm.x - base_x)
        temp_list.append(lm.y - base_y)
        
    # 2. Normalizacja skali
    max_val = max(map(abs, temp_list))
    if max_val > 0:
        temp_list = [n / max_val for n in temp_list]
        
    return temp_list

cap = cv2.VideoCapture(0)
print(f"'s' aby zapisać klatkę, 'q' aby wyjść")

with open(FILE_NAME, mode='a', newline='') as f:
    writer = csv.writer(f)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                data = normalize_landmarks(hand_landmarks.landmark)
                writer.writerow([LABEL] + data)
                print(f"Zapisano próbkę dla: {LABEL}")
            elif key & 0xFF == ord('q'):
                break

        cv2.imshow("Zbieranie danych", frame)

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import pickle
import numpy as np
from evdev import UInput, ecodes as e

ui = UInput({e.EV_KEY: [e.KEY_SPACE, e.KEY_UP, e.KEY_LEFT, e.KEY_RIGHT]})

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

with open('hand_model.pkl', 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

space_pressed = False
up_pressed = False  

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    hand_results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for idx, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                cv2.circle(image, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(image, str(idx), (cx + 5, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                
            features = normalize_landmarks(hand_landmarks.landmark)

            prediction = model.predict([features])[0]
            probability = np.max(model.predict_proba([features]))

            if probability > 0.8:
                cv2.putText(image, f"GEST: {prediction} ({probability:.2f})", 
                            (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                if prediction == "rotate":
                    if not space_pressed:
                        ui.write(e.EV_KEY, e.KEY_SPACE, 1)
                        ui.syn()
                        space_pressed = True
                else:
                    if space_pressed:
                        ui.write(e.EV_KEY, e.KEY_SPACE, 0)
                        ui.syn()
                        space_pressed = False
    else:
        if space_pressed:
            ui.write(e.EV_KEY, e.KEY_SPACE, 0)
            ui.syn()
            space_pressed = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            for idx in [1, 13, 14, 33, 263]:                 # czubek nosta, górna warga, dolna warga, lewe oko, prawe oko
                lm = face_landmarks.landmark[idx]
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(image, (x, y), 5, (0, 255, 255), -1)
                cv2.putText(image, str(idx), (x + 7, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

            # Otwarcie ust -> skok
            upper_lip_y = face_landmarks.landmark[13].y
            lower_lip_y = face_landmarks.landmark[14].y
            mouth_distance = lower_lip_y - upper_lip_y

            if mouth_distance > 0.05:
                if not up_pressed:
                    ui.write(e.EV_KEY, e.KEY_UP, 1)
                    ui.syn()
                    up_pressed = True
                cv2.putText(image, "JUMP", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                if up_pressed:
                    ui.write(e.EV_KEY, e.KEY_UP, 0)
                    ui.syn()
                    up_pressed = False

            # Obrót głowy -> lewo/prawo
            nose_x = face_landmarks.landmark[1].x
            nose_y = face_landmarks.landmark[1].y
            upper_lip_x = face_landmarks.landmark[13].x
            left_eye_x = face_landmarks.landmark[33].x
            right_eye_x = face_landmarks.landmark[263].x

            eye_distance = right_eye_x - left_eye_x

            if eye_distance > 0:
                ratio = (nose_x - left_eye_x) / eye_distance
            else:
                ratio = 0.5

            if ratio < 0.38:
                ui.write(e.EV_KEY, e.KEY_LEFT, 1)
                ui.write(e.EV_KEY, e.KEY_RIGHT, 0)
                ui.syn()
                cv2.putText(image, "LEFT", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif ratio > 0.56:
                ui.write(e.EV_KEY, e.KEY_LEFT, 0)
                ui.write(e.EV_KEY, e.KEY_RIGHT, 1)
                ui.syn()
                cv2.putText(image, "RIGHT", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                ui.write(e.EV_KEY, e.KEY_LEFT, 0)
                ui.write(e.EV_KEY, e.KEY_RIGHT, 0)
                ui.syn()

    cv2.imshow('Blobby Volley Controller', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ui.write(e.EV_KEY, e.KEY_SPACE, 0)
ui.write(e.EV_KEY, e.KEY_UP, 0)
ui.write(e.EV_KEY, e.KEY_LEFT, 0)
ui.write(e.EV_KEY, e.KEY_RIGHT, 0)
ui.syn()

cap.release()
cv2.destroyAllWindows()
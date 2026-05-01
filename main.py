import cv2
import mediapipe as mp
# import pyautogui

# pyautogui.PAUSE = 0

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape

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
                # pyautogui.press('up')
                cv2.putText(image, "JUMP", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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

            if ratio < 0.42:
                # pyautogui.keyDown('left')
                # pyautogui.keyUp('right')
                cv2.putText(image, "LEFT", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif ratio > 0.52:
                # pyautogui.keyDown('right')
                # pyautogui.keyUp('left')
                cv2.putText(image, "RIGHT", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # else:
                # pyautogui.keyUp('left')
                # pyautogui.keyUp('right')

    cv2.imshow('Blobby Volley Controller', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
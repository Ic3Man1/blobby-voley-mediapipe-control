import cv2
import mediapipe as mp
import pyautogui

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    results = face_mesh.process(image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            upper_lip_y = face_landmarks.landmark[13].y
            lower_lip_y = face_landmarks.landmark[14].y
            mouth_distance = lower_lip_y - upper_lip_y
            
            if mouth_distance > 0.05:
                pyautogui.press('up')
                cv2.putText(image, "JUMP!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  
            nose_x = face_landmarks.landmark[1].x
            
            if nose_x < 0.4:
                pyautogui.keyDown('left')
                pyautogui.keyUp('right')
                cv2.putText(image, "LEFT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif nose_x > 0.6:
                pyautogui.keyDown('right')
                pyautogui.keyUp('left')
                cv2.putText(image, "RIGHT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                pyautogui.keyUp('left')
                pyautogui.keyUp('right')

    cv2.imshow('Blobby Volley Controller', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pyautogui.keyUp('left')
pyautogui.keyUp('right')
import cv2
import os

label = "gest" # Zmień na "tlo" dla drugiej serii
save_path = f"dataset/{label}"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    cv2.imshow("Zbieranie danych - naciśnij 's' by zapisac, 'q' by wyjsc", frame)
    
    key = cv2.waitKey(1)
    if key == ord('s'):
        img_name = f"{save_path}/{count}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Zapisano: {img_name}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
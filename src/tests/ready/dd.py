import cv2
import numpy as np
import mediapipe as mp

# Ініціалізуємо відстеження руки
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Створюємо "полотно" для малювання (чорний фон)
canvas = np.zeros((400, 400, 3), dtype=np.uint8)

# Змінні для зберігання останньої точки
prev_x, prev_y = None, None

# Запускаємо веб-камеру
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Відображаємо зображення (дзеркальний ефект)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Виявлення руки
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            if prev_x is not None and prev_y is not None:
                # Малюємо товсту лінію на полотні
                cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 20)  # Збільшена товщина лінії
            
            prev_x, prev_y = x, y

    # Відображаємо об'єднаний екран
    frame[0:400, 0:400] = canvas
    cv2.imshow("Малювання пальцем", frame)

    key = cv2.waitKey(1)
    if key != -1:
        print(key)
    # Очистити полотно (клавіша "c")
    if key == 99:
        canvas[:] = 0

    # Зберегти малюнок і вийти (клавіша "s")
    if key == 115:
        break

cap.release()
cv2.destroyAllWindows()

# Перетворюємо зображення для нейронної мережі
gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)  # Перетворюємо в градації сірого
resized_canvas = cv2.resize(gray_canvas, (28, 28), interpolation=cv2.INTER_NEAREST)  # Змінюємо розмір на 28x28
flattened_array = resized_canvas.flatten()  # Перетворюємо в 1D масив

s = ','.join(map(str, flattened_array))  # Готовий масив для подачі в нейронну мережу
print(s)

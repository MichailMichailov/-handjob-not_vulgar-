import cv2
import numpy as np
import mediapipe as mp

# Инициализируем отслеживание руки
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Создаем "холст" для рисования (черный фон)
canvas = np.zeros((400, 400, 3), dtype=np.uint8)

# Переменные для хранения последней точки
prev_x, prev_y = None, None

# Запускаем веб-камеру
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Отражаем изображение (зеркальный эффект)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Обнаружение руки
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            if prev_x is not None and prev_y is not None:
                # Рисуем жирную линию на холсте
                cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 20)  # Увеличенная толщина линии
            
            prev_x, prev_y = x, y

    # Отображаем объединенный экран
    frame[0:400, 0:400] = canvas
    cv2.imshow("Draw with Finger", frame)

    key = cv2.waitKey(1)
    if key!=-1:
        print(key)
    # Очистить холст (клавиша "c")
    if key == 99:
        canvas[:] = 0

    # Сохранить рисунок и выйти (клавиша "s")
    if key == 115:
        break

cap.release()
cv2.destroyAllWindows()

# Преобразуем изображение для нейросети
gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)  # Переводим в градации серого
resized_canvas = cv2.resize(gray_canvas, (28, 28), interpolation=cv2.INTER_NEAREST)  # Меняем размер на 28x28
flattened_array = resized_canvas.flatten()  # Превращаем в 1D массив

s = ','.join(map(str, flattened_array))  # Готовый массив для подачи в нейросеть
print(s)

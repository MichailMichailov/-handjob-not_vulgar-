import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

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
                cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 20)
            
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

# Преобразуем изображение для выделения букв
gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)  # Переводим в градации серого
ret, thresh = cv2.threshold(gray_canvas, 0, 255, cv2.THRESH_BINARY)
img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

# Получаем контуры
contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
letters = []

for idx, contour in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(contour)
    if hierarchy[0][idx][3] == -1:  # Проверяем, что это внешний контур
        letter_crop = gray_canvas[y:y + h, x:x + w]
        size_max = max(w, h)
        letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
        
        if w > h:
            y_pos = size_max // 2 - h // 2
            letter_square[y_pos:y_pos + h, 0:w] = letter_crop
        elif w < h:
            x_pos = size_max // 2 - w // 2
            letter_square[0:h, x_pos:x_pos + w] = letter_crop
        else:
            letter_square = letter_crop
        
        letters.append((x, cv2.resize(letter_square, (28, 28), interpolation=cv2.INTER_AREA)))

# Сортируем буквы по их позиции на холсте
letters.sort(key=lambda x: x[0])
letters_array = [letter[1] for letter in letters]  # Массив изображений букв (28x28)

# Отображение всех букв с помощью matplotlib
fig, axes = plt.subplots(1, len(letters_array), figsize=(len(letters_array) * 2, 2))
if len(letters_array) == 1:
    axes = [axes]
for ax, letter_img in zip(axes, letters_array):
    ax.imshow(letter_img, cmap='gray')
    ax.axis("off")
plt.show()

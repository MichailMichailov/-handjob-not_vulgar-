# Імпортуємо OpenCV для роботи із зображеннями
import cv2
# Імпортуємо NumPy для роботи з масивами
import numpy as np
# Імпортуємо Matplotlib для візуалізації деяких зображень
from matplotlib import pyplot as plt
# Імпортуємо типи для анотації функцій
from typing import List, Dict, Tuple, Set

class DrawingCanvas:
    """Клас для роботи з полотном для малювання"""

    def __init__(self, width: int, height: int):
        """
        Ініціалізація полотна.
        
        :param width: Ширина полотна.
        :param height: Висота полотна.
        """
        # Створюємо чорне полотно
        self.canvas = np.zeros((width, height, 3), dtype=np.uint8)
        # Координати попередньої точки для малювання лінії
        self.prev_x = None
        self.prev_y = None

    def draw_line(self, x: int, y: int, color: Tuple[int, int, int] = (255, 255, 255), size: int = 20):
        """
        Малює лінію на полотні від попередньої точки до поточної.
        
        :param x: Координата X нової точки.
        :param y: Координата Y нової точки.
        :param color: Колір лінії (за замовчуванням білий).
        :param size: Товщина лінії (за замовчуванням 20).
        """
        # Якщо попередні координати існують
        if self.prev_x is not None and self.prev_y is not None:
                # Малюємо лінію
                cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), color, size)
        self.prev_x, self.prev_y = x, y # Оновлюємо попередні координати

    def clear(self):
        """Очищає полотно, заповнюючи його чорним кольором"""
        self.canvas[:] = 0
    
    def clear_prev(self):
        """Скидає збережені попередні координати"""
        self.prev_x = None
        self.prev_y = None

    def get_canvas(self) -> np.ndarray:
        """Повертає поточне зображення полотна"""
        return self.canvas

    def get_single_letter(self)  -> np.ndarray:
        """
        Перетворює полотно у чорно-біле зображення розміром 28x28 для роботи з нейромережею.
        
        :return: 1D масив (flattened) пікселів зображення 28x28.
        """
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)  # Перетворюємо у градації сірого
        resized_canvas = cv2.resize(gray_canvas, (28, 28), interpolation=cv2.INTER_NEAREST)  # Змінюємо розмір на 28x28
        flattened_array = resized_canvas.flatten()  # Перетворюємо у 1D масив
        return flattened_array
    
    def print_data_array(self, flattened_array: np.ndarray):
        """
        (Допоміжна функція) Виводить одномірний масив пікселів у вигляді рядка, розділеного комами.
        
        :param flattened_array: Одновимірний масив пікселів.
        """
        s = ','.join(map(str, flattened_array))  # Готовий масив для подачі у нейромережу
        print(s)
    
    def get_several_letters(self) -> List[np.ndarray]:
        """
        Виділяє кілька букв з полотна, обрізає їх і масштабує до 28x28 пікселів.
        
        :return: Список зображень букв у форматі 28x28.
        """
        # Перетворюємо зображення для виділення букв
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)  # Перетворюємо у градації сірого
        ret, thresh = cv2.threshold(gray_canvas, 0, 255, cv2.THRESH_BINARY) # Бінаризація
        img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1) # Видаляємо шуми
        # Отримуємо контури
        contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        letters = []

        for idx, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour) # Визначаємо координати та розміри букви
            if hierarchy[0][idx][3] == -1:  # Перевіряємо, що це зовнішній контур
                letter_crop = gray_canvas[y:y + h, x:x + w] # Вирізаємо букву
                size_max = max(w, h) # Визначаємо максимальну сторону квадрата
                # Створюємо квадратне зображення букви
                letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
                
                if w > h: # Якщо буква ширша, ніж вища
                    y_pos = size_max // 2 - h // 2
                    letter_square[y_pos:y_pos + h, 0:w] = letter_crop
                elif w < h: # Якщо буква вища, ніж ширша
                    x_pos = size_max // 2 - w // 2
                    letter_square[0:h, x_pos:x_pos + w] = letter_crop
                else: # Якщо буква квадратна
                    letter_square = letter_crop
                # Додаємо букву у список (x-координата + саме зображення)
                letters.append((x, cv2.resize(letter_square, (28, 28), interpolation=cv2.INTER_AREA)))
        # Сортуємо букви за їх позицією на полотні
        letters.sort(key=lambda x: x[0])
        letters_array = [letter[1] for letter in letters]  # Масив зображень букв (28x28)
        return letters_array

    def show_leters(self, letters_array: List[np.ndarray]):
        """
        (Допоміжна функція) Відображає вирізані букви з полотна за допомогою Matplotlib.
        
        :param letters_array: Список зображень букв розміром 28x28.
        """
        fig, axes = plt.subplots(1, len(letters_array), figsize=(len(letters_array) * 2, 2))
        if len(letters_array) == 1:  # Якщо одна буква, перетворюємо axes у список
            axes = [axes]
        # Перебираємо букви
        for ax, letter_img in zip(axes, letters_array):
            # Відображаємо зображення у відтінках сірого
            ax.imshow(letter_img, cmap='gray')
            ax.axis("off") # Видаляємо осі
        plt.show() # Показуємо результат

# Импортируем OpenCV для работы с изображениями
import cv2
# Импортируем NumPy для работы с массивами
import numpy as np
# Импортируем Matplotlib для визуализации некоторых изображений
from matplotlib import pyplot as plt
# Импортируем типы для аннотации функций
from typing import List, Dict, Tuple, Set

class DrawingCanvas:
    """Класс для работы с холстом для рисования"""

    def __init__(self, width: int, height: int):
        """
        Инициализация холста.
        
        :param width: Ширина холста.
        :param height: Высота холста.
        """
        # Создаём чёрный холст
        self.canvas = np.zeros((width, height, 3), dtype=np.uint8)
        # Координаты предыдущей точки для рисования линии
        self.prev_x = None
        self.prev_y = None

    def draw_line(self, x: int, y: int, color: Tuple[int, int, int] = (255, 255, 255), size: int = 20):
        """
        Рисует линию на холсте от предыдущей точки до текущей.
        
        :param x: Координата X новой точки.
        :param y: Координата Y новой точки.
        :param color: Цвет линии (по умолчанию белый).
        :param size: Толщина линии (по умолчанию 20).
        """
        # Если предыдущие координаты существуют
        if self.prev_x is not None and self.prev_y is not None:
                # Рисуем линию
                cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), color, size)
        self.prev_x, self.prev_y = x, y # Обновляем предыдущие координаты

    def clear(self):
        """Очищает холст, заполняя его чёрным цветом"""
        self.canvas[:] = 0
    
    def clear_prev(self):
        """Сбрасывает сохранённые предыдущие координаты"""
        self.prev_x = None
        self.prev_y = None

    def get_canvas(self) -> np.ndarray:
        """Возвращает текущее изображение холста"""
        return self.canvas

    def get_single_letter(self)  -> np.ndarray:
        """
        Преобразует холст в чёрно-белое изображение размером 28x28 для работы с нейросетью.
        
        :return: 1D массив (flattened) пикселей изображения 28x28.
        """
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)  # Переводим в градации серого
        resized_canvas = cv2.resize(gray_canvas, (28, 28), interpolation=cv2.INTER_NEAREST)  # Меняем размер на 28x28
        flattened_array = resized_canvas.flatten()  # Превращаем в 1D массив
        return flattened_array
    
    def print_data_array(self, flattened_array: np.ndarray):
        """
        (Вспомогательная функция) Выводит одномерный массив пикселей в виде строки, разделённой запятыми.
        
        :param flattened_array: Одномерный массив пикселей.
        """
        s = ','.join(map(str, flattened_array))  # Готовый массив для подачи в нейросеть
        print(s)
    
    def get_several_letters(self) -> List[np.ndarray]:
        """
        Выделяет несколько букв с холста, обрезает их и масштабирует до 28x28 пикселей.
        
        :return: Список изображений букв в формате 28x28.
        """
        # Преобразуем изображение для выделения букв
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)  # Переводим в градации серого
        ret, thresh = cv2.threshold(gray_canvas, 0, 255, cv2.THRESH_BINARY) # Бинаризация
        img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1) # Убираем шумы
        # Получаем контуры
        contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        letters = []

        for idx, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour) # Определяем координаты и размеры буквы
            if hierarchy[0][idx][3] == -1:  # Проверяем, что это внешний контур
                letter_crop = gray_canvas[y:y + h, x:x + w] # Вырезаем букву
                size_max = max(w, h) # Определяем максимальную сторону квадрата
                # Создаём квадратное изображение буквы
                letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
                
                if w > h: # Если буква шире, чем выше
                    y_pos = size_max // 2 - h // 2
                    letter_square[y_pos:y_pos + h, 0:w] = letter_crop
                elif w < h: # Если буква выше, чем шире
                    x_pos = size_max // 2 - w // 2
                    letter_square[0:h, x_pos:x_pos + w] = letter_crop
                else: # Если буква квадратная
                    letter_square = letter_crop
                # Добавляем букву в список (x-координата + само изображение)
                letters.append((x, cv2.resize(letter_square, (28, 28), interpolation=cv2.INTER_AREA)))
        # Сортируем буквы по их позиции на холсте
        letters.sort(key=lambda x: x[0])
        letters_array = [letter[1] for letter in letters]  # Массив изображений букв (28x28)
        return letters_array

    def show_leters(self, letters_array: List[np.ndarray]):
        """
        (Вспомогательная функция) Отображает вырезанные буквы с холста с помощью Matplotlib.
        
        :param letters_array: Список изображений букв размером 28x28.
        """
        fig, axes = plt.subplots(1, len(letters_array), figsize=(len(letters_array) * 2, 2))
        if len(letters_array) == 1:  # Если одна буква, превращаем axes в список
            axes = [axes]
        # Реребераем буквы
        for ax, letter_img in zip(axes, letters_array):
            # Отображаем изображение в оттенках серого
            ax.imshow(letter_img, cmap='gray')
            ax.axis("off") # Убираем оси
        plt.show() # Показываем результат


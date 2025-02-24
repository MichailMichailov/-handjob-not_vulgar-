import cv2
from matplotlib import pyplot as plt
import numpy as np

class DrawingCanvas:
    def __init__(self, width, height):
        self.canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        self.prev_x = None
        self.prev_y = None

    def draw_line(self, x, y, color=(255, 255, 255), size=20):
        """Рисует линию на холсте"""
        if self.prev_x is not None and self.prev_y is not None:
                cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), color, size)
        self.prev_x, self.prev_y = x, y

    def clear(self):
        """Очищает холст"""
        self.canvas[:] = 0
    
    def clear_prev(self):
         self.prev_x = None
         self.prev_y = None

    def get_canvas(self):
        """Возвращает холст"""
        return self.canvas

    def get_single_letter(self):
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)  # Переводим в градации серого
        resized_canvas = cv2.resize(gray_canvas, (28, 28), interpolation=cv2.INTER_NEAREST)  # Меняем размер на 28x28
        flattened_array = resized_canvas.flatten()  # Превращаем в 1D массив
        return flattened_array
    
    def print_data_array(self, flattened_array):
        s = ','.join(map(str, flattened_array))  # Готовый массив для подачи в нейросеть
        print(s)
    
    def get_several_letters(self):
        # Преобразуем изображение для выделения букв
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)  # Переводим в градации серого
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
        return letters_array

    def show_leters(self, letters_array):
        '''Отображение всех букв с помощью matplotlib'''
        fig, axes = plt.subplots(1, len(letters_array), figsize=(len(letters_array) * 2, 2))
        if len(letters_array) == 1:
            axes = [axes]
        for ax, letter_img in zip(axes, letters_array):
            ax.imshow(letter_img, cmap='gray')
            ax.axis("off")
        plt.show()


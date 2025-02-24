
# from canva.AirWritingApp import AirWritingApp
# if __name__ == "__main__":
#     app = AirWritingApp()
#     app.run()

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Dense, Input  # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from data.config import train_data_folder
import matplotlib.pyplot as plt

# Определение имен файлов
train_file = f"{train_data_folder}\\emnist-byclass-train.csv"
test_file = f"{train_data_folder}\\emnist-byclass-test.csv"
model_file = f"{train_data_folder}\\trained_model.h5"

# Функция загрузки данных
def load_data(data_path):
    """Загружает и подготавливает данные EMNIST"""
    df = pd.read_csv(data_path, header=None)  # Загружаем без преобразования в строки
    y = df.iloc[:, 0].values  # Метки классов (первый столбец)
    X = df.iloc[:, 1:].values  # Остальные столбцы - пиксели
    for x in X:
        image = x.reshape(28, 28).T  # Преобразуем в 28x28 (EMNIST использует этот размер)
        plt.imshow(image, cmap='gray')
        plt.title(f"Тестовое изображение, метка: {y[i]}")
        plt.show()
    return X, y

# Загрузка данных
print('loadData1')
X_train, y_train = load_data(train_file)
print('loadtest1')
X_test, y_test = load_data(test_file)

# Нормализация данных
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
print(f"Размер X_train: {X_train.shape}, Размер y_train: {y_train.shape}")
print(f"Уникальные классы: {np.unique(y_train)}")
# Определение модели
if os.path.exists(model_file):
    print("Загружаем ранее обученную модель...")
    model = load_model(model_file)
else:
    print("Обучаем новую модель...")
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(62, activation='softmax')  # 62 класса для EMNIST
    ])
    print('0')
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print('1')
    # Обучение модели
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    print('2')
    # Сохранение модели
    model.save(model_file)
    print(f"Модель сохранена в файл {model_file}")

# # Оценка модели
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Тестовый loss: {loss}, Точность: {accuracy}")

coordinates = [(187, 218), (190, 220), (193, 221), (194, 222), (194, 222), (194, 223), (195, 223), (194, 223), (194, 223), (194, 222), (194, 221), (195, 221), (196, 220), (197, 219), (197, 219), (203, 195), (214, 174), (217, 158), (219, 159), (230, 133), (237, 119), (236, 119), (263, 86), (266, 70), (277, 56), (276, 56), (281, 51), (288, 40), (287, 41), (298, 24), (303, 21), (308, 15), (308, 15), (310, 13), (312, 14), (312, 16), (315, 25), (321, 35), (324, 41), (324, 41), (327, 49), (334, 67), (338, 76), (354, 105), (358, 116), (361, 125), (361, 126), (366, 143), (369, 162), (371, 172), (375, 193), (378, 198), (380, 201), (381, 200), (381, 200), (381, 198), (380, 193), (375, 184), (376, 181), (372, 172), (372, 172), (371, 167), (369, 162), (360, 142), (359, 135), (356, 127), (356, 126), (355, 123), (352, 120), (346, 113), (346, 111), (345, 110), (346, 110), (345, 110), (343, 111), (337, 112), (337, 112), (334, 113), (329, 114), (324, 116), (312, 120), (283, 126), (284, 125), (270, 128), (269, 128), (263, 129), (256, 129), (236, 133), (232, 134), (229, 134), (224, 135), (220, 136), (215, 137), (213, 135), (213, 135), (214, 135), (214, 134), (215, 134), (215, 134), (217, 134), (218, 134), (218, 134), (218, 134), (218, 134), (218, 135), (218, 135), (219, 135), (219, 135), (221, 134), (221, 133), (222, 133), (224, 132), (226, 132), (229, 130), (230, 131), (231, 130), (231, 130), (231, 130)]

# def points_to_image(points, img_size=(28, 28)):
#     """ Преобразует список точек в изображение. """
#     img = np.zeros(img_size)  # Создаем пустое изображение 28x28 пикселей
#     for (x, y) in points:
#         if 0 <= x < img_size[0] and 0 <= y < img_size[1]:
#             img[y, x] = 1  # Помечаем пиксели, соответствующие точкам
#     return img
# image = points_to_image(coordinates, img_size=(28, 28))
# plt.imshow(image, cmap='gray')
# plt.title("Тестовое изображение")
# plt.show()
# Нормализация данных. Делим каждую координату на 255 для масштабирования.
coordinates = np.array(coordinates)
coordinates_normalized = coordinates / 255.0  # Приводим координаты к диапазону [0, 1]

# Преобразуем в одномерный вектор
flattened_data = coordinates_normalized.flatten()

# Дополняем или усечем данные до 784 элементов
flattened_data = flattened_data[:784]  # Урезаем, если элементов больше 784
flattened_data = np.pad(flattened_data, (0, 784 - len(flattened_data)), mode='constant')  # Дополняем нулями, если элементов меньше 784
# Добавляем размерность, чтобы передать в модель (batch size = 1)
flattened_data = np.expand_dims(flattened_data, axis=0)
# Предсказание
predictions = model.predict(flattened_data)

# Определение класса с наибольшей вероятностью
predicted_class = np.argmax(predictions, axis=1)
print(f"Предсказанный класс: {predicted_class[0]}")

# Классы от 0 до 9 — это цифры 0–9.
# Классы от 10 до 35 — это буквы латинского алфавита от A до Z.
# Классы от 36 до 61 — это дополнительные символы или расширенные буквы, если они имеются в наборе.
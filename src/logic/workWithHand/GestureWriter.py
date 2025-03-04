import os  # Модуль для работы с операционной системой (файлы, пути и т.д.)
import cv2  # Библиотека для работы с компьютерным зрением
import numpy as np  # Библиотека для работы с массивами и матрицами данных
import pandas as pd  # Библиотека для работы с данными в формате таблиц
import matplotlib.pyplot as plt  # Модуль для создания графиков и визуализации данных
from PIL import Image, ImageDraw, ImageFilter  # Библиотека для работы с изображениями
# Модуль для загрузки и создания нейросетевых моделей
from tensorflow.keras.models import Sequential, load_model # type: ignore
# Слои для сверточных нейросетей
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
# Функция для преобразования меток в категориальные значения
from tensorflow.keras.utils import to_categorical # type: ignore
from data.config import train_data_folder, emnist_labels  # Пути к данным
# Импортируем типы для аннотации функций
from typing import List, Dict, Tuple, Set


class GestureWriter:
    """Класс для работы с нейросетью"""
    def __init__(self, training_data_folder: str = None):
        """
        Инициализирует объект GestureWriter.
        
        Параметры:
        - training_data_folder (str, опционально): Путь к папке с обучающими данными.
        Загружает или обучает модель на основе доступных данных.
        """
        self.model = None # Модель нейросети (по умолчанию None)
        self.get_model(training_data_folder)  # Загружаем или обучаем модель
        # self.test_model(training_data_folder)  # Проверка модели (раскомментировать по необходимости)

    def get_model(self, training_data_folder: str):
        """
        Загружает или обучает KNN-модель
        
        Параметры:
        - training_data_folder (str): Путь к папке с обучающими данными.
        """
        # Путь к файлу модели
        model_file = f"{training_data_folder}\\emnist_cnn_model.h5"
        # Проверяем, существует ли сохраненная модель
        if os.path.exists(model_file):
            print("Загружаем ранее обученную модель...")
            self.model = load_model(model_file) # Загружаем модель
        else:
            print("Обучаем новую модель...")
            # Путь к файлу обучающих данных
            train_file = f"{train_data_folder}\\emnist-byclass-train.csv"
            # Путь к файлу тестовых данных
            test_file = f"{train_data_folder}\\emnist-byclass-test.csv"
            model_file = "emnist_cnn_model.h5" # Путь для сохранения модели
            # Загружаем данные для обучения
            X_train, y_train = self.load_emnist_data(train_file)
            # Загружаем данные для тестирования
            X_test, y_test = self.load_emnist_data(test_file)
            num_classes = len(set(y_train)) # Количество классов (уникальных меток)
            # Преобразуем метки в категориальные
            y_train_categorical = to_categorical(y_train, num_classes=num_classes)
            y_test_categorical = to_categorical(y_test, num_classes=num_classes)
            # Создаем модель
            self.model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Первый сверточный слой
                MaxPooling2D((2, 2)), # Слой подвыборки
                Conv2D(64, (3, 3), activation='relu'),  # Второй сверточный слой
                MaxPooling2D((2, 2)),  # Слой подвыборки
                Flatten(),  # Преобразуем данные в одномерный вектор
                Dense(128, activation='relu'), # Полносвязный слой
                Dropout(0.5), # Слой регуляризации (пропуск половины нейронов)
                Dense(num_classes, activation='softmax') # Выходной слой с количеством нейронов = количеству классов
            ])
            # Компиляция модели
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print("Начинаем обучение...")
            # Обучаем модель
            self.model.fit(X_train, y_train_categorical, epochs=10, batch_size=128, validation_data=(X_test, y_test_categorical)) 
            # Сохраняем модель
            self.model.save("emnist_cnn_model.h5")
            print(f"Модель сохранена в файл {model_file}")

    def test_model(self, training_data_folder: str):
        """
        Тестирует модель на тестовых данных.

        Параметры:
        - training_data_folder (str): Путь к папке с тестовыми данными.
        """
        # Путь к файлу тестовых данных
        test_file = f"{training_data_folder}\\emnist-byclass-test.csv"
        X_test, y_test = self.load_emnist_data(test_file)
         # Оценка модели на тестовых данных
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Тестовый loss: {loss}, Точность: {accuracy}")

    def load_emnist_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Загружает и подготавливает данные EMNIST.

        Параметры:
        - data_path (str): Путь к CSV файлу с данными.
        
        Возвращает:
        - X (np.ndarray): Входные данные (изображения).
        - y (np.ndarray): Метки классов.
        """
        print(f'Загрузка данных из {data_path}...')
        df = pd.read_csv(data_path, header=None) # Чтение данных из CSV файла
        y = df.iloc[:, 0].values  # Метки классов (первый столбец)
        X = df.iloc[:, 1:].values  # Остальные столбцы - пиксели
        X = X.reshape(-1, 28, 28, 1).astype("float32") / 255.0  # Нормализация
        return X, y
    
    def show_sample_images(self, X: np.ndarray, y: np.ndarray, number: int, num_samples: int = 5):
        """
        (Вспомогательная функция) Визуализирует несколько изображений из датасета.

        Параметры:
        - X (np.ndarray): Массив изображений.
        - y (np.ndarray): Массив меток классов.
        - number (int): Метка класса для фильтрации.
        - num_samples (int): Количество выводимых изображений.
        """
        t = 0 # Счётчик выводимых изображений
        for i in range(len(X)):
            if(y[i] == number): # Если метка изображения совпадает с заданной
                image = X[i].reshape(28, 28).T  # Преобразуем в 28x28 (EMNIST использует этот размер)
                plt.imshow(image, cmap='gray') # Отображаем изображение
                # Заголовок с меткой
                plt.title(f"Тестовое изображение, метка: {y[i]} - {emnist_labels[y[i]]}")
                plt.show() # Показываем изображение
                t += 1
            if t == num_samples: # Если выведено нужное количество изображений
                break

    def recognize_letter(self, data: List[int]) -> Tuple[str, float]:
        """
        Распознает одиночную букву из траектории.

        Параметры:
        - data (List[int]): Данные траектории (список точек).
        
        Возвращает:
        - Tuple[str, float]: Распознанную букву и уверенность в распознавании.
        """
        # Прогнозируем метку для изображения
        prediction = self.model.predict(self.normalize_image(data))
        predicted_class = np.argmax(prediction)# Находим индекс наиболее вероятного класса
        confidence = np.max(prediction)  # Уверенность в прогнозе
        return emnist_labels[predicted_class], confidence # Возвращаем букву и уверенность

    def normalize_image(self, data: List[int]) -> np.ndarray:
        """
        Преобразует траекторию в изображение.

        Параметры:
        - data (List[int]): Данные траектории.
        
        Возвращает:
        - np.ndarray: Нормализованное изображение.
        """
        image_data = np.array(data) # Преобразуем данные в numpy массив
        image_data = image_data.reshape(28, 28) # Преобразуем в размер 28x28
        image_data = image_data.astype("float32") / 255.0 # Нормализуем изображение
        image_data = np.expand_dims(image_data, axis=-1) # Добавляем размерность для канала
        image_data = np.expand_dims(image_data, axis=0)  # Добавляем размерность для партии
        return image_data.T  # Возвращаем транспонированное изображение
    
    def recognize_letters(self, data: List[List[int]]) -> Tuple[str, int]:
        """
        Распознает несколько букв из траектории.

        Параметры:
        - data (List[List[int]]): Список траекторий (каждая траектория — список точек).
        
        Возвращает:
        - Tuple[str, int]: Строку с распознанными буквами и индекс ошибки (по умолчанию 0).
        """
        s = ''
        for e in data:
            s+=self.recognize_letter(e)[0]  # Распознаем каждую букву и добавляем в строку
        return s, 0 # Возвращаем строку и 0 чтобы ответ был похож на recognize_letter
        

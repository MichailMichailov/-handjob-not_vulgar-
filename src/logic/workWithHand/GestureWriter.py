import os  # Модуль для роботи з операційною системою (файли, шляхи тощо)
import cv2  # Бібліотека для роботи з комп'ютерним зором
import numpy as np  # Бібліотека для роботи з масивами та матрицями даних
import pandas as pd  # Бібліотека для роботи з даними у форматі таблиць
import matplotlib.pyplot as plt  # Модуль для створення графіків і візуалізації даних
from PIL import Image, ImageDraw, ImageFilter  # Бібліотека для роботи із зображеннями
# Модуль для завантаження та створення нейромережевих моделей
from tensorflow.keras.models import Sequential, load_model # type: ignore
# Шари для згорткових нейромереж
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
# Функція для перетворення міток у категоріальні значення
from tensorflow.keras.utils import to_categorical # type: ignore
from data.config import train_data_folder, emnist_labels  # Шляхи до даних
# Імпортуємо типи для анотації функцій
from typing import List, Tuple


class GestureWriter:
    """Клас для роботи з нейромережею"""
    def __init__(self, training_data_folder: str = None):
        """
        Ініціалізує об'єкт GestureWriter.
        
        Параметри:
        - training_data_folder (str, опціонально): Шлях до папки з навчальними даними.
        Завантажує або навчає модель на основі доступних даних.
        """
        self.model = None # Модель нейромережі (за замовчуванням None)
        self.get_model(training_data_folder)  # Завантажуємо або навчаємо модель
        # self.test_model(training_data_folder)  # Перевірка моделі (розкоментувати за потреби)

    def get_model(self, training_data_folder: str):
        """
        Завантажує або навчає KNN-модель
        
        Параметри:
        - training_data_folder (str): Шлях до папки з навчальними даними.
        """
        # Шлях до файлу моделі
        model_file = f"{training_data_folder}\\emnist_cnn_model.h5"
        # Перевіряємо, чи існує збережена модель
        if os.path.exists(model_file):
            print("Завантажуємо раніше навчану модель...")
            self.model = load_model(model_file) # Завантажуємо модель
        else:
            print("Навчаємо нову модель...")
            # Шлях до файлу навчальних даних
            train_file = f"{train_data_folder}\\emnist-byclass-train.csv"
            # Шлях до файлу тестових даних
            test_file = f"{train_data_folder}\\emnist-byclass-test.csv"
            model_file = "emnist_cnn_model.h5" # Шлях для збереження моделі
            # Завантажуємо дані для навчання
            X_train, y_train = self.load_emnist_data(train_file)
            # Завантажуємо дані для тестування
            X_test, y_test = self.load_emnist_data(test_file)
            num_classes = len(set(y_train)) # Кількість класів (унікальних міток)
            # Перетворюємо мітки в категоріальні
            y_train_categorical = to_categorical(y_train, num_classes=num_classes)
            y_test_categorical = to_categorical(y_test, num_classes=num_classes)
            # Створюємо модель
            self.model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Перший згортковий шар
                MaxPooling2D((2, 2)), # Шар підвибірки
                Conv2D(64, (3, 3), activation='relu'),  # Другий згортковий шар
                MaxPooling2D((2, 2)),  # Шар підвибірки
                Flatten(),  # Перетворюємо дані в одномірний вектор
                Dense(128, activation='relu'), # Повнозв'язний шар
                Dropout(0.5), # Шар регуляризації (відключення половини нейронів)
                Dense(num_classes, activation='softmax') # Вихідний шар з кількістю нейронів = кількості класів
            ])
            # Компіляція моделі
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print("Починаємо навчання...")
            # Навчаємо модель
            self.model.fit(X_train, y_train_categorical, epochs=10, batch_size=128, validation_data=(X_test, y_test_categorical)) 
            # Зберігаємо модель
            self.model.save("emnist_cnn_model.h5")
            print(f"Модель збережено у файл {model_file}")

    def test_model(self, training_data_folder: str):
        """
        Тестує модель на тестових даних.

        Параметри:
        - training_data_folder (str): Шлях до папки з тестовими даними.
        """
        # Шлях до файлу тестових даних
        test_file = f"{training_data_folder}\\emnist-byclass-test.csv"
        X_test, y_test = self.load_emnist_data(test_file)
        # Оцінка моделі на тестових даних
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Тестовий loss: {loss}, Точність: {accuracy}")

    def load_emnist_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Завантажує та готує дані EMNIST.

        Параметри:
        - data_path (str): Шлях до CSV-файлу з даними.
        
        Повертає:
        - X (np.ndarray): Вхідні дані (зображення).
        - y (np.ndarray): Мітки класів.
        """
        print(f'Завантаження даних із {data_path}...')
        df = pd.read_csv(data_path, header=None)  # Читання даних із CSV-файлу
        y = df.iloc[:, 0].values  # Мітки класів (перший стовпець)
        X = df.iloc[:, 1:].values  # Решта стовпців — пікселі
        X = X.reshape(-1, 28, 28, 1).astype("float32") / 255.0  # Нормалізація
        return X, y
    
    def show_sample_images(self, X: np.ndarray, y: np.ndarray, number: int, num_samples: int = 5):
        """
        (Допоміжна функція) Візуалізує кілька зображень із датасету.

        Параметри:
        - X (np.ndarray): Масив зображень.
        - y (np.ndarray): Масив міток класів.
        - number (int): Мітка класу для фільтрації.
        - num_samples (int): Кількість виведених зображень.
        """
        t = 0  # Лічильник виведених зображень
        for i in range(len(X)):
            if y[i] == number:  # Якщо мітка зображення збігається із заданою
                image = X[i].reshape(28, 28).T  # Перетворюємо в 28x28 (EMNIST використовує цей розмір)
                plt.imshow(image, cmap='gray')  # Відображаємо зображення
                # Заголовок із міткою
                plt.title(f"Тестове зображення, мітка: {y[i]} - {emnist_labels[y[i]]}")
                plt.show()  # Показуємо зображення
                t += 1
            if t == num_samples:  # Якщо виведено потрібну кількість зображень
                break

    def recognize_letter(self, data: List[int]) -> Tuple[str, float]:
        """
        Розпізнає одиночну літеру з траєкторії.

        Параметри:
        - data (List[int]): Дані траєкторії (список точок).
        
        Повертає:
        - Tuple[str, float]: Розпізнану літеру та впевненість у розпізнаванні.
        """
        # Прогнозуємо мітку для зображення
        prediction = self.model.predict(self.normalize_image(data))
        predicted_class = np.argmax(prediction)# Знаходимо індекс найбільш ймовірного класу
        confidence = np.max(prediction)  # Впевненість у прогнозі
        return emnist_labels[predicted_class], confidence # Повертаємо літеру та впевненість

    def normalize_image(self, data: List[int]) -> np.ndarray:
        """
        Перетворює траєкторію на зображення.

        Параметри:
        - data (List[int]): Дані траєкторії.
        
        Повертає:
        - np.ndarray: Нормалізоване зображення.
        """
        image_data = np.array(data) # Перетворюємо дані в numpy масив
        image_data = image_data.reshape(28, 28) # Перетворюємо в розмір 28x28
        image_data = image_data.astype("float32") / 255.0 # Нормалізуємо зображення
        image_data = np.expand_dims(image_data, axis=-1) # Додаємо розмірність для каналу
        image_data = np.expand_dims(image_data, axis=0)  # Додаємо розмірність для партії
        return image_data.T  # Повертаємо транспоноване зображення
    
    def recognize_letters(self, data: List[List[int]]) -> Tuple[str, int]:
        """
        Розпізнає кілька букв із траєкторії.

        Параметри:
        - data (List[List[int]]): Список траєкторій (кожна траєкторія — список точок).
        
        Повертає:
        - Tuple[str, int]: Рядок із розпізнаними буквами та індекс помилки (за замовчуванням 0).
        """
        s = ''
        for e in data:
            s += self.recognize_letter(e)[0]  # Розпізнаємо кожну букву та додаємо в рядок
        return s, 0  # Повертаємо рядок і 0, щоб відповідь була схожа на recognize_letter
        

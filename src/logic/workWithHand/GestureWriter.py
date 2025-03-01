import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from data.config import train_data_folder, emnist_labels  # Пути к данным
from data.config import train_data_folder, emnist_labels


class GestureWriter:
    def __init__(self, training_data_folder=None):
        self.model = None
        self.get_model(training_data_folder)
        # self.test_model(training_data_folder)

    def get_model(self, training_data_folder):
        """Загружает или обучает KNN-модель"""
        model_file = f"{training_data_folder}\\emnist_cnn_model.h5"
        if os.path.exists(model_file):
            print("Загружаем ранее обученную модель...")
            self.model = load_model(model_file)
        else:
            print("Обучаем новую модель...")
            train_file = f"{train_data_folder}\\emnist-byclass-train.csv"
            test_file = f"{train_data_folder}\\emnist-byclass-test.csv"
            model_file = "emnist_cnn_model.h5"
            X_train, y_train = self.load_emnist_data(train_file)
            X_test, y_test = self.load_emnist_data(test_file)
            num_classes = len(set(y_train))
            y_train_categorical = to_categorical(y_train, num_classes=num_classes)
            y_test_categorical = to_categorical(y_test, num_classes=num_classes)
            self.model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(num_classes, activation='softmax')
            ])
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', 
                               metrics=['accuracy'])
            print("Начинаем обучение...")
            self.model.fit(X_train, y_train_categorical, epochs=10, batch_size=128, 
                           validation_data=(X_test, y_test_categorical))
            self.model.save("emnist_cnn_model.h5")
            print(f"Модель сохранена в файл {model_file}")

    def test_model(self, training_data_folder):
        test_file = f"{training_data_folder}\\emnist-byclass-test.csv"
        X_test, y_test = self.load_emnist_data(test_file)
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Тестовый loss: {loss}, Точность: {accuracy}")


    def load_emnist_data(self, data_path):
        """Загружает и подготавливает данные EMNIST"""
        print(f'Загрузка данных из {data_path}...')
        df = pd.read_csv(data_path, header=None)
        y = df.iloc[:, 0].values  # Метки классов (первый столбец)
        X = df.iloc[:, 1:].values  # Остальные столбцы - пиксели
        X = X.reshape(-1, 28, 28, 1).astype("float32") / 255.0  # Нормализация
        return X, y
    def show_sample_images(self, X, y,number, num_samples=5):
        """Визуализация нескольких изображений из датасета"""
        t = 0
        for i in range(len(X)):
            if(y[i] == number):
                image = X[i].reshape(28, 28).T  # Преобразуем в 28x28 (EMNIST использует этот размер)
                plt.imshow(image, cmap='gray')
                plt.title(f"Тестовое изображение, метка: {y[i]} - {emnist_labels[y[i]]}")
                plt.show()
                t += 1
            if t == num_samples:
                break


    def recognize_letter(self, data):
        """Распознает текст на основе траектории"""
        prediction = self.model.predict(self.normalize_image(data))
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) 
        return emnist_labels[predicted_class], confidence

    def normalize_image(self, data):
        """Конвертирует траекторию в изображение"""
        image_data = np.array(data) 
        image_data = image_data.reshape(28, 28)
        image_data = image_data.astype("float32") / 255.0
        image_data = np.expand_dims(image_data, axis=-1)
        image_data = np.expand_dims(image_data, axis=0) 
        return image_data.T
    
    def recognize_letters(self, data):
        s = ''
        for e in data:
            s+=self.recognize_letter(e)[0]
        return s, 0
        

import joblib # type: ignore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score # type: ignore
from data.config import train_data_folder

class ModelTester:
    def __init__(self, model_file, test_data_file):
        # Завантажуємо модель з файлу
        self.recognizer = joblib.load(model_file)
        self.test_data_file = test_data_file

    def load_emnist_data(self, file_path):
        # Завантажуємо дані з CSV файлу
        data = pd.read_csv(file_path)

        # Припускаємо, що CSV файл має такий формат:
        # Перша колонка — це мітка (наприклад, буква), інші — це пікселі (28x28 = 784)
        X = data.iloc[:, 1:].values  # Усі рядки, всі стовпці, крім першого (пікселі)
        y = data.iloc[:, 0].values  # Перша колонка (мітки)

        # Перетворюємо дані в потрібний формат (наприклад, в двовимірний масив)
        X = X.astype(np.float32)  # Перетворюємо в тип float32 для подальших операцій
        y = y.astype(np.int32)  # Мітки повинні бути цілими числами

        return X, y

    def test_model(self):
        # Завантажуємо тестові дані
        X_test, y_test = self.load_emnist_data(self.test_data_file)

        # Прогнозуємо мітки для тестових даних
        y_pred = self.recognizer.predict(X_test)

        # Оцінка точності
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Точність на тестових даних: {accuracy * 100:.2f}%")

        # Перевіряємо кілька випадкових передсказань вручну
        self.check_random_predictions(X_test, y_test)

        # Візуалізуємо помилки
        self.visualize_errors(X_test, y_test, y_pred)

    def check_random_predictions(self, X_test, y_test):
        print("\nПеревірка кількох випадкових передсказань:\n")
        for i in range(5):  # Показуємо перші 5 випадкових прикладів
            sample_image = X_test[i]
            true_label = y_test[i]
            predicted_label = self.recognizer.predict([sample_image])[0]

            print(f"Правильна мітка: {true_label}, Прогнозована мітка: {predicted_label}")

            # Візуалізуємо зображення
            plt.imshow(sample_image.reshape(28, 28), cmap="gray")
            plt.title(f"Правильна: {true_label}, Прогнозована: {predicted_label}")
            plt.show()

    def visualize_errors(self, X_test, y_test, y_pred):
        # Візуалізуємо зображення, де модель помилилася
        incorrect_indices = [i for i in range(len(y_test)) if y_pred[i] != y_test[i]]

        if incorrect_indices:
            print("\nВізуалізація помилок (до 5 зображень):\n")
            for idx in incorrect_indices[:5]:
                plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
                plt.title(f"Правильна: {y_test[idx]}, Прогнозована: {y_pred[idx]}")
                plt.show()
        else:
            print("Немає помилок у передсказаннях!")

if __name__ == "__main__":
    # Вкажіть шлях до збереженого файлу моделі та CSV файлу з тестовими даними
    model_file = f'{train_data_folder}\\model.pkl'
    test_data_file = f'{train_data_folder}\\emnist-balanced-test.csv'

    # Створюємо екземпляр тестера і виконуємо перевірку
    tester = ModelTester(model_file, test_data_file)
    tester.test_model()

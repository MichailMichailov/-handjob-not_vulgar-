import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from data.config import train_data_folder

class ModelTester:
    def __init__(self, model_file, test_data_file):
        # Загружаем модель из файла
        self.recognizer = joblib.load(model_file)
        self.test_data_file = test_data_file

    def load_emnist_data(self, file_path):
        # Загружаем данные из CSV файла
        data = pd.read_csv(file_path)

        # Предполагаем, что CSV файл имеет формат:
        # Первая колонка — это метка (например, буква), остальные — это пиксели (28x28 = 784)
        X = data.iloc[:, 1:].values  # Все строки, все столбцы кроме первого (пиксели)
        y = data.iloc[:, 0].values  # Первая колонка (метки)

        # Преобразуем данные в нужный формат (например, в двумерный массив)
        X = X.astype(np.float32)  # Преобразуем в тип float32 для дальнейших операций
        y = y.astype(np.int32)  # Метки должны быть целыми числами

        return X, y

    def test_model(self):
        # Загружаем тестовые данные
        X_test, y_test = self.load_emnist_data(self.test_data_file)

        # Прогнозируем метки для тестовых данных
        y_pred = self.recognizer.predict(X_test)

        # Оценка точности
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy on test data: {accuracy * 100:.2f}%")

        # Проверяем несколько предсказаний вручную
        self.check_random_predictions(X_test, y_test)

        # Визуализируем ошибки
        self.visualize_errors(X_test, y_test, y_pred)

    def check_random_predictions(self, X_test, y_test):
        print("\nTesting some random predictions:\n")
        for i in range(5):  # Покажем первые 5 случайных примеров
            sample_image = X_test[i]
            true_label = y_test[i]
            predicted_label = self.recognizer.predict([sample_image])[0]

            print(f"True label: {true_label}, Predicted label: {predicted_label}")

            # Визуализируем изображение
            plt.imshow(sample_image.reshape(28, 28), cmap="gray")
            plt.title(f"True: {true_label}, Pred: {predicted_label}")
            plt.show()

    def visualize_errors(self, X_test, y_test, y_pred):
        # Визуализируем изображения, где модель ошиблась
        incorrect_indices = [i for i in range(len(y_test)) if y_pred[i] != y_test[i]]

        if incorrect_indices:
            print("\nVisualizing errors (up to 5 images):\n")
            for idx in incorrect_indices[:5]:
                plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
                plt.title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
                plt.show()
        else:
            print("No errors in the predictions!")

if __name__ == "__main__":
    # Укажи путь к сохраненному файлу модели и CSV файлу с тестовыми данными
    model_file = f'{train_data_folder}\\model.pkl'
    test_data_file = f'{train_data_folder}\\emnist-balanced-test.csv'

    # Создаем экземпляр тестера и выполняем проверку
    tester = ModelTester(model_file, test_data_file)
    tester.test_model()
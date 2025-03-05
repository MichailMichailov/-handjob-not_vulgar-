# Імпортуємо модуль os для роботи з файловою системою
import os 
# Імпорт dotenv для завантаження змінних оточення
# from dotenv import load_dotenv                    
# Завантажуємо змінні оточення з .env-файлу
# load_dotenv()

# Отримуємо абсолютний шлях до папки, де знаходиться цей скрипт
parent_dir = os.path.dirname(os.path.abspath(__file__))
# Приклад формування шляху до файлу
resultFile = os.path.join(parent_dir, '', "result.png")
# Формуємо шлях до папки knn для зберігання навчальних даних та моделі
train_data_folder = os.path.join(parent_dir, '', "knn")
# Словник відповідності індексів символам EMNIST
emnist_labels = {
    **{i: str(i) for i in range(10)},  # 0-9 (цифри)
    **{i + 10: chr(65 + i) for i in range(26)},  # 10-35 (A-Z)
    **{i + 36: chr(97 + i) for i in range(26)}  # 36-61 (a-z)
}
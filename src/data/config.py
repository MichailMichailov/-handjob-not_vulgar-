# Импортируем модуль os для работы с файловой системой
import os 
# Импорт dotenv для загрузки переменных окружения
# from dotenv import load_dotenv                    
# Загружаем переменные окружения из .env-файла
# load_dotenv()

# Получаем абсолютный путь к папке, где находится этот скрипт
parent_dir = os.path.dirname(os.path.abspath(__file__))
# Пример формирования пути к файлу
resultFile = os.path.join(parent_dir, '', "result.png")
# Формируем путь к папке knn для хранения обучающих данных и модели
train_data_folder = os.path.join(parent_dir, '', "knn")
# Словарь соответствия индексов символам EMNIST
emnist_labels = {
    **{i: str(i) for i in range(10)},  # 0-9 (цифры)
    **{i + 10: chr(65 + i) for i in range(26)},  # 10-35 (A-Z)
    **{i + 36: chr(97 + i) for i in range(26)}  # 36-61 (a-z)
}
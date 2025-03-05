# Імпортуємо OpenCV для роботи з відеопотоком
import cv2
# Імпортуємо типи для інструкції функцій
from typing import List, Dict

# Функція не використовується - залишена для можливого розширення функціоналу програми
def find_cameras(max_cameras: int = 2) -> List[Dict[str, str]]:
    """
    Функція пошуку доступних камер в системі.

    :param max_cameras: Максимальна кількість індексів камер, які перевіряються (за замовчуванням 2).
    :return: Список словників зі знайденими камерами, що містить 'id' (індекс камери) та 'name' (назва бекенда)."""
    cameras = []  # Список для зберігання інформації про знайдені камери
    for i in range(max_cameras):  # Перевіряємо камери від 0 до max_cameras - 1
        cap = cv2.VideoCapture(i)  # Відкриваємо відеопотік із індексом i
        if cap.isOpened():  # Перевіряємо, чи успішно відкрито камеру
            cameras.append({'id': i, 'name': cap.getBackendName()})  # Додаємо інформацію про камеру
            cap.release()  # Визволяємо ресурс камери, щоб не блокувати її
        else:
            print(f"Не вдалося відкрити камеру з індексом {i}.")  # Виводимо повідомлення, якщо камеру не вдалося відкрити
    return cameras  # Повертаємо список знайдених камер

import cv2
from typing import List, Dict, Tuple, Set


def find_cameras(max_cameras:int=2)->List:
    cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append({'id':i, 'name': cap.getBackendName()})  # Добавляем камеру в список
            cap.release()  # Освобождаем ресурс камеры
        else:
            print(f"Не удалось открыть камеру с индексом {i}.")
    return cameras
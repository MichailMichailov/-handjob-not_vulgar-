# Библиотека для работы с изображениями и видео
import cv2
# для работы с масивами
import numpy as np
# Библиотека для компьютерного зрения, включая распознавание рук
import mediapipe as mp
# Импортируем типы для аннотации функций
from typing import List, Dict, Tuple, Set, Optional



class HandTracker:
    """Класс для распознования жестов"""
    def __init__(self, min_detection_confidence: float = 0.8, min_tracking_confidence: float = 0.8):
        """
        Инициализирует объект для отслеживания рук.

        Параметры:
        - min_detection_confidence (float): Минимальная уверенность для детекции руки.
        - min_tracking_confidence (float): Минимальная уверенность для отслеживания руки.
        """
        # Используем решение для распознавания рук в MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=min_detection_confidence, 
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils # Используем утилиту для рисования на изображениях

    def get_finger_position(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Получает координаты кончика указательного пальца на изображении.

        Параметры:
        - frame (np.ndarray): Изображение (кадр видео).

        Возвращает:
        - Tuple[int, int] или None: Координаты кончика указательного пальца (x, y), если найдено, иначе None.
        """
        h, w, _ = frame.shape # Получаем размеры изображения
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Преобразуем изображение в RGB формат
        results = self.hands.process(frame_rgb) # Обрабатываем изображение для нахождения рук
        if results.multi_hand_landmarks: # Если найдены руки
            for hand_landmarks in results.multi_hand_landmarks:
                # Получаем координаты кончика указательного пальца
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h) # Преобразуем в пиксели
                return x, y
        return None
    
    def fist_detect(self, frame: np.ndarray) -> bool:
        """
        Определяет, сжат ли кулак.

        Параметры:
        - frame (np.ndarray): Изображение (кадр видео).

        Возвращает:
        - bool: True, если найден кулак, иначе False.
        """
        # Обрабатываем изображение для нахождения рук
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks: # Если найдены руки
            for hand_landmarks in results.multi_hand_landmarks:
                # self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                tips = [
                        self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                        self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        self.mp_hands.HandLandmark.RING_FINGER_TIP,
                        self.mp_hands.HandLandmark.PINKY_TIP
                ]
                mcps = [
                        self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
                        self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                        self.mp_hands.HandLandmark.RING_FINGER_MCP,
                        self.mp_hands.HandLandmark.PINKY_MCP
                ]

                is_fist = True # Предполагаем, что это кулак
                for tip, mcp in zip(tips, mcps): # Для каждого пальца проверяем, находится ли его кончик ниже основания
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y:
                        is_fist = False # Если хотя бы одно условие не выполняется, это не кулак
                        break

                return is_fist if True else False
        return False
 
    def is_thumb_up(self, frame: np.ndarray) -> bool:
        """
        Определяет, поднят ли большой палец вверх.

        Параметры:
        - results (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): Результаты обработки рук.
        
        Возвращает:
        - bool: True, если большой палец поднят вверх, иначе False.
        """
        # Обрабатываем изображение для нахождения рук
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks: # Если найдены руки
            for hand_landmarks in results.multi_hand_landmarks:
                # self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark  # Получаем координаты всех точек на руке

                thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
                thumb_cmc = landmarks[self.mp_hands.HandLandmark.THUMB_CMC]
                index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
                middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_mcp = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
                ring_mcp = landmarks[self.mp_hands.HandLandmark.RING_FINGER_MCP]
                pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
                pinky_mcp = landmarks[self.mp_hands.HandLandmark.PINKY_MCP]
                # Проверяем, что большой палец выше его основания
                thumb_up = thumb_tip.y < thumb_cmc.y
                # Проверяем, что остальные пальцы согнуты
                fingers_folded = (
                    index_tip.y > index_mcp.y 
                    and middle_tip.y > middle_mcp.y 
                    and ring_tip.y > ring_mcp.y 
                    and pinky_tip.y > pinky_mcp.y
                )
                if thumb_up and fingers_folded:
                    return True
            return False 
        return False
    
    def write_text_on_canvas(self, frame: np.ndarray, text: str, x: int, y: int):
        """
        (Вспомогательная функция)  Рисует текст на изображении.

        Параметры:
        - frame (np.ndarray): Изображение, на котором нужно нарисовать текст.
        - text (str): Текст, который нужно нарисовать.
        - x (int): Координата x для размещения текста.
        - y (int): Координата y для размещения текста.
        """
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (0, 255, 0), 2, cv2.LINE_AA)

                       


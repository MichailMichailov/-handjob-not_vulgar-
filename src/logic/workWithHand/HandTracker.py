# Бібліотека для роботи із зображеннями та відео
import cv2
# Для роботи з масивами
import numpy as np
# Бібліотека для комп'ютерного зору, включаючи розпізнавання рук
import mediapipe as mp
# Імпортуємо типи для анотації функцій
from typing import List, Dict, Tuple, Set, Optional


class HandTracker:
    """Клас для розпізнавання жестів"""
    def __init__(self, min_detection_confidence: float = 0.8, min_tracking_confidence: float = 0.8):
        """
        Ініціалізує об'єкт для відстеження рук.

        Параметри:
        - min_detection_confidence (float): Мінімальна впевненість для детекції руки.
        - min_tracking_confidence (float): Мінімальна впевненість для відстеження руки.
        """
        # Використовуємо рішення для розпізнавання рук у MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=min_detection_confidence, 
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils # Використовуємо утиліту для малювання на зображеннях

    def get_finger_position(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Отримує координати кінчика вказівного пальця на зображенні.

        Параметри:
        - frame (np.ndarray): Зображення (кадр відео).

        Повертає:
        - Tuple[int, int] або None: Координати кінчика вказівного пальця (x, y), якщо знайдено, інакше None.
        """
        h, w, _ = frame.shape # Отримуємо розміри зображення
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Перетворюємо зображення у формат RGB
        results = self.hands.process(frame_rgb) # Обробляємо зображення для знаходження рук
        if results.multi_hand_landmarks: # Якщо знайдено руки
            for hand_landmarks in results.multi_hand_landmarks:
                # Отримуємо координати кінчика вказівного пальця
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h) # Перетворюємо у пікселі
                return x, y
        return None
    
    def fist_detect(self, frame: np.ndarray) -> bool:
        """
        Визначає, чи стиснутий кулак.

        Параметри:
        - frame (np.ndarray): Зображення (кадр відео).

        Повертає:
        - bool: True, якщо знайдено кулак, інакше False.
        """
        # Обробляємо зображення для знаходження рук
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:  # Якщо знайдено руки
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

                is_fist = True  # Припускаємо, що це кулак
                for tip, mcp in zip(tips, mcps):  # Для кожного пальця перевіряємо, чи його кінчик нижче основи
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y:
                        is_fist = False  # Якщо хоча б одна умова не виконується, це не кулак
                        break

                return is_fist if True else False
        return False

    def is_thumb_up(self, frame: np.ndarray) -> bool:
        """
        Визначає, чи піднятий великий палець вгору.

        Параметри:
        - frame (np.ndarray): Зображення (кадр відео).

        Повертає:
        - bool: True, якщо великий палець піднятий вгору, інакше False.
        """
        # Обробляємо зображення для знаходження рук
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:  # Якщо знайдено руки
            for hand_landmarks in results.multi_hand_landmarks:
                # self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark  # Отримуємо координати всіх точок на руці

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
                # Перевіряємо, що великий палець вище його основи
                thumb_up = thumb_tip.y < thumb_cmc.y
                # Перевіряємо, що інші пальці зігнуті
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
        (Допоміжна функція) Малює текст на зображенні.

        Параметри:
        - frame (np.ndarray): Зображення, на якому потрібно намалювати текст.
        - text (str): Текст, який потрібно намалювати.
        - x (int): Координата x для розміщення тексту.
        - y (int): Координата y для розміщення тексту.
        """
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (0, 255, 0), 2, cv2.LINE_AA)

                       


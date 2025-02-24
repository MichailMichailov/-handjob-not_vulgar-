import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, min_detection_confidence=0.8, min_tracking_confidence=0.8):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=min_detection_confidence, 
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def get_finger_position(self, frame):
        """Возвращает координаты кончика указательного пальца (x, y)"""
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                return x, y
        return None
    
    def fist_detect(self, results ):
        """Возвращает истину если найден кулак"""
        if results.multi_hand_landmarks:
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

                is_fist = True
                for tip, mcp in zip(tips, mcps):
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y:
                        is_fist = False
                        break

                return is_fist if True else False
        return None
 
    def is_thumb_up(self, results):
        """ Определяет, поднят ли большой палец вверх """
        if results.multi_hand_landmarks: 
            for hand_landmarks in results.multi_hand_landmarks:
                # self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark

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

                thumb_up = thumb_tip.y < thumb_cmc.y
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
    
    def write_text_on_canvas(self, frame, text, x, y):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (0, 255, 0), 2, cv2.LINE_AA)

                       


import cv2
from workWithHand.GestureWriter import GestureWriter
from workWithHand.HandTracker import HandTracker
from canva.DrawingCanvas import DrawingCanvas
from data.config import resultFile, train_data_folder

class AirWritingApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.tracker = HandTracker()
        self.canvas = None
        self.writer = GestureWriter(train_data_folder)

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            if self.canvas is None:
                self.canvas = DrawingCanvas(w, h)
            finger_pos = self.tracker.get_finger_position(frame)
            if finger_pos:
                if finger_pos == -1:
                    self.canvas.clear_prev()
                else:
                    x, y = finger_pos
                    self.canvas.draw_line(x, y)
            frame[0:400, 0:400] = self.canvas.get_canvas()
            rezult = self.writer.recognize_letter(self.canvas.get_single_letter())
            self.tracker.write_text_on_canvas(frame, rezult, 400, 400)

            cv2.imshow("Air Writing", frame)
            key = cv2.waitKey(1)
            if key == 27: # esc
                break
            elif key == 99: # c
                self.canvas.clear()
        self.cap.release()
        cv2.destroyAllWindows()

import cv2
from logic.workWithHand.GestureWriter import GestureWriter
from logic.workWithHand.HandTracker import HandTracker
from logic.canva.DrawingCanvas import DrawingCanvas
from data.config import resultFile, train_data_folder

class AirWritingApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.tracker = HandTracker()
        self.canvas = None
        self.writer = GestureWriter(train_data_folder)
        self.is_write = True

    def set_is_write(self, value):
        self.is_write = value

    def generate_frames(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            if self.canvas is None:
                self.canvas = DrawingCanvas(w, h)

            if (not self.tracker.fist_detect(frame)) and self.is_write:
                finger_pos = self.tracker.get_finger_position(frame)
                if finger_pos:
                    x, y = finger_pos
                    self.canvas.draw_line(x, y)
            else:
                self.canvas.clear_prev()
            canvas_img = self.canvas.get_canvas()
            # Проверяем количество каналов (избегаем ошибки OpenCV)
            if len(canvas_img.shape) == 2:
                canvas_bgr = cv2.cvtColor(canvas_img, cv2.COLOR_GRAY2BGR)
            else:
                canvas_bgr = canvas_img
            
            ch, cw, _ = canvas_bgr.shape  # Размеры канваса
            x_offset = (w - cw) // 2
            y_offset = (h - ch) // 2
            # Вставляем холст по центру
            frame[y_offset:y_offset + ch, x_offset:x_offset + cw] = canvas_bgr

            # Распознаем букву
            # result, confidence = self.writer.recognize_letter(self.canvas.get_single_letter())
            # if confidence > 0.7:
            #     text = f"{result} ({confidence*100:.2f}%)"
            #     self.tracker.write_text_on_canvas(frame, text, 400, 400)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')





 # def run(self):
    #     while self.cap.isOpened():
    #         ret, frame = self.cap.read()
    #         if not ret:
    #             break
    #         frame = cv2.flip(frame, 1)
    #         h, w, _ = frame.shape
    #         if self.canvas is None:
    #             self.canvas = DrawingCanvas(w, h)
    #         finger_pos = self.tracker.get_finger_position(frame)
    #         if finger_pos:
    #             if finger_pos == -1:
    #                 self.canvas.clear_prev()
    #             else:
    #                 x, y = finger_pos
    #                 self.canvas.draw_line(x, y)
    #         frame[0:400, 0:400] = self.canvas.get_canvas()
    #         rezult = self.writer.recognize_letter(self.canvas.get_single_letter())
    #         self.tracker.write_text_on_canvas(frame, rezult, 400, 400)

    #         cv2.imshow("Air Writing", frame)
    #         key = cv2.waitKey(1)
    #         if key == 27: # esc
    #             break
    #         elif key == 99: # c
    #             self.canvas.clear()
    #     self.cap.release()
    #     cv2.destroyAllWindows()
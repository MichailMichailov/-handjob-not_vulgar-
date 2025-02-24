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

    def generate_frames(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            if self.canvas is None:
                self.canvas = DrawingCanvas(w, h)
            
            # Получаем координаты пальца
            finger_pos = self.tracker.get_finger_position(frame)
            if finger_pos:
                if finger_pos == -1:
                    self.canvas.clear_prev()
                else:
                    x, y = finger_pos
                    self.canvas.draw_line(x, y)

            # Получаем изображение canvas
            canvas_img = self.canvas.get_canvas()
            
            # Проверяем количество каналов (избегаем ошибки OpenCV)
            if len(canvas_img.shape) == 2:
                canvas_bgr = cv2.cvtColor(canvas_img, cv2.COLOR_GRAY2BGR)
            else:
                canvas_bgr = canvas_img
            
            # Встраиваем canvas в кадр камеры
            frame[0:400, 0:400] = canvas_bgr  

            # Распознаем букву
            # result, confidence = self.writer.recognize_letter(self.canvas.get_single_letter())
            # text = f"{result} ({confidence*100:.2f}%)"
            text = 'ddd'
            self.tracker.write_text_on_canvas(frame, text, 400, 400)

            # Кодируем изображение в формат JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
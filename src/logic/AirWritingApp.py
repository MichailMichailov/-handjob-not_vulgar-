import cv2 # Импорт библиотеки OpenCV для обработки изображений и видео
from logic.workWithHand.GestureWriter import GestureWriter  # Импорт класса для распознавания текста нейросетью
from logic.workWithHand.HandTracker import HandTracker  # Импорт класса для отслеживания положения рук
from logic.canva.DrawingCanvas import DrawingCanvas  # Импорт класса для рисования на холсте
from data.config import train_data_folder  # Импорт путей к файлам конфигурации

class AirWritingApp:
    """Главный клас логики программы"""
    def __init__(self):
        """ Инициализация """
        self.cap = cv2.VideoCapture(0) # Открытие видеопотока с камеры по умолчанию
        self.tracker = HandTracker() # Инициализация трекера рук
        self.canvas = None # Переменная для холста, на котором будет рисоваться
        self.writer = GestureWriter(train_data_folder) # Инициализация распознавателя жестов
        self.is_write = True # Флаг для включения/выключения режима рисования

    def set_is_write(self, value: bool):
        """ Устанавливает флаг рисования. """
        self.is_write = value

    def generate_frames(self):
        """
        Генерирует кадры для видеопотока с обработанным изображением.

        Возвращает:
        - Генератор: каждый кадр в виде JPEG изображения.
        """
        while True:
            ret, frame = self.cap.read() # Чтение кадра из видеопотока
            if not ret:  # Если кадр не получен, выходим
                break
            frame = cv2.flip(frame, 1)  # Зеркальное отображение изображения по горизонтали
            h, w, _ = frame.shape  # Получаем размеры кадра
            if self.canvas is None: # Если холст ещё не создан, создаём его
                self.canvas = DrawingCanvas(400, 400)

            if (not self.tracker.fist_detect(frame)) and self.is_write: # Если не обнаружен кулак и рисование включено
                finger_pos = self.tracker.get_finger_position(frame) # Получаем координаты указательного пальца
                if finger_pos: # Если координаты пальца найдены
                    x, y = finger_pos  # Извлекаем координаты
                    self.canvas.draw_line(x, y) # Рисуем линию на холсте по координатам
            else:
                self.canvas.clear_prev() # Если обнаружен кулак, очищаем предыдущие координаты
            canvas_img = self.canvas.get_canvas() # Получаем изображение холста
            # Проверяем количество каналов (избегаем ошибки OpenCV)
            if len(canvas_img.shape) == 2:  # Если изображение холста в градациях серого (1 канал)
                canvas_bgr = cv2.cvtColor(canvas_img, cv2.COLOR_GRAY2BGR) # Преобразуем в 3 канала (цветное изображение)
            else:
                canvas_bgr = canvas_img  # Если изображение уже цветное, просто используем его
            
            # Размеры канваса
            ch, cw, _ = canvas_bgr.shape
            x_offset = (w - cw) // 2
            y_offset = (h - ch) // 2
            # Вставляем холст по центру
            frame[y_offset:y_offset + ch, x_offset:x_offset + cw] = canvas_bgr
            # Код для отправки изображения в виде потока
            _, buffer = cv2.imencode('.jpg', frame) # Кодируем кадр в формат JPEG
            frame_bytes = buffer.tobytes() # Преобразуем в байты для отправки по сети
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')  # Отправка кадра как части HTTP-ответа
 



# Старая функция generate_frames
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
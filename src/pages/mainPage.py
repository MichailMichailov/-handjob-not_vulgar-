from logic.AirWritingApp import AirWritingApp # Импорт основного класса приложения для рисования в воздухе
# Импорт библиотек Flask для создания веб-приложения
from flask import Flask, Response, jsonify, render_template, request
# Инициализация Flask приложения
app = Flask(__name__,  template_folder="../templates", static_folder="../static")
# Создаем экземпляр приложения для рисования в воздухе
app_instance = AirWritingApp()

@app.route('/')
def index() -> str:
    """
    Обработчик маршрута для главной страницы.

    Возвращает:
    - str: HTML шаблон для главной страницы.
    """
    return render_template('index.html') # Отображение главной страницы с помощью шаблона

@app.route('/video_feed')
def video_feed() -> Response:
    """
    Обработчик маршрута для передачи видеопотока.

    Возвращает:
    - Response: Видеопоток с кадрами, передаваемыми в формате multipart.
    """
    # Генерация кадров и передача их как видеопотока
    return Response(app_instance.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/clear_canvas', methods=['POST'])
def clear_canvas() -> str:
    """
    Обработчик маршрута для очистки холста.

    Возвращает:
    - str: Пустой ответ с кодом 204, что означает успешное выполнение запроса.
    """
    app_instance.canvas.clear()  # Очищаем холст
    return '', 204  # Возвращаем пустой ответ

@app.route('/get_letter', methods=['POST'])
def get_letter() -> str:
    """
    (Вспомогательная функция) Обработчик маршрута для распознавания одной буквы.

    Возвращает:
    - str: Пустой ответ с кодом 204.
    """
    # Печатаем распознанную букву в консоль
    print(app_instance.writer.recognize_letter(app_instance.canvas.get_single_letter()))
    return '', 204  # Возвращаем пустой ответ

@app.route('/set_is_write', methods=['POST'])
def set_is_write() -> str: 
    """
    Обработчик маршрута для изменения флага режима рисования.

    Возвращает:
    - str: Пустой ответ с кодом 204.
    """
    is_write = request.get_json().get('is_write', False)   # Получаем значение флага из JSON запроса
    app_instance.set_is_write(is_write) # Устанавливаем флаг рисования
    return '', 204 

@app.route('/get_letters', methods=['POST'])
def get_letters() -> Response:
    """
    Обработчик маршрута для распознавания нескольких букв.

    Возвращает:
    - Response: Ответ с JSON, содержащим распознанные буквы и уверенность.
    """
    if app_instance.canvas is None:
        return jsonify({'error': 'Canvas is not initialized'}), 500 # Возвращаем ошибку, если холст не инициализирован
    # Получаем результат распознавания одной буквы
    result, confidence = app_instance.writer.recognize_letter(app_instance.canvas.get_single_letter())
    # Получаем результат распознавания нескольких букв
    # result, confidence = app_instance.writer.recognize_letters(app_instance.canvas.get_several_letters())
    return jsonify({'letter': str(result), 'confidence': float(confidence)}) # Возвращаем JSON с результатом
from logic.AirWritingApp import AirWritingApp  # Імпорт основного класу програми для малювання в повітрі
# Імпорт бібліотек Flask для створення веб-додатку
from flask import Flask, Response, jsonify, render_template, request
# Ініціалізація Flask додатку
app = Flask(__name__, template_folder="../templates", static_folder="../static")
# Створюємо екземпляр додатку для малювання в повітрі
app_instance = AirWritingApp()

@app.route('/')
def index() -> str:
    """
    Обробник маршруту для головної сторінки.

    Повертає:
    - str: HTML шаблон для головної сторінки.
    """
    return render_template('index.html')  # Відображення головної сторінки за допомогою шаблону

@app.route('/video_feed')
def video_feed() -> Response:
    """
    Обробник маршруту для передачі відеопотоку.

    Повертає:
    - Response: Відеопотік з кадрами, передаваними у форматі multipart.
    """
    # Генерація кадрів і передача їх як відеопотоку
    return Response(app_instance.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/clear_canvas', methods=['POST'])
def clear_canvas() -> str:
    """
    Обробник маршруту для очищення полотна.

    Повертає:
    - str: Порожня відповідь з кодом 204, що означає успішне виконання запиту.
    """
    app_instance.canvas.clear()  # Очищаємо полотно
    return '', 204  # Повертаємо порожню відповідь

@app.route('/get_letter', methods=['POST'])
def get_letter() -> str:
    """
    (Допоміжна функція) Обробник маршруту для розпізнавання однієї букви.

    Повертає:
    - str: Порожня відповідь з кодом 204.
    """
    # Виводимо розпізнану букву в консоль
    print(app_instance.writer.recognize_letter(app_instance.canvas.get_single_letter()))
    return '', 204  # Повертаємо порожню відповідь

@app.route('/set_is_write', methods=['POST'])
def set_is_write() -> str:
    """
    Обробник маршруту для зміни прапора режиму малювання.

    Повертає:
    - str: Порожня відповідь з кодом 204.
    """
    is_write = request.get_json().get('is_write', False)  # Отримуємо значення прапора з JSON запиту
    app_instance.set_is_write(is_write)  # Встановлюємо прапор малювання
    return '', 204

@app.route('/get_letters', methods=['POST'])
def get_letters() -> Response:
    """
    Обробник маршруту для розпізнавання кількох букв.

    Повертає:
    - Response: Відповідь з JSON, що містить розпізнані букви та впевненість.
    """
    if app_instance.canvas is None:
        return jsonify({'error': 'Canvas is not initialized'}), 500  # Повертаємо помилку, якщо полотно не ініціалізовано
    # Отримуємо результат розпізнавання однієї букви
    result, confidence = app_instance.writer.recognize_letter(app_instance.canvas.get_single_letter())
    # Отримуємо результат розпізнавання кількох букв
    # result, confidence = app_instance.writer.recognize_letters(app_instance.canvas.get_several_letters())
    return jsonify({'letter': str(result), 'confidence': float(confidence)})  # Повертаємо JSON з результатом

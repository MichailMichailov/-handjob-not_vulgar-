
from logic.AirWritingApp import AirWritingApp
from flask import Flask, Response, jsonify, render_template, request
# if __name__ == "__main__":
#     app = AirWritingApp()
#     app.run()

app = Flask(__name__)
app_instance = AirWritingApp()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(app_instance.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/clear_canvas', methods=['POST'])
def clear_canvas():
    app_instance.canvas.clear()  # Очищаем холст
    return '', 204  # Возвращаем пустой ответ

@app.route('/get_letter', methods=['POST'])
def get_letter():
    print(app_instance.writer.recognize_letter(app_instance.canvas.get_single_letter()))
    return '', 204  # Возвращаем пустой ответ

@app.route('/set_is_write', methods=['POST'])
def set_is_write(): 
    is_write = request.get_json().get('is_write', False)  
    app_instance.set_is_write(is_write)
    print(is_write)
    return '', 204 

@app.route('/get_letters', methods=['POST'])
def get_letters():
    if app_instance.canvas is None:
        return jsonify({'error': 'Canvas is not initialized'}), 500
    # result, confidence = app_instance.writer.recognize_letter(app_instance.canvas.get_single_letter())
    result, confidence = app_instance.writer.recognize_letters(app_instance.canvas.get_several_letters())
    print("Отправляем JSON:", {'letter': result, 'confidence': confidence})
    return jsonify({'letter': str(result), 'confidence': float(confidence)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

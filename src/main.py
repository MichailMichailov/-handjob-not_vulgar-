
from canva.AirWritingApp import AirWritingApp
from flask import Flask, Response, render_template
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
    return Response(app_instance.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/clear_canvas', methods=['POST'])
def clear_canvas():
    app_instance.canvas.clear()  # Очищаем холст
    return '', 204  # Возвращаем пустой ответ

@app.route('/get_letter', methods=['POST'])
def get_letter():
    print(app_instance.writer.recognize_letter(app_instance.canvas.get_single_letter()))
    return '', 204  # Возвращаем пустой ответ

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

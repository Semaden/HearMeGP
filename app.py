from flask import Flask, render_template, Response, jsonify
import cv2
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

# API ve model bilgileri
API_URL = "https://detect.roboflow.com"
API_KEY = "Gs0X6B7FVqulAniaOzPh"
MODEL_ID = "turk-isaret-dili/2"

# Inference HTTP Client oluştur
client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

# Algılanan kelimeyi saklayacak değişken
detected_sign = None

def infer_image(image):
    global detected_sign
    try:
        # İnferans yap
        result = client.infer(image, model_id=MODEL_ID)
        predictions = result.get("predictions", [])
        if predictions:
            # Yeni bir işaret algılandığında güncelle
            detected_sign = predictions[0].get("label", detected_sign)
    except Exception as e:
        # Hata durumunda kelimeyi değiştirme
        pass

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open camera.")
    
    while True:
        # Kameradan görüntü al
        ret, frame = cap.read()
        if not ret:
            break

        # Görüntüyü işaret dili modeline gönder
        infer_image(frame)

        # Görüntüyü encode et
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Görüntüyü stream et
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detected_sign')
def get_detected_sign():
    return jsonify({"detected_sign": detected_sign or "Algılanan kelime bulunamadı."})

if __name__ == "__main__":
    app.run(debug=True)

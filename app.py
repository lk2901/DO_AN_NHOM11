from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import joblib

from trich_xuat import extract_features
from config import CLASS_NAMES

app = Flask(__name__)

print("Đang tải model...")

try:
    model = joblib.load("random_forest.pkl")
    print("Tải model thành công!")
except Exception as e:
    print("Lỗi tải model:", e)
    model = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_image', methods=['POST'])
def predict_image():

    if model is None:
        return jsonify({"result": "Model chưa được tải"})

    if 'image' not in request.files:
        return jsonify({"result": "Không tìm thấy ảnh"})

    file = request.files['image']

    npimg = np.frombuffer(file.read(), np.uint8)

    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"result": "Ảnh không hợp lệ"})

    features = extract_features(img)

    if features is None:
        return jsonify({"result": "Không thể xử lý ảnh"})

    result = model.predict([features])[0]
    probs = model.predict_proba([features])[0]

    confidence = np.max(probs) * 100
    label = CLASS_NAMES[result]

    return jsonify({
        "result": f"♻️ {label} ({confidence:.2f}%)"
    })


def generate_frames():

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        x1 = w // 4
        y1 = h // 4
        x2 = 3 * w // 4
        y2 = 3 * h // 4

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        roi = frame[y1:y2, x1:x2]

        if roi.size > 0:

            features = extract_features(roi)

            if features is not None:

                result = model.predict([features])[0]
                probs = model.predict_proba([features])[0]

                confidence = np.max(probs) * 100
                label = CLASS_NAMES[result]

                text = f"{label} {confidence:.1f}%"

                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        ret, buffer = cv2.imencode('.jpg', frame)

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )

    cap.release()


@app.route('/video_feed')
def video_feed():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)
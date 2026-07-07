import cv2
import joblib
import numpy as np

from trich_xuat import extract_features
from config import CLASS_NAMES

# Tải model
try:
    model = joblib.load("random_forest.pkl")
    print("✅ Tải model thành công!")
except Exception as e:
    print("❌ Lỗi tải model:", e)
    exit()


def scan_webcam():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Không mở được webcam")
        return

    print("=" * 50)
    print("♻️ HỆ THỐNG NHẬN DIỆN RÁC THẢI")
    print("=" * 50)
    print("SPACE : Quét rác")
    print("ESC   : Thoát")
    print("=" * 50)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        x1 = w // 4
        y1 = h // 4
        x2 = 3 * w // 4
        y2 = 3 * h // 4

        # Khung nhận diện
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "DAT RAC THAI VAO KHUNG",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "SPACE: Scan | ESC: Exit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Waste Classification",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # SPACE
        if key == 32:

            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            features = extract_features(roi)

            if features is None:
                print("❌ Không trích xuất được đặc trưng")
                continue

            try:

                prediction = model.predict(
                    [features]
                )[0]

                probs = model.predict_proba(
                    [features]
                )[0]

                confidence = np.max(probs) * 100

                label = CLASS_NAMES[prediction]

                result_text = (
                    f"{label} - {confidence:.2f}%"
                )

                print(
                    f"♻️ Kết quả: {result_text}"
                )

                result_frame = frame.copy()

                cv2.rectangle(
                    result_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    result_frame,
                    result_text,
                    (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

                cv2.imshow(
                    "Waste Classification",
                    result_frame
                )

                cv2.waitKey(2000)

            except Exception as e:

                print(
                    "❌ Lỗi dự đoán:",
                    e
                )

        # ESC
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    scan_webcam()
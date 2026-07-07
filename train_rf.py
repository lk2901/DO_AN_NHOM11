import os
import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from trich_xuat import extract_features

CLASS_MAP = {
    "Plastic": 0,
    "Paper": 1,
    "Metal": 2,
    "Glass": 3,
    "Organic": 4
}


def load_dataset(folder_root):

    X = []
    y = []

    for class_name, label in CLASS_MAP.items():

        folder = os.path.join(
            folder_root,
            class_name
        )

        if not os.path.exists(folder):
            print("Không tìm thấy:", folder)
            continue

        print(f"Đang đọc {class_name}...")

        count = 0

        for file_name in os.listdir(folder):

            path = os.path.join(
                folder,
                file_name
            )

            img = cv2.imread(path)

            if img is None:
                continue

            features = extract_features(img)

            if features is not None:
                X.append(features)
                y.append(label)
                count += 1

        print("Số ảnh:", count)

    return np.array(X), np.array(y)


print("Đọc dữ liệu train...")
X_train, y_train = load_dataset("dataset/train")

print("Đọc dữ liệu test...")
X_test, y_test = load_dataset("dataset/test")

print("Bắt đầu huấn luyện...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Huấn luyện xong!")

pred = model.predict(X_test)

acc = accuracy_score(
    y_test,
    pred
)

print("Accuracy:", acc)

print(
    classification_report(
        y_test,
        pred,
        target_names=[
            "Plastic",
            "Paper",
            "Metal",
            "Glass",
            "Organic"
        ]
    )
)

joblib.dump(
    model,
    "random_forest.pkl"
)

print("Đã lưu random_forest.pkl")
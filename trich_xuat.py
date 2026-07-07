import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

print("Đang tải MobileNetV2...")

cnn_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

print("MobileNetV2 sẵn sàng!")


def extract_features(image):

    if image is None:
        return None

    try:

        img = cv2.resize(
            image,
            (224, 224)
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img = np.expand_dims(
            img,
            axis=0
        )

        img = preprocess_input(img)

        features = cnn_model.predict(
            img,
            verbose=0
        )

        return features[0]

    except Exception as e:

        print("Lỗi:", e)

        return None
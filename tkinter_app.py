import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import joblib
import numpy as np

from PIL import Image, ImageTk

from trich_xuat import extract_features
from config import CLASS_NAMES

MODEL_PATH = "random_forest.pkl"


class WasteClassifierApp:

    def __init__(self, root):

        self.root = root

        self.root.title("Phân Loại Rác Thải AI")

        self.root.geometry("900x700")

        self.root.configure(bg="#f5f5f5")

        self.model = None

        self.tk_image = None

        self.load_model()

        title = tk.Label(
            root,
            text="♻️ PHÂN LOẠI RÁC THẢI THÔNG MINH",
            font=("Arial", 20, "bold"),
            bg="#f5f5f5"
        )

        title.pack(pady=20)

        btn = tk.Button(
            root,
            text="Chọn Ảnh",
            command=self.upload_image,
            font=("Arial", 14, "bold"),
            bg="#4CAF50",
            fg="white",
            width=20
        )

        btn.pack(pady=10)

        self.image_label = tk.Label(root)

        self.image_label.pack(pady=10)

        self.result_label = tk.Label(
            root,
            text="Chưa có kết quả",
            font=("Arial", 18, "bold"),
            bg="#f5f5f5"
        )

        self.result_label.pack(pady=20)

    def load_model(self):

        try:

            self.model = joblib.load(MODEL_PATH)

            print("Tải model thành công")

        except Exception as e:

            messagebox.showerror(
                "Lỗi",
                str(e)
            )

    def upload_image(self):

        if self.model is None:

            return

        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image Files",
                 "*.jpg *.jpeg *.png *.bmp")
            ]
        )

        if not file_path:

            return

        img = cv2.imread(file_path)

        if img is None:

            messagebox.showerror(
                "Lỗi",
                "Không đọc được ảnh"
            )

            return

        self.show_image(img)

        features = extract_features(img)

        if features is None:

            messagebox.showerror(
                "Lỗi",
                "Không trích xuất được đặc trưng"
            )

            return

        pred = self.model.predict(
            [features]
        )[0]

        probs = self.model.predict_proba(
            [features]
        )[0]

        confidence = np.max(probs) * 100

        label = CLASS_NAMES[pred]

        self.result_label.config(
            text=f"{label} ({confidence:.2f}%)"
        )

    def show_image(self, image):

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        h, w = image.shape[:2]

        scale = min(
            500 / w,
            350 / h,
            1
        )

        new_w = int(w * scale)

        new_h = int(h * scale)

        image = cv2.resize(
            image,
            (new_w, new_h)
        )

        pil_img = Image.fromarray(image)

        self.tk_image = ImageTk.PhotoImage(
            pil_img
        )

        self.image_label.config(
            image=self.tk_image
        )


if __name__ == "__main__":

    root = tk.Tk()

    app = WasteClassifierApp(root)

    root.mainloop()
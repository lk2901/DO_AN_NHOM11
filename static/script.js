const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const result = document.getElementById("result");

imageInput.addEventListener("change", function () {
  const file = this.files[0];

  if (!file) return;

  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
});

function predictImage() {
  const file = imageInput.files[0];

  if (!file) {
    alert("Vui lòng chọn ảnh!");
    return;
  }

  const formData = new FormData();

  formData.append("image", file);

  result.innerHTML = "⏳ Đang xử lý...";

  fetch("/predict_image", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      result.innerHTML = data.result;
    })
    .catch((error) => {
      console.error(error);
      result.innerHTML = "❌ Lỗi xử lý ảnh";
    });
}

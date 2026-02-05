import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==============================
# KONFIGURASI HALAMAN
# ==============================
st.set_page_config(
    page_title="Sistem Diagnosis Gangguan Mental",
    page_icon="🧠",
    layout="centered"
)

# ==============================
# STYLE MODERN & KLINIS
# ==============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7fa, #e4ecf7);
}

.header-card {
    background: white;
    padding: 28px;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    text-align: center;
    margin-bottom: 30px;
}

.upload-card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}

.card {
    background: white;
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.12);
    margin-top: 25px;
}

.result {
    font-size: 26px;
    font-weight: bold;
    text-align: center;
    color: #2c3e50;
}

.confidence {
    font-size: 16px;
    text-align: center;
    color: #555;
}

.footer {
    text-align: center;
    color: #777;
    font-size: 14px;
    margin-top: 50px;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #1f77b4, #4facfe);
    color: white;
    font-size: 16px;
    padding: 12px;
    border-radius: 10px;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #4facfe, #1f77b4);
}
</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================
st.markdown("""
<div class="header-card">
    <h2>🏥 Sistem Diagnosis Gangguan Mental</h2>
    <h4>Clinical Decision Support System (CDSS) Berbasis CNN</h4>
    <p><b>Rahman Aldiansyah</b> — Teknik Informatika</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# DOWNLOAD & LOAD MODEL
# ==============================
MODEL_PATH = "model/best_model.h5"
GDRIVE_URL = "https://drive.google.com/uc?id=1xLAy2fQF8lGvtuXKGX4P2xnyXdNRiMf9"

if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    with st.spinner("📥 Mengunduh model CNN..."):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ==============================
# NAMA KELAS
# ==============================
class_names = [
    "Gangguan Kecemasan (Anxiety Disorder)",
    "Gangguan Depresi (Depressive Disorder)"
]

# ==============================
# RESIZE OTOMATIS (HP FRIENDLY)
# ==============================
MAX_DIM = 1024

def resize_image(img):
    w, h = img.size
    if max(w, h) > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)))
    return img

# ==============================
# UPLOAD GAMBAR
# ==============================
st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
st.subheader("📤 Upload Gambar untuk Diagnosis")
uploaded_file = st.file_uploader(
    "Format JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"]
)
st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# PROSES PREDIKSI
# ==============================
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img = resize_image(img)

    st.image(img, caption="🖼️ Gambar Input", width=350)

    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Proses Diagnosis"):
        with st.spinner("Menganalisis kondisi mental..."):
            prediction = model.predict(img_array)

        confidence = float(np.max(prediction) * 100)
        result = class_names[np.argmax(prediction)]

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🧠 Hasil Diagnosis Klinis")
        st.markdown(f"<div class='result'>{result}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='confidence'>Tingkat Keyakinan Model: {confidence:.2f}%</div>",
            unsafe_allow_html=True
        )
        st.progress(int(confidence))
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# FOOTER
# ==============================
st.markdown("""
<div class="footer">
    Aplikasi ini dikembangkan sebagai bagian dari penelitian skripsi<br>
    © 2026 — Rahman Aldiansyah
</div>
""", unsafe_allow_html=True)

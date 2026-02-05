import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==============================
# KONFIGURASI HALAMAN
# ==============================
st.set_page_config(
    page_title="Diagnosis Gangguan Mental Gen Z",
    page_icon="🧠",
    layout="centered"
)

# ==============================
# CUSTOM STYLE
# ==============================
st.markdown("""
<style>
/* Background klinis */
.stApp {
    background-color: #f4f8fb;
    font-family: 'Segoe UI', sans-serif;
}

/* Container */
.block-container {
    padding-top: 2rem;
    max-width: 900px;
}

/* Header bar */
.header {
    background: #0d6efd;
    color: white;
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 24px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.15);
}

/* Card medis */
.card {
    background: #ffffff;
    padding: 24px;
    border-radius: 14px;
    border-left: 6px solid #0d6efd;
    box-shadow: 0 6px 20px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

/* Judul hasil */
.result {
    font-size: 26px;
    font-weight: 700;
    color: #0d6efd;
}

/* Confidence */
.confidence {
    font-size: 18px;
    color: #495057;
}

/* Upload box */
.css-1cpxqw2 {
    background-color: #f9fcff;
    border: 2px dashed #0d6efd;
    border-radius: 12px;
}

/* Button medis */
.stButton > button {
    background-color: #0d6efd;
    color: white;
    font-weight: 600;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    border: none;
}

.stButton > button:hover {
    background-color: #084298;
}
</style>
""", unsafe_allow_html=True)


# ==============================
# HEADER
# ==============================
st.markdown("""
<div class="header">
    <h2>🏥 Sistem Diagnosis Gangguan Mental</h2>
    <p>
    Clinical Decision Support System (CDSS) Berbasis CNN<br>
    <strong>Rahman Aldiansyah</strong> — Teknik Informatika
    </p>
</div>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/best_model.h5")

model = load_model()

# ==============================
# NAMA KELAS (OUTPUT MODEL)
# ==============================
class_names = [
    "Gangguan Kecemasan (Anxiety Disorder)",
    "Gangguan Depresi (Depressive Disorder)"
]

# ==============================
# UPLOAD GAMBAR
# ==============================
st.subheader("📤 Upload Gambar untuk Diagnosis")
uploaded_file = st.file_uploader(
    "Format JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar Input", width=350)

    # Preprocessing
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ==============================
    # PREDIKSI
    # ==============================
    if st.button("🔍 Proses Diagnosis"):
        with st.spinner("Memproses diagnosis..."):
            prediction = model.predict(img_array)

        confidence = np.max(prediction) * 100
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
<hr>
<p style='text-align:center; font-size:13px; color:gray;'>
Aplikasi ini dikembangkan sebagai bagian dari penelitian skripsi<br>
© 2026 — Rahman Aldiansyah
</p>
""", unsafe_allow_html=True)

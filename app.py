import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.figure_factory as ff

# ==============================
# KONFIGURASI MODEL
# ==============================
MODEL_PATH = "model/best_model.h5"
GDRIVE_FILE_ID = "1xLAy2fQF8lGvtuXKGX4P2xnyXdNRiMf9"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    with st.spinner("🔄 Mengunduh model CNN..."):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ==============================
# KONFIGURASI HALAMAN
# ==============================
st.set_page_config(
    page_title="Sistem Diagnosis Gangguan Mental",
    page_icon="🏥",
    layout="centered"
)

# ==============================
# CUSTOM CSS
# ==============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f4f8fb, #ffffff);
    font-family: 'Segoe UI', sans-serif;
}
.block-container {
    max-width: 900px;
    padding-top: 2rem;
}
.header {
    background: linear-gradient(90deg, #0d6efd, #0a58ca);
    color: white;
    padding: 22px;
    border-radius: 16px;
    margin-bottom: 25px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.15);
}
.card {
    background: #ffffff;
    padding: 24px;
    border-radius: 16px;
    border-left: 6px solid #0d6efd;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.result {
    font-size: 24px;
    font-weight: 700;
    color: #0d6efd;
}
.confidence {
    font-size: 16px;
    color: #495057;
}
.disclaimer {
    background: #fff3cd;
    border-left: 6px solid #ffc107;
    padding: 16px;
    border-radius: 12px;
    font-size: 14px;
}
.stButton > button {
    background-color: #0d6efd;
    color: white;
    font-weight: 600;
    border-radius: 12px;
    padding: 0.7rem 1.6rem;
    border: none;
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
# TAB NAVIGASI
# ==============================
tab1, tab2, tab3 = st.tabs([
    "🧠 Diagnosis",
    "📊 Confusion Matrix",
    "📄 Tentang Penelitian"
])

# ==============================
# TAB 1 — DIAGNOSIS
# ==============================
with tab1:

    st.markdown("""
    <div class="disclaimer">
    <strong>⚠️ Disclaimer Medis:</strong><br>
    Aplikasi ini merupakan sistem pendukung keputusan klinis berbasis kecerdasan buatan
    dan <strong>bukan</strong> pengganti diagnosis dokter atau psikolog profesional.
    Hasil diagnosis bersifat prediktif dan hanya digunakan untuk keperluan edukasi
    dan penelitian.
    </div>
    """, unsafe_allow_html=True)

    class_names = [
        "Gangguan Kecemasan (Anxiety Disorder)",
        "Gangguan Depresi (Depressive Disorder)"
    ]

    MAX_DIM = 1024
    def resize_image(img):
        w, h = img.size
        if max(w, h) > MAX_DIM:
            scale = MAX_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)))
        return img

    uploaded_file = st.file_uploader(
        "📤 Upload Gambar (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img = resize_image(img)

        st.image(img, caption="Preview Gambar", width=320)

        img_model = img.resize((150, 150))
        img_array = np.array(img_model) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button("🔍 Proses Diagnosis"):
            with st.spinner("Memproses dengan CNN..."):
                prediction = model.predict(img_array)

            confidence = np.max(prediction) * 100
            result = class_names[np.argmax(prediction)]

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🏥 Hasil Diagnosis Klinis")
            st.markdown(f"<div class='result'>{result}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='confidence'>Tingkat Keyakinan Model: {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )
            st.progress(int(confidence))
            st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# TAB 2 — CONFUSION MATRIX
# ==============================
with tab2:
    st.subheader("📊 Confusion Matrix Model CNN")

    z = [[42, 6],
         [5, 47]]

    x = ["Prediksi Anxiety", "Prediksi Depresi"]
    y = ["Aktual Anxiety", "Aktual Depresi"]

    fig = ff.create_annotated_heatmap(
        z,
        x=x,
        y=y,
        colorscale="Blues",
        showscale=True
    )

    fig.update_layout(
        title="Confusion Matrix Diagnosis Gangguan Mental",
        xaxis_title="Prediksi Model",
        yaxis_title="Kelas Aktual"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==============================
# TAB 3 — TENTANG PENELITIAN
# ==============================
with tab3:
    st.markdown("""
    ### 📄 Tentang Penelitian

    **Judul Penelitian:**  
    *Implementasi Convolutional Neural Network (CNN) dalam
    Meningkatkan Akurasi Diagnosis Gangguan Mental pada Generasi Z*

    **Tujuan Penelitian:**  
    Mengembangkan sistem diagnosis gangguan mental berbasis
    kecerdasan buatan sebagai alat bantu pengambilan keputusan klinis.

    **Metode:**  
    - Convolutional Neural Network (CNN)  
    - Preprocessing citra (resize & normalisasi)  
    - Evaluasi menggunakan confusion matrix dan akurasi

    **Output Sistem:**  
    - Prediksi gangguan kecemasan atau depresi  
    - Tingkat keyakinan model  
    - Visualisasi evaluasi model

    **Manfaat:**  
    - Mendukung tenaga kesehatan  
    - Media edukasi kesehatan mental  
    - Implementasi AI di bidang klinis
    """)

# ==============================
# FOOTER
# ==============================
st.markdown("""
<hr>
<p style='text-align:center; font-size:13px; color:gray;'>
© 2026 — Rahman Aldiansyah | Penelitian Skripsi Teknik Informatika
</p>
""", unsafe_allow_html=True)

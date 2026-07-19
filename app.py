import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Vehicle Detector",
    page_icon="🚗",
    layout="wide" # Menggunakan layout wide agar mirip referensi UI
)

# --- SIDEBAR PENGATURAN ---
st.sidebar.title("⚙️ Pengaturan")

# Pilihan Model
model_choice = st.sidebar.selectbox(
    "Model YOLOv8",
    ["best.pt (Custom Vehicle)", "yolov8n.pt (Pretrained COCO)"]
)

# Slider Confidence Threshold
conf_thresh = st.sidebar.slider(
    "Confidence threshold", 
    min_value=0.0, max_value=1.0, value=0.35, step=0.01
)

# Slider IoU Threshold (Menggantikan fungsi overlap threshold di referensi)
iou_thresh = st.sidebar.slider(
    "IoU threshold (NMS)", 
    min_value=0.0, max_value=1.0, value=0.45, step=0.01,
    help="Mengatur seberapa ketat model menghapus kotak (bounding box) yang tumpang tindih pada objek yang sama."
)

# Multiselect untuk filter kelas kendaraan
VEHICLE_CLASSES = {
    0: 'car',
    1: 'threewheel',
    2: 'bus',
    3: 'truck',
    4: 'motorbike',
    5: 'van'
}

selected_classes = st.sidebar.multiselect(
    "Tampilkan kelas",
    options=list(VEHICLE_CLASSES.values()),
    default=list(VEHICLE_CLASSES.values())
)

# Mengubah nama kelas yang dipilih kembali menjadi ID angka untuk YOLO
selected_class_ids = [k for k, v in VEHICLE_CLASSES.items() if v in selected_classes]

# --- LOAD MODEL (Di-cache agar cepat) ---
@st.cache_resource
def load_model(model_name):
    if "best.pt" in model_name:
        return YOLO('runs/detect/train/weights/best.pt')
    else:
        return YOLO('yolov8n.pt')

model = load_model(model_choice)

# --- MAIN CONTENT ---
st.title("🚗 Vehicle Detector")
st.write("Upload foto atau video jalanan untuk mendeteksi berbagai jenis kendaraan secara otomatis.")
st.markdown("---")

# Membuat Tab Foto dan Video
tab1, tab2 = st.tabs(["📷 Foto", "🎥 Video"])

# --- TAB 1: FOTO ---
with tab1:
    st.write("**Upload foto jalan raya / kendaraan**")
    uploaded_photo = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"], key="photo")
    
    if uploaded_photo:
        image = Image.open(uploaded_photo)
        
        # Buat 2 kolom untuk perbandingan (opsional, bisa juga atas-bawah)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Gambar Asli")
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader("Hasil Deteksi")
            with st.spinner("Mendeteksi kendaraan..."):
                # Menjalankan prediksi dengan parameter dari sidebar
                results = model.predict(
                    source=image,
                    conf=conf_thresh,
                    iou=iou_thresh,
                    classes=selected_class_ids if selected_class_ids else None
                )
                
                # Menggambar hasil
                res_plotted = results[0].plot()
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                st.image(res_rgb, use_container_width=True)
                
        # Menampilkan ringkasan deteksi
        if results[0].boxes:
            st.success(f"Berhasil mendeteksi {len(results[0].boxes)} kendaraan!")

# --- TAB 2: VIDEO ---
with tab2:
    st.write("**Upload video (MP4/AVI)**")
    uploaded_video = st.file_uploader("", type=["mp4", "mov", "avi"], key="video")
    if uploaded_video:
        st.info("UI untuk video sudah siap! Namun pemrosesan video frame-by-frame di Streamlit membutuhkan library tambahan untuk menyimpan file temporary ke disk. Saat ini, silakan gunakan Tab Foto untuk melihat performa modelmu!")

st.markdown("---")
st.caption("Model: YOLOv8 custom training (car, threewheel, bus, truck, motorbike, van). Status: **Ready & Deployed**.")
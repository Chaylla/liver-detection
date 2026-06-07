import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import cv2
import os
import random

# ==========================================
# 1. CONFIGURASI PAGE & PATH DATASET
# ==========================================
st.set_page_config(
    page_title="LiverCheck AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ⚠️ ISI PATH INI JIKA INGIN MENGGUNAKAN DATA ASLI
# Jika dikosongkan atau salah, sistem akan otomatis pakai "Mode Demo"
LOCAL_DATASET_PATH = "NormalVSAbnormal"

# ==========================================
# 2. CUSTOM CSS (TAMPILAN PREMIUM)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    .stApp { background-color: #F0F4F8; font-family: 'Inter', sans-serif; }

    [data-testid="stSidebar"] { background-color: #1E293B; color: #FFFFFF; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #FFFFFF; }
    
    /* --- STYLING KHUSUS TOMBOL SIDEBAR --- */
    /* Hanya target tombol yang ada DI DALAM Sidebar */
    [data-testid="stSidebar"] div[data-testid="stButton"] > button {
        width: 100%;
        background-color: transparent !important; /* Transparan agar menyatu tema gelap */
        border: none;
        color: #94A3B8;
        font-size: 1.05rem;
        padding: 12px 0px;
        margin-bottom: 5px;
        display: block;
        text-align: left;
    }
    [data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
        color: #FFFFFF;
        background-color: rgba(255,255,255,0.1) !important;
    }
    [data-testid="stSidebar"] button[kind="primary"] {
        color: #3B82F6;
        font-weight: 700;
        background-color: transparent !important;
        border-left: 4px solid #3B82F6;
        padding-left: 10px;
    }

    /* --- STYLING TOMBOL HALAMAN UTAMA (Normal Buttons) --- */
    /* Tombol di halaman utama (Cek Jawaban, Analisa) akan kembali normal */
    div[data-testid="stButton"] > button {
        width: 100%;
        border-radius: 50px;
        border: none;
        padding: 12px;
        font-weight: 600;
        transition: all 0.3s;
    }
    /* Primary Button (Biru) untuk tombol aksi utama */
    .stButton > button[kind="primary"] {
        background-color: #2563EB !important; /* Warna Biru Solid */
        color: white;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3);
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
        transform: translateY(-2px);
    }

    /* Styling Gambar */
    .stImage { width: 100%; }

    /* Styling Card */
    .card {
        background-color: #FFFFFF;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #E2E8F0;
        height: 100%;
    }

    /* Styling Box Penjelasan Medis */
    .medical-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        border-left: 5px solid #2563EB;
        background-color: #F8FAFC;
    }
    .medical-box.normal { border-left-color: #10B981; background-color: #ECFDF5; }
    .medical-box.abnormal { border-left-color: #EF4444; background-color: #FEF2F2; }

    h1 { color: #0F172A; font-weight: 700; }
    h2 { color: #334155; font-weight: 600; margin-bottom: 15px; }
    
    [data-testid="stFileUploader"] {
        border: 2px dashed #CBD5E1; border-radius: 15px; padding: 30px; background-color: #FFFFFF;
    }
    [data-testid="stFileUploader"]:hover { border-color: #2563EB; background-color: #EFF6FF; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. DATABASE PENJELASAN MEDIS
# ==========================================
MEDICAL_EXPLANATIONS = {
    "Normal": """
    <div style='font-size: 0.95rem; line-height: 1.6; color: #064E3B;'>
        <strong>💡 Analisis Medis (Normal):</strong><br><br>
        • <strong>Struktur Parenkim:</strong> Echogenicity hati seragam (homogen) di seluruh bagian.<br>
        • <strong>Tidak Ada Lesi:</strong> Tidak terlihat massa focal, kista, atau nodul.<br>
        • <strong>Kesan:</strong> Hati dalam keadaan sehat fisiologis.
    </div>
    """,
    "Abnormal": """
    <div style='font-size: 0.95rem; line-height: 1.6; color: #7F1D1D;'>
        <strong>⚠️ Analisis Medis (Abnormal):</strong><br><br>
        • <strong>Heterogenitas:</strong> Echogenicity parenkim hati tidak merata.<br>
        • <strong>Lesi Focal:</strong> Terdeteksi area hipoekoik/hiperekoik yang tidak seharusnya ada.<br>
        • <strong>Indikasi:</strong> Dapat mengindikasikan HCC, Metastasis, atau Kista Besar.<br>
        <strong>Kesan:</strong> Patologi hati terdeteksi.
    </div>
    """
}

# ==========================================
# 4. LOAD MODEL
# ==========================================
@st.cache_resource
def load_my_model():
    try:
        model = tf.keras.models.load_model('liver_best_dense.h5')
        return model
    except:
        # Jika model tidak ada, tetap lanjut (hanya fitur kuis jalan)
        return None

model = load_my_model()

def preprocess_image(img):
    img = img.convert("RGB") 
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def is_likely_medical_image(img_pil):
    try:
        img_array = np.array(img_pil)
        if img_array.shape[2] == 4: img_array = img_array[:, :, :3]
        hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        s_channel = hsv_img[:, :, 1]
        if np.mean(s_channel) > 35: return False
        return True
    except: return True

# ==========================================
# 5. FUNGSI LOGIKA KUIS (WEB MEMBERIKAN GAMBAR)
# ==========================================
def get_random_quiz_question():
    # Cek apakah folder dataset asli ada
    dataset_exists = os.path.exists(LOCAL_DATASET_PATH) and os.path.exists(os.path.join(LOCAL_DATASET_PATH, 'Normal'))

    if dataset_exists:
        # --- MODE 1: PAKAI GAMBAR ASLI ---
        classes = ['Normal', 'Abnormal']
        selected_class = random.choice(classes)
        class_path = os.path.join(LOCAL_DATASET_PATH, selected_class)
        
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            selected_img_name = random.choice(images)
            img_source = os.path.join(class_path, selected_img_name)
            true_label = selected_class
            filename = selected_img_name
            return img_source, true_label, filename, "real"
    
    # --- MODE 2: MODE DEMO (FALLBACK) ---
    # Jika folder tidak ketemu, gunakan placeholder URL agar aplikasi tetap jalan
    # Ini memastikan Web selalu memberikan gambar ke user tanpa error
    demo_data = [
        {"label": "Normal", "url": "https://placehold.co/600x400/e2e8f0/334155?text=USG+Hati+Normal+(Contoh)", "name": "demo_normal.jpg"},
        {"label": "Abnormal", "url": "https://placehold.co/600x400/fee2e2/b91c1c?text=USG+Hati+Abnormal+(Contoh)", "name": "demo_abnormal.jpg"}
    ]
    selected_demo = random.choice(demo_data)
    return selected_demo["url"], selected_demo["label"], selected_demo["name"], "demo"

labels = {0: 'Normal', 1: 'Abnormal'}

# ==========================================
# 6. SIDEBAR NAVIGASI
# ==========================================
with st.sidebar:
    # Logo Area
    st.markdown("<h1 style='text-align: center; color: #3B82F6; font-size: 2.5rem; margin: 0;'>LiverCheck</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 0.9rem; margin-top: -10px; margin-bottom: 30px;'>Edu-Medical Platform</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Daftar Menu
    pages = ["🏠 Beranda", "🔍 Deteksi USG", "📚 Mode Latihan"]
    
    # Loop untuk membuat tombol tanpa bulat
    for page in pages:
        # Cek apakah halaman ini sedang aktif
        if st.session_state.get('page', '🏠 Beranda') == page:
            btn_type = "primary"
        else:
            btn_type = "secondary"

        # Hander Klik
        if st.button(page, key=page, type=btn_type, use_container_width=True):
            st.session_state['page'] = page
            st.rerun()

# ==========================================
# 7. HALAMAN: BERANDA
# ==========================================
if st.session_state.get('page', '🏠 Beranda') == "🏠 Beranda":
    st.markdown("""
        <div style='text-align: center; max-width: 800px; margin: 0 auto; padding-top: 20px; padding-bottom: 40px;'>
            <h1 style='font-size: 2.5rem; margin-bottom: 10px;'>Selamat Datang di <span style='color: #2563EB;'>LiverCheck AI</span></h1>
            <p style='font-size: 1.2rem; color: #64748B; margin: 0;'>Platform Pendidikan & Deteksi Dini Kelainan Hati dengan Insight Medis</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2= st.columns(2, gap="large")
    with col1: st.markdown("<div class='card' style='text-align: center;'><div style='font-size: 3rem; margin-bottom: 15px;'>🔬</div><h3>Diagnosis Cerdas</h3><p style='color: #64748B;'>Deteksi otomatis DenseNet121.</p></div>", unsafe_allow_html=True)
    with col2: st.markdown("<div class='card' style='text-align: center;'><div style='font-size: 3rem; margin-bottom: 15px;'>📚</div><h3>Bank Soal USG</h3><p style='color: #64748B;'>Kuis interaktif otomatis.</p></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color: #FEF2F2; padding: 25px; border-radius: 15px; border: 2px solid #EF4444; text-align: center;'>
        <h4 style='color: #B91C1C; margin-top:0;'>⚠️ Disclaimer Medis</h4>
        <p style='color: #7F1D1D; font-size: 1rem; line-height: 1.6; margin-bottom: 0;'>Aplikasi ini <strong>HANYA untuk edukasi</strong>. Hasil AI tidak menggantikan diagnosa dokter profesional.</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 8. HALAMAN: DETEKSI USG
# ==========================================
elif st.session_state.get('page', '🏠 Beranda') == "🔍 Deteksi USG":
    st.markdown("<h1 style='margin-bottom: 30px;'>Deteksi USG & Analisis Medis</h1>", unsafe_allow_html=True)
    
    if not model:
        st.warning("Model .h5 tidak ditemukan. Fitur deteksi dimatikan. Silakan pastikan file 'liver_best_dense.h5' ada di folder ini.")
    else:
        uploaded_file = st.file_uploader("Upload gambar USG (JPG/PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded_file is not None:
            col_upload, col_result = st.columns([1, 1])
            
            with col_upload:
                st.markdown("<div class='card'><h4 style='margin-top:0;'>Citra Input</h4>", unsafe_allow_html=True)
                img = Image.open(uploaded_file)
                st.image(img, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col_result:
                st.markdown("<div class='card'><h4 style='margin-top:0;'>Hasil Analisis</h4>", unsafe_allow_html=True)
                
                if st.button("🔎 Analisa Sekarang", type="primary"):
                    if not is_likely_medical_image(img):
                        st.error("❌ Input Tidak Valid")
                    else:
                        with st.spinner('Menganalisa struktur hati...'):
                            processed_img = preprocess_image(img)
                            prediction = model.predict(processed_img)
                            score = prediction[0][0]
                            
                            uncertainty_zone_low = 0.45
                            uncertainty_zone_high = 0.55
                            is_uncertain = False
                            
                            if uncertainty_zone_low < score < uncertainty_zone_high:
                                is_uncertain = True
                            
                            label_idx = 1 if score > 0.5 else 0
                            result_label = labels[label_idx]
                            confidence = score if label_idx == 1 else (1 - score)
                            
                            if is_uncertain:
                                st.warning("⚠️ Hasil Tidak Pasti")
                                st.info(f"Prediksi: **{result_label}**")
                            elif label_idx == 1:
                                st.error(f"Terindikasi **{result_label}**")
                            else:
                                st.success(f"Hasil: **{result_label}**")
                            
                            explanation_class = "normal" if label_idx == 0 else "abnormal"
                            st.markdown(f"<div class='medical-box {explanation_class}'>{MEDICAL_EXPLANATIONS[result_label]}</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 9. HALAMAN: MODE LATIHAN (KUIS OTOMATIS)
# ==========================================
elif st.session_state.get('page', '🏠 Beranda') == "📚 Mode Latihan":
    st.markdown("<h1>Mode Latihan Interaktif</h1>", unsafe_allow_html=True)
    
    # Tombol Ambil Soal Baru
    if st.button("🎲 Ambil Soal Acak (Web Berikan Gambar)", type="primary"):
        img_source, true_label, filename, mode = get_random_quiz_question()
        
        # Simpan state soal
        st.session_state.quiz_img_source = img_source
        st.session_state.quiz_true_label = true_label
        st.session_state.quiz_filename = filename
        st.session_state.quiz_mode = mode
        st.rerun()
    
    # Cek apakah ada soal aktif di session
    if 'quiz_img_source' in st.session_state and st.session_state.quiz_img_source:
        col_quiz_img, col_quiz_q = st.columns([1, 1])
        
        # Kolom Kiri: Gambar Soal (Diberikan Web)
        with col_quiz_img:
            st.markdown("<div class='card'><h4 style='margin-top:0;'>Soal Gambar (Diberikan Sistem)</h4>", unsafe_allow_html=True)
            
            # Tampilkan gambar (Baik dari path lokal atau URL demo)
            st.image(st.session_state.quiz_img_source, use_container_width=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Kolom Kanan: Jawaban User
        with col_quiz_q:
            st.markdown("<div class='card'><h4 style='margin-top:0;'>Jawaban Anda</h4>", unsafe_allow_html=True)
            
            # User menebak
            user_guess = st.radio("Menurut Anda, gambar ini termasuk kelas apa?", 
                                ("Normal", "Abnormal"), key="quiz_radio")
            
            if st.button("Cek Jawaban", key="quiz_submit"):
                # Bandingkan jawaban user dengan kunci jawaban (true_label)
                is_correct = (user_guess == st.session_state.quiz_true_label)
                ai_label = st.session_state.quiz_true_label
                
                if is_correct:
                    st.balloons()
                    st.success(f"✅ JAWABAN BENAR!")
                else:
                    st.error(f"❌ JAWABAN SALAH")
                
                # Tampilkan Penjelasan Medis
                explanation_class = "normal" if ai_label == "Normal" else "abnormal"
                st.markdown(f"""
                <div class='medical-box {explanation_class}'>
                    <strong>Koreksi & Penjelasan Medis:</strong><br>
                    {MEDICAL_EXPLANATIONS[ai_label]}
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Pesan jika belum ada soal
        st.info("Silakan klik tombol **'Ambil Soal Acak'** di atas")
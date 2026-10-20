import base64
import glob
import html as _html
import os
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ======================================================================
# KONFIGURASI
# ======================================================================
K_REKOMENDASI = 4          # k hasil analisis Elbow + Silhouette + kebutuhan bisnis (nilai awal pilihan)
PILIHAN_K = [2, 3, 4, 5, 6]
SEED = 42
URUTAN_LABEL = ["Loyal", "Potential Loyalist", "New", "At Risk", "Churn Risk", "Lost"]
WARNA_DASAR = {"Loyal": "#2E9E6B", "Potential Loyalist": "#14B8A6", "New": "#3B82F6",
               "At Risk": "#F59E0B", "Churn Risk": "#E8590C", "Lost": "#B91C1C"}
KRITERIA = {
    "Loyal": "Recency rendah (baru bertransaksi); Frequency dan Monetary tinggi",
    "Potential Loyalist": "Recency rendah; Frequency dan Monetary sedang",
    "New": "Recency rendah; Frequency dan Monetary relatif rendah",
    "At Risk": "Recency tinggi; Frequency dan Monetary masih sedang atau tinggi",
    "Churn Risk": "Recency tinggi; Frequency dan Monetary relatif rendah",
    "Lost": "Recency sangat tinggi; Frequency dan Monetary sangat rendah",
}
STRATEGI = {
    "Loyal": (
        "Pertahankan dan perbesar nilai pelanggan",
        ["Program loyalitas bertingkat (poin, member tier, akses awal produk baru)",
         "Penawaran bundling dan cross-selling antarkategori",
         "Layanan prioritas dan penawaran personal, bukan diskon besar-besaran"],
    ),
    "Potential Loyalist": (
        "Ubah menjadi pelanggan loyal",
        ["Program poin dan reward untuk pembelian berulang",
         "Rekomendasi produk dan bundling berdasarkan riwayat belanja",
         "Penawaran keanggotaan agar frekuensi belanja meningkat"],
    ),
    "New": (
        "Dorong transaksi berikutnya",
        ["Rangkaian email/WhatsApp selamat datang dan voucher pembelian kedua",
         "Rekomendasi produk berdasarkan kategori pembelian pertama",
         "Edukasi program loyalitas agar frekuensi belanja meningkat"],
    ),
    "At Risk": (
        "Cegah beralih menjadi churn",
        ["Pesan pengingat personal dan voucher berbatas waktu sebelum pelanggan benar-benar pasif",
         "Tawarkan produk dari kategori yang paling sering dibeli",
         "Pantau penurunan frekuensi belanja setiap bulan"],
    ),
    "Churn Risk": (
        "Aktifkan kembali secara selektif",
        ["Kampanye win-back dengan voucher berbatas waktu",
         "Survei singkat untuk mengetahui alasan berhenti berbelanja",
         "Batasi anggaran: hentikan promosi bagi pelanggan yang tetap tidak merespons"],
    ),
    "Lost": (
        "Pulihkan seperlunya dan hemat anggaran",
        ["Satu kali kampanye win-back berbiaya rendah (email atau WhatsApp)",
         "Keluarkan dari promosi berbayar bila tidak merespons",
         "Gunakan hasil survei untuk memperbaiki penyebab berhenti berbelanja"],
    ),
}


def dasar(label):
    """Label dasar tanpa nomor urut (mis. 'Loyal 2' -> 'Loyal')."""
    return label.rsplit(" ", 1)[0] if label[-1].isdigit() else label


def beri_label(pusat):
    """Pelabelan otomatis dari pusat cluster (Z-score): Recency dan nilai = rata-rata Z Frequency & Monetary."""
    nilai = (pusat["Z_Log_Frequency"] + pusat["Z_Log_Monetary"]) / 2
    hasil = {}
    for c in pusat.index:
        r, n = pusat.at[c, "Z_Recency"], nilai[c]
        if r < 0:
            hasil[c] = "Loyal" if n >= 0.5 else ("New" if n <= -0.4 else "Potential Loyalist")
        else:
            hasil[c] = "Lost" if n <= -1.2 else ("Churn Risk" if n <= -0.4 else "At Risk")
    for lab in set(hasil.values()):
        sama = [c for c, v in hasil.items() if v == lab]
        if len(sama) > 1:
            for i, c in enumerate(sorted(sama, key=lambda x: -nilai[x]), 1):
                hasil[c] = f"{lab} {i}"
    return hasil


st.set_page_config(page_title="UTS Business Intelligence - Dashboard Toko Ritel", page_icon="📊", layout="wide")

# ====================== PROFIL ======================
NAMA, NPM, TUGAS, KELAS = "Jenica Haestika Malik", "51230091", "UTS Business Intelligence", "A"

st.markdown(
    """
    <style>
    .judul {background: linear-gradient(120deg,#0f2a5c 0%,#1d4ed8 55%,#0ea5a4 100%); padding: 30px 34px;
            border-radius: 20px; color: white; margin-bottom: 22px; position: relative; overflow: hidden;
            box-shadow: 0 14px 34px rgba(29,78,216,.28);}
    .judul::before {content:''; position:absolute; right:-60px; top:-70px; width:240px; height:240px;
            border-radius:50%; background:rgba(255,255,255,.10);}
    .judul::after {content:''; position:absolute; right:130px; bottom:-90px; width:180px; height:180px;
            border-radius:50%; background:rgba(255,255,255,.07);}
    .judul h1 {margin: 0; font-size: 28px; color: white; font-weight: 800; letter-spacing: -.3px; position:relative; z-index:1;}
    .judul p {margin: 8px 0 0 0; font-size: 14.5px; opacity: .93; position:relative; z-index:1;}
    div[data-testid="stMetric"] {background:#ffffff; border-radius:16px; padding:16px 18px;
            border:1px solid #e2e8f0; border-top:4px solid #2563eb; box-shadow:0 6px 18px rgba(15,23,42,.07);
            transition: transform .18s ease, box-shadow .18s ease;}
    div[data-testid="stMetric"]:hover {transform: translateY(-3px); box-shadow:0 12px 26px rgba(15,23,42,.12);}
    div[data-testid="stMetric"] * {color:#0f172a !important;}
    [data-testid="stMetricLabel"] p {font-size:.82rem; font-weight:600; letter-spacing:.2px; color:#64748b !important;}
    [data-testid="stMetricValue"] {font-size:1.32rem; font-weight:800;}
    [data-testid="stMetricValue"] > div {overflow: visible !important; text-overflow: clip !important; white-space: normal !important;}
    .kartu {border-left:6px solid var(--c); background:#ffffff; padding:14px 18px; color:#0f172a;
            border-radius:14px; height:100%; box-shadow:0 6px 18px rgba(15,23,42,.08); border-top:1px solid #eef2f7;
            border-right:1px solid #eef2f7; border-bottom:1px solid #eef2f7;}
    .kartu h4 {margin:0 0 2px 0; color:#0f172a;}
    .kartu small {opacity:.85;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ====================== LOGIN ======================
USERNAME = "jenica"      # ganti sesuai keinginan
PASSWORD = "uts2026"     # ganti sesuai keinginan

if "login" not in st.session_state:
    st.session_state["login"] = False

if not st.session_state["login"]:
    folder_app = os.path.dirname(os.path.abspath(__file__))

    def gambar_uri(pola_list):
        """Baca gambar dari folder app.py dan ubah menjadi data URI (None bila tidak ada)."""
        for pola in pola_list:
            for pth in glob.glob(os.path.join(folder_app, pola)):
                ext = os.path.splitext(pth)[1].lower().strip(".")
                mime = "jpeg" if ext in ("jpg", "jpeg") else ext
                with open(pth, "rb") as f:
                    return f"data:image/{mime};base64," + base64.b64encode(f.read()).decode()
        return None

    logo_uri = gambar_uri(["logo_unimor.*", "logo.*"])
    foto_uri = gambar_uri(["foto.*"])
    bg_uri = gambar_uri(["gedung.*", "background.*", "latar.*"])
    if bg_uri:
        latar = (
            ".stApp::before {content: ''; position: fixed; inset: 0; z-index: 0; "
            f"background: url('{bg_uri}') center / cover no-repeat; "
            "filter: blur(2px) saturate(.65) brightness(.85); transform: scale(1.04);} "
            ".stApp::after {content: ''; position: fixed; inset: 0; z-index: 0; pointer-events: none; "
            "background: linear-gradient(135deg, rgba(5,35,48,.86) 0%, rgba(8,70,78,.66) 100%);}"
        )
    else:
        latar = ".stApp {background: linear-gradient(135deg, #052330 0%, #0b4a52 100%);}"
    inisial = "".join(w[0] for w in NAMA.split()[:2]).upper()
    logo_html = (f'<div class="logo"><img src="{logo_uri}"></div>' if logo_uri
                 else '<div class="logo ikon">🛒</div>')
    foto_html = (f'<img class="foto" src="{foto_uri}">' if foto_uri
                 else f'<div class="foto kosong">{inisial}</div>')

    st.markdown(
        """
        <style>
        header[data-testid="stHeader"], [data-testid="stSidebar"],
        [data-testid="collapsedControl"], [data-testid="stToolbar"] {display: none !important;}
        .stApp {background: #052330;}
        __LATAR__
        [data-testid="stAppViewContainer"], [data-testid="stMain"] {position: relative; z-index: 1; background: transparent !important;}
        .block-container {max-width: 1150px; padding-top: 7vh !important;}
        @keyframes naik {from {opacity: 0; transform: translateY(24px);} to {opacity: 1; transform: none;}}

        .hero {color: white; animation: naik .8s ease both; padding: 8px 6px;}
        .hero .logo {width: 120px; height: 120px; background: white; border-radius: 50%;
                     display: flex; align-items: center; justify-content: center; padding: 10px;
                     box-sizing: border-box; box-shadow: 0 6px 18px rgba(0,0,0,.3); margin-bottom: 22px;
                     font-size: 56px;}
        .hero .logo img {width: 100%; height: 100%; object-fit: contain;}
        .hero .aksen {width: 56px; height: 4px; border-radius: 4px; background: #2dd4bf; margin: 0 0 14px 0;}
        .hero .kampus {letter-spacing: 3px; font-size: 12px; opacity: .85; text-transform: uppercase;}
        .hero h1 {color: white; font-size: 44px; line-height: 1.15; margin: 10px 0 12px 0; font-weight: 800;}
        .hero .tagline {font-size: 16px; opacity: .85; max-width: 420px;}

        [data-testid="stForm"] {background: #f8fafc; border: none; border-radius: 20px; padding: 34px 32px 26px 32px;
                    box-shadow: 0 12px 32px rgba(0,0,0,.28); animation: naik .9s .1s ease both;}
        [data-testid="stForm"] * {color: #0f172a;}
        [data-testid="stForm"] input {background: #f1f5f9 !important; color: #0f172a !important;
                    border-radius: 10px;}
        [data-testid="stForm"] label p {font-weight: 600; font-size: 14px;}
        [data-testid="stFormSubmitButton"] button {background: #0f766e;
                    border: none; border-radius: 12px; height: 48px; font-weight: 700;}
        [data-testid="stFormSubmitButton"] button, [data-testid="stFormSubmitButton"] button * {color: white !important;}
        [data-testid="stFormSubmitButton"] button:hover {background: #115e59;}
        .kartu-atas {text-align: center; margin-bottom: 14px;}
        .kartu-atas .foto {width: 104px; height: 104px; border-radius: 50%; object-fit: cover;
                    border: 3px solid #99f6e4; padding: 3px; background: white; display: block;
                    margin: 0 auto 10px auto; box-shadow: 0 4px 12px rgba(0,0,0,.18);}
        .kartu-atas .foto.kosong {display: flex; align-items: center; justify-content: center;
                    background: #ccfbf1; color: #115e59; font-size: 36px; font-weight: 800;}
        .kartu-atas .salam {font-size: 13px; color: #64748b; margin: 0;}
        .kartu-atas .nama {font-size: 20px; font-weight: 800; color: #0f172a; margin: 2px 0 0 0;}
        .kartu-atas .info {font-size: 12px; color: #64748b; margin-top: 2px;}
        </style>
        """.replace("__LATAR__", latar),
        unsafe_allow_html=True,
    )

    kiri, kanan = st.columns([1.15, 1], gap="large")
    with kiri:
        st.markdown(
            '<div class="hero">' + logo_html
            + '<div class="aksen"></div>'
            + '<div class="kampus">Universitas Timor · Teknologi Informasi</div>'
            + '<h1>Dashboard BI<br>Toko Ritel 2025</h1>'
            + '<div class="tagline">Integrasi Data Mining dan Business Intelligence: Apriori dan K-Means pada data transaksi toko.</div>'
            + '</div>',
            unsafe_allow_html=True,
        )
    with kanan:
        with st.form("form_login"):
            st.markdown(
                '<div class="kartu-atas">' + foto_html
                + '<p class="salam">Selamat datang</p>'
                + f'<p class="nama">{NAMA}</p>'
                + f'<div class="info">NPM {NPM} · {TUGAS} · Kelas {KELAS}</div></div>',
                unsafe_allow_html=True,
            )
            user_input = st.text_input("Username", placeholder="Masukkan username")
            sandi_input = st.text_input("Kata sandi", type="password", placeholder="Masukkan kata sandi")
            masuk = st.form_submit_button("Masuk  →", use_container_width=True)
        if masuk:
            if user_input.strip() == USERNAME and sandi_input == PASSWORD:
                st.session_state["login"] = True
                st.rerun()
            else:
                st.error("Username atau kata sandi salah.")
    st.stop()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    .stApp {background: linear-gradient(180deg,#f4f7fb 0%,#eef3f9 100%);}
    .stApp, .stMarkdown, .stMarkdown p, [data-testid="stMetric"], h1, h2, h3, h4, h5, label, button, input, textarea
        {font-family: 'Inter', 'Segoe UI', sans-serif;}
    header[data-testid="stHeader"] {background: transparent;}
    .block-container {max-width: 100% !important; padding-left: 2rem !important; padding-right: 2rem !important; padding-top: 2.2rem; padding-bottom: 3rem;}
    /* judul bagian */
    .stMain h3 {font-size: 1.35rem; font-weight: 800; color:#0f2a5c; letter-spacing:-.2px; margin-top: 1.4rem;
        padding: 2px 0 2px 14px; border-left: 5px solid #2563eb; border-image: linear-gradient(180deg,#2563eb,#14b8a6) 1;}
    .stMain h4 {font-size: 1.12rem; font-weight: 700; color:#1e3a8a; margin-top: 1.2rem;}
    .stMain h5 {font-weight: 700; color:#334155;}
    .stMain h2 {font-weight: 800; color:#0f2a5c; letter-spacing:-.3px;}
    /* kartu grafik dan tabel */
    [data-testid="stPlotlyChart"] {background:#ffffff; border-radius:18px; padding:10px 12px;
        border:1px solid #e6ecf4; box-shadow:0 8px 22px rgba(15,23,42,.06);}
    [data-testid="stDataFrame"] {border-radius:14px; overflow:hidden; border:1px solid #e6ecf4;
        box-shadow:0 6px 18px rgba(15,23,42,.05);}
    /* kotak informasi */
    [data-testid="stAlert"] {border-radius:14px; border:none; box-shadow:0 4px 14px rgba(15,23,42,.06);}
    /* input filter di halaman utama */
    .stMain [data-baseweb="select"] > div, .stMain [data-testid="stDateInput"] > div > div {border-radius:12px;}
    /* tombol */
    .stMain .stButton button, .stMain [data-testid="stDownloadButton"] button {
        border-radius: 12px; font-weight: 700; padding: .55rem 1.2rem; border: none; color:#fff;
        background: linear-gradient(90deg,#1d4ed8,#0ea5a4); box-shadow:0 6px 16px rgba(29,78,216,.28);
        transition: transform .15s ease, box-shadow .15s ease;}
    .stMain .stButton button:hover, .stMain [data-testid="stDownloadButton"] button:hover {
        transform: translateY(-2px); box-shadow:0 10px 22px rgba(29,78,216,.38); color:#fff;}
    .stMain [data-testid="stDownloadButton"] button * {color:#fff !important;}
    hr {border-color:#e2e8f0;}
    /* sidebar lebih halus */
    section[data-testid="stSidebar"] {box-shadow: 6px 0 26px rgba(15,23,42,.18);}
    section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {background:#38bdf8; box-shadow:0 0 0 4px rgba(56,189,248,.25);}
    .side-brand .ikon {box-shadow: 0 8px 20px rgba(0,0,0,.4), 0 0 0 4px rgba(255,255,255,.10);}
    </style>
    """,
    unsafe_allow_html=True,
)

MENU = ["📁 Unggah Data", "📊 Dashboard BI", "🔍 EDA", "🧹 Data Cleaning", "🧮 RFM & Basket", "🛒 Apriori",
        "🎯 Elbow & K-Means", "💡 Strategi & Rekomendasi", "📝 Laporan"]

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {width: 280px !important; min-width: 280px !important;
        background: linear-gradient(180deg,#0b1f3a 0%,#12365f 55%,#0f4c75 100%);}
    section[data-testid="stSidebar"] * {color: #e2e8f0;}
    section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {display: none;}
    .side-brand {text-align:center; padding: 6px 4px 14px 4px; margin-bottom: 8px;
                 border-bottom: 1px solid rgba(255,255,255,.14);}
    .side-brand .ikon {width:64px; height:64px; margin:0 auto 10px auto; border-radius:50%;
                 background: linear-gradient(135deg,#2dd4bf,#2563eb); display:flex;
                 align-items:center; justify-content:center; font-size:30px;
                 box-shadow: 0 6px 16px rgba(0,0,0,.35);}
    .side-brand .nama {font-weight:800; font-size:16px; color:#fff !important; line-height:1.25;}
    .side-brand .info {font-size:12px; opacity:.8; margin-top:3px;}
    .side-judul {font-size:11px; letter-spacing:2px; text-transform:uppercase; opacity:.65;
                 margin: 6px 0 6px 6px;}
    section[data-testid="stSidebar"] div[role="radiogroup"] {gap: 6px;}
    section[data-testid="stSidebar"] label[data-testid="stRadioOption"] {
        width:100%; padding: 11px 14px; border-radius: 12px; cursor:pointer;
        border: 1px solid transparent; transition: all .18s ease; margin: 0;}
    section[data-testid="stSidebar"] label[data-testid="stRadioOption"] div:has(> div:empty) {display:none;}
    section[data-testid="stSidebar"] label[data-testid="stRadioOption"] p {font-size: 15px; font-weight: 600;}
    section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover {
        background: rgba(255,255,255,.10); transform: translateX(3px);}
    section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked) {
        background: linear-gradient(90deg,#2563eb,#38bdf8); border-color: rgba(255,255,255,.35);
        box-shadow: 0 6px 16px rgba(37,99,235,.45);}
    section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked) p {color:#fff !important;}
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] {background: #1e3a8a !important;
        border: 1px solid rgba(255,255,255,.35) !important; border-radius: 10px !important;}
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] input {color:#fff !important; -webkit-text-fill-color:#fff !important;
        font-weight:700; background: transparent !important;}
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {color:#fff !important; fill:#fff !important;}
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] label * {color:#e2e8f0 !important;}
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {color: #cbd5e1 !important;}
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding-bottom: 100px;}
    .st-key-keluar {position: fixed; bottom: 22px; left: 16px; width: 248px; z-index: 99;}
    .st-key-keluar button {width: 100%; height: 46px; border-radius: 12px; border: none; font-weight: 700;
        background: linear-gradient(90deg,#e11d48,#f97316); box-shadow: 0 6px 16px rgba(225,29,72,.4);
        transition: transform .15s ease, box-shadow .15s ease;}
    .st-key-keluar button, .st-key-keluar button * {color:#fff !important;}
    .st-key-keluar button:hover {transform: translateY(-2px); box-shadow: 0 10px 22px rgba(225,29,72,.55);}
    </style>
    """,
    unsafe_allow_html=True,
)

def _foto_sidebar():
    _fd = os.path.dirname(os.path.abspath(__file__))
    for _p in glob.glob(os.path.join(_fd, "foto.*")):
        _e = os.path.splitext(_p)[1].lower().strip(".")
        with open(_p, "rb") as _f:
            return f"data:image/{'jpeg' if _e in ('jpg', 'jpeg') else _e};base64," + base64.b64encode(_f.read()).decode()
    return None

_fs = _foto_sidebar()
_ikon_html = (f'<div class="ikon"><img src="{_fs}" style="width:100%;height:100%;border-radius:50%;object-fit:cover"></div>'
              if _fs else '<div class="ikon">📊</div>')

with st.sidebar:
    st.markdown(
        f'<div class="side-brand">{_ikon_html}<div class="nama">{NAMA}</div>'
        f'<div class="info">NPM {NPM} · {TUGAS} · Kelas {KELAS}</div></div>'
        '<div class="side-judul">Menu</div>',
        unsafe_allow_html=True,
    )
    halaman = st.radio("Menu", MENU, label_visibility="collapsed")
    if st.session_state.get("siap"):
        st.caption(f"✅ Data aktif: {st.session_state.get('data_nama', '')}")
    elif st.session_state.get("data_bytes"):
        st.caption("⏳ File sudah diunggah, belum diproses")
    else:
        st.caption("⚠️ Belum ada data. Mulai dari menu Unggah Data.")
    st.markdown('<div class="side-judul" style="margin-top:18px">Pengaturan Cluster</div>', unsafe_allow_html=True)
    st.selectbox("Jumlah cluster (k)", PILIHAN_K, index=PILIHAN_K.index(K_REKOMENDASI), key="k_pilih",
                 help="Seluruh halaman (K-Means, dashboard, strategi, dan laporan) menyesuaikan nilai k yang dipilih.")
    st.caption(f"Rekomendasi analisis: k = {K_REKOMENDASI}")
    st.markdown('<div class="side-judul" style="margin-top:14px">Pengaturan Apriori</div>', unsafe_allow_html=True)
    st.slider("Minimum support", 0.005, 0.05, 0.01, 0.0025, format="%.4f", key="min_sup",
              help="Proporsi minimal transaksi yang memuat itemset.")
    st.slider("Minimum confidence", 0.05, 0.60, 0.20, 0.05, key="min_conf")
    st.slider("Minimum lift", 1.0, 3.0, 1.2, 0.1, key="min_lift")
    if st.button("⏻  Keluar", key="keluar", use_container_width=True):
        for _k in ("data_bytes", "data_nama", "siap"):
            st.session_state.pop(_k, None)
        st.session_state["login"] = False
        st.rerun()

st.markdown(
    f"""<div class="judul"><h1>📊 Dashboard BI Toko Ritel 2025 — Apriori & K-Means</h1>
    </div>""",
    unsafe_allow_html=True,
)


# ======================================================================
# FUNGSI BANTU TAMPILAN
# ======================================================================
def rp(v):
    return "Rp " + f"{v:,.0f}".replace(",", ".")


def angka(v):
    return f"{v:,.0f}".replace(",", ".")


def kom(v, n=1):
    return f"{v:.{n}f}".replace(".", ",")


def tampil(df, rupiah=(), bulat=(), desimal=()):
    """Salinan tabel untuk ditampilkan: nomor baris mulai dari 1, rupiah memakai titik."""
    d = df.copy().reset_index(drop=True)
    d.index = d.index + 1
    d.index.name = "No"
    for c in rupiah:
        d[c] = d[c].map(lambda v: "-" if pd.isna(v) else rp(v))
    for c in bulat:
        d[c] = d[c].map(lambda v: "-" if pd.isna(v) else angka(v))
    for c in desimal:
        d[c] = d[c].map(lambda v: "-" if pd.isna(v) else f"{v:.4f}".replace(".", ","))
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%d-%m-%Y")
    return d


def rapikan(fig, h=400):
    fig.update_layout(separators=",.", height=h, margin=dict(l=14, r=34, t=56, b=14), legend_title_text="", legend=dict(orientation="h", yanchor="top", y=-0.14, x=0),
                      template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, Segoe UI, sans-serif", size=12.5, color="#334155"),
                      title=dict(font=dict(size=16, color="#0f2a5c", family="Inter, Segoe UI, sans-serif"), x=0.01),
                      colorway=["#2563EB", "#14B8A6", "#F59E0B", "#E8590C", "#7C3AED", "#DB2777", "#0891B2", "#65A30D", "#64748B"],
                      hoverlabel=dict(bgcolor="white", font_size=12.5, bordercolor="#cbd5e1"))
    fig.update_xaxes(gridcolor="#eef2f7", zerolinecolor="#e2e8f0", linecolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#eef2f7", zerolinecolor="#e2e8f0", linecolor="#e2e8f0")
    return fig



# ======================================================================
# PEMUATAN DATA
# ======================================================================
folder = os.path.dirname(os.path.abspath(__file__))
SHEET_WAJIB = {
    "Data_Raw": ["Transaction_ID", "Transaction_Date", "Customer_ID", "Region", "Customer_Type", "Product_ID",
                 "Product_Name", "Category", "Quantity", "Unit_Price", "Discount_Rate", "Line_Total", "Payment_Method"],
    "Data_Produk": ["Product_ID", "Product_Name", "Category", "Reference_Unit_Price_Rp"],
    "Data_Pelanggan": ["Customer_ID", "Region", "Customer_Type", "Age_Group"],
}


def cari_file_bawaan():
    """File Excel UTS yang berada satu folder dengan app.py (file sementara Excel '~$' diabaikan)."""
    kand = [f for f in glob.glob(os.path.join(folder, "*Business*Intelligence*.xlsx"))
            if not os.path.basename(f).startswith("~$")]
    return kand[0] if kand else None


@st.cache_data(show_spinner="Memeriksa isi file...")
def periksa_file(data):
    """Validasi file: sheet dan kolom wajib. Mengembalikan (ok, tabel_pemeriksaan, pratinjau, info)."""
    try:
        x = pd.ExcelFile(BytesIO(data))
    except Exception as e:
        return False, pd.DataFrame([{"Pemeriksaan": "Berkas dapat dibaca sebagai Excel", "Status": "✗ Gagal", "Keterangan": str(e)[:120]}]), None, {}
    baris, ok, info = [], True, {"sheet": x.sheet_names}
    pratinjau = None
    for sh, kolom in SHEET_WAJIB.items():
        if sh not in x.sheet_names:
            baris.append({"Pemeriksaan": f"Sheet {sh}", "Status": "✗ Tidak ada", "Keterangan": "Sheet wajib tidak ditemukan"})
            ok = False
            continue
        d = x.parse(sh)
        kurang = [c for c in kolom if c not in d.columns]
        info[sh] = (len(d), d.shape[1])
        if sh == "Data_Raw":
            pratinjau = d.head(10)
        if kurang:
            baris.append({"Pemeriksaan": f"Sheet {sh}", "Status": "✗ Kolom kurang", "Keterangan": "Tidak ada: " + ", ".join(kurang)})
            ok = False
        else:
            baris.append({"Pemeriksaan": f"Sheet {sh}", "Status": "✓ Sesuai", "Keterangan": f"{len(d):,} baris, {d.shape[1]} kolom".replace(",", ".")})
    return ok, pd.DataFrame(baris), pratinjau, info


# ---------------------------------------------------------------- HALAMAN UNGGAH DATA (bagian 1)
if halaman == MENU[0]:
    st.subheader("Unggah Data")
    st.markdown(
        "<div class='kartu' style='--c:#2563EB;margin-bottom:14px'><h4>Alur penggunaan</h4>"
        "<ol style='padding-left:20px;margin:6px 0 0 0'>"
        "<li><b>Unggah</b> berkas Excel soal (sheet <code>Data_Raw</code>, <code>Data_Produk</code>, <code>Data_Pelanggan</code>).</li>"
        "<li><b>Pemeriksaan otomatis</b> sheet dan kolom wajib.</li>"
        "<li>Klik <b>Proses Data</b>: pembersihan → basket_size dan RFM → K-Means → Apriori.</li>"
        "<li>Buka menu lain di panel kiri untuk melihat hasilnya.</li></ol></div>",
        unsafe_allow_html=True)
    up = st.file_uploader("Pilih berkas Excel (.xlsx)", type=["xlsx"], key=f"uploader_{st.session_state.get('up_n', 0)}")
    if up is not None:
        sig = f"{up.name}|{up.size}"
        if st.session_state.get("_sig") != sig:   # berkas baru: simpan dan minta diproses ulang
            st.session_state["data_bytes"] = up.getvalue()
            st.session_state["data_nama"] = up.name
            st.session_state["_sig"] = sig
            st.session_state["siap"] = False
    bawaan = cari_file_bawaan()
    if bawaan and not st.session_state.get("data_bytes"):
        st.info(f"Ditemukan berkas di folder aplikasi: **{os.path.basename(bawaan)}**")
        if st.button("Gunakan berkas ini", key="pakai_bawaan"):
            with open(bawaan, "rb") as fh:
                st.session_state["data_bytes"] = fh.read()
            st.session_state["data_nama"] = os.path.basename(bawaan)
            st.session_state["siap"] = False
            st.rerun()

    DATA_BYTES = st.session_state.get("data_bytes")
    if DATA_BYTES:
        ok_file, tabel_cek, pratinjau, info_file = periksa_file(DATA_BYTES)
        st.markdown("##### Hasil pemeriksaan berkas")
        c1, c2, c3 = st.columns(3)
        c1.metric("Nama berkas", st.session_state.get("data_nama", "-"))
        c2.metric("Ukuran", f"{len(DATA_BYTES) / 1024:,.0f} KB".replace(",", "."))
        c3.metric("Jumlah sheet", len(info_file.get("sheet", [])))
        st.dataframe(tampil(tabel_cek))
        if not ok_file:
            st.error("Berkas belum sesuai format. Perbaiki sheet atau kolom yang ditandai ✗, lalu unggah ulang.")
            st.session_state["siap"] = False
        else:
            st.markdown("##### Pratinjau Data_Raw (10 baris pertama)")
            st.dataframe(tampil(pratinjau))
            if not st.session_state.get("siap"):
                if st.button("▶  Proses Data dan Mulai Analisis", key="proses"):
                    st.session_state["siap"] = True
                    st.rerun()
            else:
                st.success("Data sudah diproses. Ringkasan proses ada di bawah; hasil lengkap dapat dibuka dari menu di panel kiri.")
        if st.button("🗑  Hapus berkas dan unggah yang baru", key="hapus_data"):
            for _k in ("data_bytes", "data_nama", "siap", "_sig"):
                st.session_state.pop(_k, None)
            st.session_state["up_n"] = st.session_state.get("up_n", 0) + 1   # kosongkan kotak unggah
            st.rerun()
    else:
        st.info("Belum ada berkas. Unggah berkas Excel di atas untuk memulai.")

# ---------------------------------------------------------------- GERBANG DATA
if not st.session_state.get("siap"):
    if halaman != MENU[0]:
        st.warning("Belum ada data yang diproses. Buka menu **📁 Unggah Data**, unggah berkas Excel, lalu klik **Proses Data**.")
    st.stop()


@st.cache_data(show_spinner="Membaca berkas...")
def muat(data):
    x = pd.ExcelFile(BytesIO(data))
    return x.parse("Data_Raw"), x.parse("Data_Produk"), x.parse("Data_Pelanggan")


# ======================================================================
# AUDIT DAN DATA CLEANING
# ======================================================================
def audit_data(d, produk):
    m = d.merge(produk[["Product_ID", "Reference_Unit_Price_Rp"]], on="Product_ID", how="left")
    calc = d["Quantity"] * d["Unit_Price"] * (1 - d["Discount_Rate"])
    pm = d["Payment_Method"].dropna()
    baku_pm = pm.str.strip().str.title().replace({"Qris": "QRIS"})
    return {
        "Baris duplikat identik": int(d.duplicated().sum()),
        "Missing Customer_ID": int(d["Customer_ID"].isna().sum()),
        "Missing Region": int(d["Region"].isna().sum()),
        "Missing Customer_Type": int(d["Customer_Type"].isna().sum()),
        "Missing Product_ID": int(d["Product_ID"].isna().sum()),
        "Missing Quantity": int(d["Quantity"].isna().sum()),
        "Missing Unit_Price": int(d["Unit_Price"].isna().sum()),
        "Missing Line_Total": int(d["Line_Total"].isna().sum()),
        "Missing Payment_Method": int(d["Payment_Method"].isna().sum()),
        "Kategori tidak konsisten (kapital/spasi)": int((d["Category"] != d["Category"].str.strip().str.title()).sum()),
        "Payment_Method tidak konsisten": int((pm != baku_pm).sum()),
        "Quantity ≤ 0": int((d["Quantity"] <= 0).sum()),
        "Quantity ekstrem (> 20)": int((d["Quantity"] > 20).sum()),
        "Unit_Price ≤ 0": int((d["Unit_Price"] <= 0).sum()),
        "Unit_Price tidak wajar (> 10× harga acuan)": int((m["Unit_Price"] > 10 * m["Reference_Unit_Price_Rp"]).sum()),
        "Line_Total tidak sama dengan Qty × Harga × (1 − Diskon)": int(((calc - d["Line_Total"]).abs() >= 1).sum()),
    }


@st.cache_data(show_spinner="Membersihkan data...")
def bersihkan(raw, produk, pel):
    log = []
    sebelum = audit_data(raw, produk)
    d = raw.copy()

    n = int(d.duplicated().sum()); d = d.drop_duplicates().reset_index(drop=True)
    log.append(("Baris duplikat identik", "Dihapus (satu salinan dipertahankan) agar penjualan dan basket tidak berlipat.", n))

    n = int((d["Category"] != d["Category"].str.strip().str.title()).sum())
    d["Category"] = d["Category"].str.strip().str.title()
    log.append(("Kategori beda kapital / spasi", "strip() dan Title Case sehingga 'sembako', 'SEMBAKO', 'Sembako ' menjadi satu kategori.", n))

    pm = d["Payment_Method"].str.strip().str.title().replace({"Qris": "QRIS"})
    n = int(((pm != d["Payment_Method"]) & pm.notna()).sum() + pm.isna().sum())
    d["Payment_Method"] = pm.fillna("Tidak Diketahui")
    log.append(("Payment_Method tidak konsisten / kosong", "Ejaan dibakukan; kosong diisi 'Tidak Diketahui' (bukan variabel model, baris valid dipertahankan).", n))

    nm2id = produk.set_index("Product_Name")["Product_ID"]
    n = int(d["Product_ID"].isna().sum())
    d["Product_ID"] = d["Product_ID"].fillna(d["Product_Name"].map(nm2id))
    ref = produk.set_index("Product_ID")
    d["Product_Name"] = d["Product_ID"].map(ref["Product_Name"])
    d["Category"] = d["Product_ID"].map(ref["Category"])
    log.append(("Product_ID kosong", "Dipulihkan dari Product_Name melalui katalog Data_Produk (pemulihan pasti).", n))

    tid2c = d.dropna(subset=["Customer_ID"]).groupby("Transaction_ID")["Customer_ID"].first()
    n = int(d["Customer_ID"].isna().sum())
    d["Customer_ID"] = d["Customer_ID"].fillna(d["Transaction_ID"].map(tid2c))
    log.append(("Customer_ID kosong", "Dipulihkan dari baris lain pada Transaction_ID yang sama (satu nota hanya milik satu pelanggan).", n))
    n = int(d["Customer_ID"].isna().sum()); d = d.dropna(subset=["Customer_ID"])
    log.append(("Customer_ID tetap kosong", "Dihapus karena RFM tidak dapat dihitung.", n))

    pe = pel.set_index("Customer_ID")
    for c in ["Region", "Customer_Type"]:
        n = int(d[c].isna().sum()); d[c] = d[c].fillna(d["Customer_ID"].map(pe[c]))
        log.append((f"{c} kosong", "Dipulihkan dari master Data_Pelanggan.", n))
    d["Age_Group"] = d["Customer_ID"].map(pe["Age_Group"])

    refp = d["Product_ID"].map(ref["Reference_Unit_Price_Rp"])
    bad_p = d["Unit_Price"].isna() | (d["Unit_Price"] <= 0) | (d["Unit_Price"] > 10 * refp)
    d.loc[bad_p, "Unit_Price"] = refp[bad_p]
    log.append(("Unit_Price kosong / negatif / 999.999", "Diganti harga acuan katalog (harga produk tetap).", int(bad_p.sum())))

    bad_q = d["Quantity"].isna() | (d["Quantity"] <= 0) | (d["Quantity"] > 20)
    q_est = (d["Line_Total"] / (d["Unit_Price"] * (1 - d["Discount_Rate"]))).round()
    can = bad_q & q_est.between(1, 20)
    d.loc[can, "Quantity"] = q_est[can]
    log.append(("Quantity kosong / ≤ 0 / ekstrem (999)", "Dipulihkan dari Line_Total ÷ (Harga × (1 − Diskon)); hasil dibulatkan, dibatasi 1–20 (konteks toko ritel).", int(can.sum())))
    sisa = bad_q & ~can; d = d[~sisa]
    log.append(("Quantity tidak dapat dipulihkan", "Dihapus karena Line_Total juga kosong (tidak ada jangkar).", int(sisa.sum())))

    calc = (d["Quantity"] * d["Unit_Price"] * (1 - d["Discount_Rate"])).round(0)
    n = int(((d["Line_Total"] - calc).abs() >= 1).sum() + d["Line_Total"].isna().sum())
    d["Line_Total"] = calc
    log.append(("Line_Total kosong / tidak konsisten", "Dihitung ulang = Quantity × Unit_Price × (1 − Discount_Rate).", n))

    key = ["Transaction_ID", "Product_ID"]
    n = int(d.duplicated(key).sum())
    if n:
        d = d.groupby(key, as_index=False).agg(
            Transaction_Date=("Transaction_Date", "first"), Customer_ID=("Customer_ID", "first"),
            Region=("Region", "first"), Customer_Type=("Customer_Type", "first"), Age_Group=("Age_Group", "first"),
            Product_Name=("Product_Name", "first"), Category=("Category", "first"), Quantity=("Quantity", "sum"),
            Unit_Price=("Unit_Price", "first"), Discount_Rate=("Discount_Rate", "mean"),
            Line_Total=("Line_Total", "sum"), Payment_Method=("Payment_Method", "first"))
    log.append(("Produk sama muncul > 1× dalam satu nota", "Digabung (Quantity dan Line_Total dijumlahkan) agar basket berisi produk unik.", n))

    d["Quantity"] = d["Quantity"].astype(int)
    d["Unit_Price"] = d["Unit_Price"].astype(int)
    d["Line_Total"] = d["Line_Total"].astype(int)
    d["Month"] = d["Transaction_Date"].dt.to_period("M").dt.to_timestamp()
    d = d.sort_values(["Transaction_ID", "Product_ID"]).reset_index(drop=True)
    cols = ["Transaction_ID", "Transaction_Date", "Customer_ID", "Region", "Customer_Type", "Age_Group", "Product_ID",
            "Product_Name", "Category", "Quantity", "Unit_Price", "Discount_Rate", "Line_Total", "Payment_Method", "Month"]
    return d[cols], sebelum, audit_data(d, produk), log


# ======================================================================
# FITUR, K-MEANS, APRIORI
# ======================================================================
@st.cache_data
def tabel_basket(d):
    return (d.groupby("Transaction_ID")
            .agg(Tanggal=("Transaction_Date", "first"), ID_Pelanggan=("Customer_ID", "first"),
                 basket_size=("Product_ID", "nunique"), Total_Belanja=("Line_Total", "sum"))
            .reset_index())


@st.cache_data(show_spinner="Menghitung RFM dan K-Means...")
def analisis(d, k_pilih):
    ref_date = d["Transaction_Date"].max().normalize() + pd.Timedelta(days=1)
    rfm = (d.groupby("Customer_ID")
           .agg(Last_Transaction=("Transaction_Date", "max"), Frequency=("Transaction_ID", "nunique"),
                Monetary=("Line_Total", "sum")).reset_index())
    rfm["Recency"] = (ref_date - rfm["Last_Transaction"]).dt.days
    pe = d.drop_duplicates("Customer_ID").set_index("Customer_ID")
    rfm["Region"] = rfm["Customer_ID"].map(pe["Region"])
    rfm["Customer_Type"] = rfm["Customer_ID"].map(pe["Customer_Type"])
    rfm["Log_Frequency"] = np.log(rfm["Frequency"] + 1)
    rfm["Log_Monetary"] = np.log(rfm["Monetary"] + 1)
    X = StandardScaler().fit_transform(rfm[["Recency", "Log_Frequency", "Log_Monetary"]])
    rfm["Z_Recency"], rfm["Z_Log_Frequency"], rfm["Z_Log_Monetary"] = X[:, 0], X[:, 1], X[:, 2]

    elbow = []
    for k in PILIHAN_K:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        elbow.append({"k": k, "Inertia / WCSS": km.inertia_, "Silhouette": silhouette_score(X, km.labels_)})
    elbow = pd.DataFrame(elbow)
    elbow["Turun (%)"] = -elbow["Inertia / WCSS"].pct_change() * 100

    km = KMeans(n_clusters=k_pilih, n_init=10, random_state=SEED, max_iter=300).fit(X)
    rfm["Cluster_Asli"] = km.labels_
    pusat = rfm.groupby("Cluster_Asli")[["Z_Recency", "Z_Log_Frequency", "Z_Log_Monetary"]].mean()
    label = beri_label(pusat)
    rfm["Segmen_Bisnis"] = rfm["Cluster_Asli"].map(label)
    urutan = sorted(set(label.values()), key=lambda x: (URUTAN_LABEL.index(dasar(x)), x))
    rfm["Cluster"] = rfm["Segmen_Bisnis"].map({sg: i for i, sg in enumerate(urutan)})
    return rfm, elbow, km.n_iter_, urutan, ref_date


@st.cache_data(show_spinner="Menjalankan Apriori...")
def jalankan_apriori(d, sup, conf, lift):
    bm = (d.assign(v=True).pivot_table(index="Transaction_ID", columns="Product_Name", values="v",
                                        aggfunc="max", fill_value=False).astype(bool))
    if len(bm) < 20:
        return len(bm), pd.DataFrame(), pd.DataFrame()
    fi = apriori(bm, min_support=sup, use_colnames=True, max_len=3)
    if fi.empty:
        return len(bm), fi, pd.DataFrame()
    fi["Itemset"] = fi["itemsets"].map(lambda s: " + ".join(sorted(s)))
    fi["Jumlah_Item"] = fi["itemsets"].map(len)
    if (fi["Jumlah_Item"] >= 2).sum() == 0:
        return len(bm), fi, pd.DataFrame()
    r = association_rules(fi[["support", "itemsets"]], metric="confidence", min_threshold=conf)
    r = r[r["lift"] >= lift].copy()
    r["Antecedent"] = r["antecedents"].map(lambda s: " + ".join(sorted(s)))
    r["Consequent"] = r["consequents"].map(lambda s: " + ".join(sorted(s)))
    r["Aturan"] = r["Antecedent"] + "  →  " + r["Consequent"]
    r = r.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    return len(bm), fi, r


def grafik_network(rules, kategori_map, n=20):
    """Network graph aturan 1 → 1 (layout melingkar; tebal garis = lift)."""
    p = rules[(rules["antecedents"].map(len) == 1) & (rules["consequents"].map(len) == 1)].head(n)
    if p.empty:
        return None
    nodes = sorted(set(p["Antecedent"]) | set(p["Consequent"]))
    ang = {nd: 2 * np.pi * i / len(nodes) for i, nd in enumerate(nodes)}
    pos = {nd: (np.cos(a), np.sin(a)) for nd, a in ang.items()}
    fig = go.Figure()
    for _, r in p.iterrows():
        (x0, y0), (x1, y1) = pos[r["Antecedent"]], pos[r["Consequent"]]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", showlegend=False, hoverinfo="text",
                                 text=f"{r['Aturan']}<br>lift {r['lift']:.2f} · confidence {r['confidence']:.1%}",
                                 line=dict(width=1 + 2.2 * r["lift"], color="rgba(37,99,235,.35)")))
    kats = sorted(set(kategori_map.values()))
    warna = ["#2563EB", "#14B8A6", "#F59E0B", "#E8590C", "#7C3AED", "#DB2777", "#0891B2", "#65A30D", "#64748B"]
    fig.add_trace(go.Scatter(x=[pos[nd][0] for nd in nodes], y=[pos[nd][1] for nd in nodes], mode="markers+text",
                             text=nodes, textposition="top center", showlegend=False,
                             hovertext=[f"{nd} ({kategori_map.get(nd, '')})" for nd in nodes], hoverinfo="text",
                             marker=dict(size=20, line=dict(width=2, color="white"),
                                         color=[warna[kats.index(kategori_map.get(nd, kats[0])) % len(warna)] for nd in nodes])))
    fig.update_xaxes(visible=False, range=[-1.5, 1.5]); fig.update_yaxes(visible=False, range=[-1.4, 1.4])
    fig.update_layout(title="Network Graph Kombinasi Produk (tebal garis = lift)")
    return fig


# ======================================================================
# EKSEKUSI
# ======================================================================
raw, produk, pel = muat(st.session_state["data_bytes"])
bersih, dq0, dq1, log = bersihkan(raw, produk, pel)
K = int(st.session_state.get("k_pilih", K_REKOMENDASI))
MIN_SUP = float(st.session_state.get("min_sup", 0.01))
MIN_CONF = float(st.session_state.get("min_conf", 0.20))
MIN_LIFT = float(st.session_state.get("min_lift", 1.2))
rfm, elbow, n_iter, URUTAN, TGL_ACUAN = analisis(bersih, K)
WARNA = {sg: WARNA_DASAR[dasar(sg)] for sg in URUTAN}
KAT_MAP = produk.set_index("Product_Name")["Category"].to_dict()
bersih_seg = bersih.merge(rfm[["Customer_ID", "Segmen_Bisnis"]], on="Customer_ID", how="left")
bt = tabel_basket(bersih)
DMIN, DMAX = bersih["Transaction_Date"].min().date(), bersih["Transaction_Date"].max().date()

profil = (rfm.groupby(["Cluster", "Segmen_Bisnis"])
          .agg(Jumlah_Pelanggan=("Customer_ID", "count"), Avg_Recency=("Recency", "mean"),
               Avg_Frequency=("Frequency", "mean"), Avg_Monetary=("Monetary", "mean"),
               Avg_Log_Frequency=("Log_Frequency", "mean"), Avg_Log_Monetary=("Log_Monetary", "mean"),
               Total_Revenue=("Monetary", "sum")).reset_index())
profil["Persen_Pelanggan"] = profil["Jumlah_Pelanggan"] / profil["Jumlah_Pelanggan"].sum() * 100
profil["Persen_Revenue"] = profil["Total_Revenue"] / profil["Total_Revenue"].sum() * 100
P = profil.set_index("Segmen_Bisnis")
KARAKTER = {
    sg: (f"Recency rata-rata {P.at[sg,'Avg_Recency']:.0f} hari, Frequency rata-rata {kom(P.at[sg,'Avg_Frequency'])} transaksi, "
         f"Monetary rata-rata {rp(P.at[sg,'Avg_Monetary'])}; {P.at[sg,'Jumlah_Pelanggan']:.0f} pelanggan "
         f"({kom(P.at[sg,'Persen_Pelanggan'])}%) menyumbang {kom(P.at[sg,'Persen_Revenue'])}% revenue.")
    for sg in URUTAN
}
N_APR, FI, RULES = jalankan_apriori(bersih, MIN_SUP, MIN_CONF, MIN_LIFT)


def tabel_dq():
    out = pd.DataFrame({"Pemeriksaan": list(dq0.keys()),
                        "Sebelum cleaning": list(dq0.values()),
                        "Sesudah cleaning": [dq1[k] for k in dq0.keys()]})
    return out.astype(str)


def buat_excel():
    buf = BytesIO()
    interp = profil.copy()
    interp["Karakteristik"] = interp["Segmen_Bisnis"].map(KARAKTER)
    interp["Strategi"] = interp["Segmen_Bisnis"].map(lambda sg: STRATEGI[dasar(sg)][0] + ": " + "; ".join(STRATEGI[dasar(sg)][1]))
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        tabel_dq().to_excel(xw, sheet_name="Data Quality", index=False)
        pd.DataFrame(log, columns=["Masalah", "Aturan / Keputusan", "Baris Terdampak"]).to_excel(xw, sheet_name="Log Cleaning", index=False)
        bersih.drop(columns="Month").to_excel(xw, sheet_name="Data Bersih", index=False)
        bt.to_excel(xw, sheet_name="Basket Size", index=False)
        rfm.drop(columns=["Cluster_Asli"]).to_excel(xw, sheet_name="RFM", index=False)
        elbow.to_excel(xw, sheet_name="Elbow", index=False)
        interp.to_excel(xw, sheet_name="Interpretasi Cluster", index=False)
        if len(RULES):
            RULES[["Aturan", "support", "confidence", "lift"]].to_excel(xw, sheet_name="Aturan Apriori", index=False)
    return buf.getvalue()



# ======================================================================
# HALAMAN
# ======================================================================
# ---------------------------------------------------------------- UNGGAH DATA (bagian 2: ringkasan proses)
if halaman == MENU[0]:
    st.subheader("Ringkasan Proses")
    n_aturan_terapan = sum(1 for _, _, n in log if n > 0)
    langkah = [
        ("1", "Pembacaan data", f"{angka(len(raw))} baris dan {raw.shape[1]} kolom dibaca dari Data_Raw; {raw['Transaction_ID'].nunique():,} transaksi, "
              f"{raw['Customer_ID'].nunique():,} pelanggan, {raw['Product_ID'].nunique()} produk.".replace(",", "."), "#2563EB"),
        ("2", "Pembersihan data", f"{n_aturan_terapan} aturan diterapkan; {angka(len(raw))} baris menjadi {angka(len(bersih))} baris bersih "
              f"({angka(len(raw) - len(bersih))} baris dihapus atau digabung).", "#14B8A6"),
        ("3", "Fitur: basket_size dan RFM", f"{angka(len(bt))} transaksi dengan rata-rata basket_size {kom(bt['basket_size'].mean(), 2)}; "
              f"{angka(len(rfm))} pelanggan dengan RFM (tanggal acuan {TGL_ACUAN:%d-%m-%Y}).", "#F59E0B"),
        ("4", "Segmentasi K-Means", f"k = {K}; segmen terbentuk: {', '.join(URUTAN)}.", "#E8590C"),
        ("5", "Apriori", f"{angka(len(FI))} frequent itemsets dan {angka(len(RULES))} aturan asosiasi"
              + (f"; terkuat: {RULES.iloc[0]['Aturan']} (lift {kom(RULES.iloc[0]['lift'], 2)})." if len(RULES) else "."), "#7C3AED"),
    ]
    for no, jd, isi, wr in langkah:
        st.markdown(f"<div class='kartu' style='--c:{wr};margin-bottom:10px'><h4>Langkah {no} — {jd} ✅</h4><small>{isi}</small></div>",
                    unsafe_allow_html=True)
    st.subheader("Langkah Selanjutnya")
    st.dataframe(tampil(pd.DataFrame({
        "Menu": ["📊 Dashboard BI", "🔍 EDA", "🧹 Data Cleaning", "🧮 RFM & Basket", "🛒 Apriori", "🎯 Elbow & K-Means",
                 "💡 Strategi & Rekomendasi", "📝 Laporan"],
        "Isi": ["KPI, tren, produk teratas, ringkasan Apriori dan K-Means, dengan filter",
                "Business understanding, statistik data mentah, audit, temuan awal",
                "Laporan data quality, aturan pembersihan, data bersih, unduh Excel",
                "basket_size, matriks transaksi, tabel RFM, hasil scaling",
                "Frequent itemsets, 10 aturan terbaik, network graph, interpretasi",
                "Elbow, Silhouette, hasil klaster, scatter 2D dan 3D",
                "Strategi per segmen, estimasi dampak, prioritas, KPI",
                "Laporan lengkap dengan angka otomatis"]})))

# ---------------------------------------------------------------- DASHBOARD BI
if halaman == MENU[1]:
    st.subheader("Alur Analisis")
    st.info("Data Cleaning → EDA → RFM & basket_size → Apriori (Market Basket) → Elbow & K-Means "
            f"(k = {K}) → Segmentasi ({' / '.join(URUTAN)}) → Dashboard BI dan Rekomendasi")
    f1, f2, f3 = st.columns([1.2, 1.4, 1.4])
    rentang = f1.date_input("Periode transaksi", (DMIN, DMAX), min_value=DMIN, max_value=DMAX)
    kategori = sorted(bersih["Category"].unique())
    pil_kat = f2.multiselect("Kategori produk", kategori, default=kategori)
    pil_seg = f3.multiselect("Segmen pelanggan", URUTAN, default=URUTAN, key=f"filter_segmen_{K}")
    if len(rentang) != 2:
        st.warning("Pilih tanggal awal dan tanggal akhir."); st.stop()
    b = bersih_seg[bersih_seg["Transaction_Date"].dt.date.between(rentang[0], rentang[1])
                   & bersih_seg["Category"].isin(pil_kat) & bersih_seg["Segmen_Bisnis"].isin(pil_seg)]
    if b.empty:
        st.warning("Tidak ada data pada kombinasi filter ini.")
    else:
        bsz = tabel_basket(b)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Penjualan", rp(b["Line_Total"].sum()))
        k2.metric("Jumlah Transaksi", angka(b["Transaction_ID"].nunique()))
        k3.metric("Pelanggan Unik", angka(b["Customer_ID"].nunique()))
        k4.metric("Rata-rata basket_size", kom(bsz["basket_size"].mean(), 2))
        k5.metric("Rata-rata Nilai Transaksi", rp(bsz["Total_Belanja"].mean()))

        st.markdown("#### Analisis Penjualan")
        tren = b.groupby("Month").agg(Transaksi=("Transaction_ID", "nunique"), Revenue=("Line_Total", "sum")).reset_index()
        ft = make_subplots(specs=[[{"secondary_y": True}]])
        ft.add_trace(go.Bar(x=tren["Month"], y=tren["Revenue"] / 1e6, name="Penjualan (Rp juta)", marker_color="#2563EB", opacity=.85), secondary_y=False)
        ft.add_trace(go.Scatter(x=tren["Month"], y=tren["Transaksi"], name="Jumlah Transaksi", mode="lines+markers",
                                line=dict(color="#E8590C", width=3)), secondary_y=True)
        ft.update_layout(title="Tren Penjualan Bulanan")
        ft.update_yaxes(title_text="Penjualan (Rp juta)", tickformat=".0f", secondary_y=False)
        ft.update_yaxes(title_text="Jumlah transaksi", rangemode="tozero", secondary_y=True)
        ft.update_xaxes(tickformat="%b %Y", dtick="M1")
        st.plotly_chart(rapikan(ft, 380))
        g1, g2 = st.columns([3, 2])
        with g1:
            top = (b.groupby(["Product_Name", "Category"])["Line_Total"].sum().nlargest(10).reset_index().sort_values("Line_Total"))
            top["Juta"] = top["Line_Total"] / 1e6
            fb = px.bar(top, x="Juta", y="Product_Name", color="Category", orientation="h", title="10 Produk dengan Penjualan Tertinggi",
                        labels={"Juta": "Penjualan (Rp juta)", "Product_Name": ""})
            st.plotly_chart(rapikan(fb, 400))
        with g2:
            kt = b.groupby("Category")["Line_Total"].sum().reset_index()
            fd = px.pie(kt, names="Category", values="Line_Total", hole=0.55, title="Distribusi Penjualan per Kategori")
            fd.update_traces(textinfo="percent+label", textposition="outside")
            st.plotly_chart(rapikan(fd, 400))

        st.markdown("#### Analisis Apriori")
        nb, fi_b, rl_b = jalankan_apriori(b, MIN_SUP, MIN_CONF, MIN_LIFT)
        if rl_b.empty:
            st.warning("Tidak ada aturan pada filter dan parameter ini. Turunkan minimum support / confidence / lift pada panel kiri.")
        else:
            a1, a2 = st.columns([1.1, 1])
            with a1:
                t = rl_b.head(10)[["Aturan", "support", "confidence", "lift"]].copy()
                t["support"] = (t["support"] * 100).map(lambda v: kom(v, 2) + "%")
                t["confidence"] = (t["confidence"] * 100).map(lambda v: kom(v, 1) + "%")
                t["lift"] = t["lift"].map(lambda v: kom(v, 2))
                t.columns = ["Aturan", "Support", "Confidence", "Lift"]
                st.markdown(f"**Aturan asosiasi utama** ({nb:,} transaksi; urut Lift lalu Confidence)".replace(",", "."))
                st.dataframe(tampil(t))
            with a2:
                fg = grafik_network(rl_b, KAT_MAP)
                if fg is not None:
                    st.plotly_chart(rapikan(fg, 400))

        st.markdown("#### Analisis K-Means")
        r = rfm[rfm["Segmen_Bisnis"].isin(pil_seg) & rfm["Customer_ID"].isin(b["Customer_ID"])]
        c1, c2 = st.columns(2)
        with c1:
            komp = r["Segmen_Bisnis"].value_counts().reindex(URUTAN).dropna().reset_index()
            komp.columns = ["Segmen", "Pelanggan"]
            fk = px.bar(komp, x="Segmen", y="Pelanggan", color="Segmen", color_discrete_map=WARNA, text="Pelanggan",
                        category_orders={"Segmen": URUTAN}, title="Jumlah Pelanggan per Segmen")
            fk.update_layout(showlegend=False)
            st.plotly_chart(rapikan(fk, 380))
        with c2:
            fs = px.scatter(r, x="Frequency", y="Monetary", color="Segmen_Bisnis", color_discrete_map=WARNA,
                            category_orders={"Segmen_Bisnis": URUTAN}, log_y=True, opacity=.7,
                            hover_data=["Customer_ID", "Recency"], title="Segmentasi Pelanggan: Frequency vs Monetary")
            fs.update_layout(yaxis_title="Monetary (Rp, skala log)")
            st.plotly_chart(rapikan(fs, 380))
        pt = profil[profil["Segmen_Bisnis"].isin(pil_seg)][["Segmen_Bisnis", "Jumlah_Pelanggan", "Avg_Recency", "Avg_Frequency",
                                                              "Avg_Monetary", "Persen_Revenue"]]
        st.dataframe(tampil(pt, rupiah=["Avg_Monetary"], bulat=["Jumlah_Pelanggan", "Avg_Recency"],
                            desimal=["Avg_Frequency", "Persen_Revenue"]))

        st.markdown("#### Bonus: Produk yang Relevan per Segmen (Apriori × K-Means)")
        tx = b.groupby("Transaction_ID")["Product_Name"].apply(set)
        sg_tx = b.groupby("Transaction_ID")["Segmen_Bisnis"].first()
        prods = sorted(b["Product_Name"].unique())
        total_p = pd.Series({p: tx.map(lambda s: p in s).mean() for p in prods})
        idx = {}
        for sg in [s for s in URUTAN if s in pil_seg]:
            sub = tx.loc[sg_tx[sg_tx == sg].index]
            if len(sub):
                idx[sg] = pd.Series({p: sub.map(lambda s: p in s).mean() for p in prods}) / total_p
        if idx:
            hm = pd.DataFrame(idx)
            pilih_p = hm.std(axis=1).nlargest(15).index
            fh = px.imshow(hm.loc[pilih_p].T, color_continuous_scale="RdYlGn", color_continuous_midpoint=1, aspect="auto",
                           text_auto=".2f", title="Indeks Penetrasi Produk per Segmen (1,00 = rata-rata; > 1 lebih sering dibeli)")
            st.plotly_chart(rapikan(fh, 380))
            sel = st.selectbox("Aturan asosiasi untuk segmen", list(idx.keys()))
            sub_b = b[b["Segmen_Bisnis"] == sel]
            if sub_b["Transaction_ID"].nunique() < 50:
                st.info("Transaksi pada segmen ini terlalu sedikit untuk Apriori.")
            else:
                n2, _, r2 = jalankan_apriori(sub_b, max(MIN_SUP, 0.01), MIN_CONF, MIN_LIFT)
                if r2.empty:
                    st.warning("Tidak ada aturan pada segmen ini dengan parameter saat ini.")
                else:
                    t2 = r2.head(10)[["Aturan", "support", "confidence", "lift"]].copy()
                    t2["support"] = (t2["support"] * 100).map(lambda v: kom(v, 2) + "%")
                    t2["confidence"] = (t2["confidence"] * 100).map(lambda v: kom(v, 1) + "%")
                    t2["lift"] = t2["lift"].map(lambda v: kom(v, 2))
                    t2.columns = ["Aturan", "Support", "Confidence", "Lift"]
                    st.dataframe(tampil(t2))
            st.caption("Segmen kecil memiliki sedikit transaksi sehingga aturannya kurang stabil; gunakan sebagai hipotesis lalu uji (A/B test).")

# ---------------------------------------------------------------- EDA
if halaman == MENU[2]:
    st.subheader("Tahap 1 — Business Understanding")
    st.markdown("""**Permasalahan utama.** Data transaksi toko hanya dipakai untuk laporan penjualan rutin. Manajemen belum mengetahui
(1) kombinasi produk yang dibeli bersamaan, (2) kelompok pelanggan bernilai dan yang berisiko berhenti berbelanja, serta
(3) strategi pemasaran yang tepat sasaran. Di sisi lain, data mentah masih mengandung nilai kosong, duplikat, teks tidak konsisten,
dan angka tidak wajar sehingga harus dibersihkan sebelum dianalisis.""")
    st.markdown("**Pertanyaan bisnis yang dijawab melalui analisis data:**")
    st.markdown("""1. Produk apa yang paling sering dibeli bersamaan dan bagaimana menatanya di rak atau menawarkannya sebagai bundling? *(Apriori)*
2. Segmen pelanggan mana yang paling bernilai dan segmen mana yang berisiko tidak kembali? *(RFM dan K-Means)*
3. Kapan penjualan tertinggi dan terendah, serta kategori mana yang paling dominan? *(EDA dan dashboard)*
4. Tindakan apa yang paling layak diprioritaskan untuk menaikkan penjualan dan mempertahankan pelanggan? *(Rekomendasi)*""")
    st.info("Hasil analisis membantu manajemen menaikkan penjualan lewat cross-selling dan bundling berdasarkan pola pembelian, "
            "serta mempertahankan pelanggan lewat program retensi yang dibedakan per segmen sehingga anggaran tidak terbuang pada kampanye massal.")

    st.subheader("Tahap 1 — Eksplorasi Data Mentah (Data_Raw)")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Baris × kolom", f"{angka(raw.shape[0])} × {raw.shape[1]}")
    e2.metric("Transaksi unik", angka(raw["Transaction_ID"].nunique()))
    e3.metric("Pelanggan unik", angka(raw["Customer_ID"].nunique()))
    e4.metric("Produk unik", angka(raw["Product_ID"].nunique()))
    e5, e6, e7, e8 = st.columns(4)
    e5.metric("Rentang waktu", f"{raw['Transaction_Date'].min():%d-%m-%Y} s.d. {raw['Transaction_Date'].max():%d-%m-%Y}")
    e6.metric("Total penjualan sementara", rp(raw["Line_Total"].sum()))
    e7.metric("Rata-rata Quantity (mentah)", kom(raw["Quantity"].mean(), 2))
    e8.metric("Rata-rata baris per transaksi", kom(len(raw) / raw["Transaction_ID"].nunique(), 2))

    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Tipe data**")
        st.dataframe(tampil(pd.DataFrame({"Kolom": raw.columns, "Tipe data": raw.dtypes.astype(str).values,
                                          "Missing": raw.isna().sum().values})))
    with cb:
        st.markdown("**Statistik deskriptif kolom numerik**")
        ds = raw[["Quantity", "Unit_Price", "Discount_Rate", "Line_Total"]].describe().T.reset_index()
        ds.columns = ["Kolom", "Jumlah", "Rata-rata", "Std", "Min", "Q1", "Median", "Q3", "Maks"]
        st.dataframe(tampil(ds, desimal=["Rata-rata", "Std", "Min", "Q1", "Median", "Q3", "Maks"]).assign(
            Jumlah=lambda x: x["Jumlah"].map(lambda v: angka(v))))

    st.markdown("**Audit kualitas data**")
    ad = pd.DataFrame({"Masalah": list(dq0.keys()), "Jumlah baris": list(dq0.values())})
    st.dataframe(tampil(ad, bulat=["Jumlah baris"]))

    st.subheader("Visualisasi EDA Awal")
    v1, v2 = st.columns(2)
    with v1:
        rt = raw.assign(Bulan=raw["Transaction_Date"].dt.to_period("M").dt.to_timestamp()).groupby("Bulan")["Line_Total"].sum().reset_index()
        rt["Juta"] = rt["Line_Total"] / 1e6
        f1 = px.line(rt, x="Bulan", y="Juta", markers=True, title="Tren Penjualan Bulanan (data mentah)")
        f1.update_traces(line=dict(color="#2563EB", width=3)); f1.update_layout(yaxis_title="Rp juta")
        f1.update_xaxes(tickformat="%b %Y", dtick="M1")
        st.plotly_chart(rapikan(f1, 380))
    with v2:
        tp = raw.groupby("Product_Name")["Line_Total"].sum().nlargest(10).reset_index().sort_values("Line_Total")
        f2 = px.bar(tp, x="Line_Total", y="Product_Name", orientation="h", title="10 Produk Penjualan Tertinggi (data mentah)",
                    color_discrete_sequence=["#14B8A6"], labels={"Line_Total": "Penjualan (Rp)", "Product_Name": ""})
        st.plotly_chart(rapikan(f2, 380))
    v3, v4 = st.columns(2)
    with v3:
        f3 = px.histogram(raw[raw["Quantity"].between(-5, 25)], x="Quantity", nbins=31, title="Distribusi Quantity (outlier 999 dipotong dari grafik)",
                          color_discrete_sequence=["#E8590C"])
        st.plotly_chart(rapikan(f3, 340))
    with v4:
        f4 = px.box(raw[raw["Unit_Price"] < 200000], x="Category", y="Unit_Price", title="Sebaran Harga Satuan per Kategori (harga 999.999 dikeluarkan)",
                    color_discrete_sequence=["#2563EB"])
        st.plotly_chart(rapikan(f4, 340))

    st.subheader("Temuan Awal")
    kat_dom = raw.assign(Category=raw["Category"].str.strip().str.title()).groupby("Category")["Line_Total"].sum().sort_values(ascending=False)
    bul = rt.loc[rt["Line_Total"].idxmax()]
    st.markdown(f"""1. **Kualitas data rendah.** Terdapat {angka(dq0['Baris duplikat identik'])} baris duplikat, {angka(dq0['Kategori tidak konsisten (kapital/spasi)'])} penulisan kategori tidak seragam, serta nilai tidak wajar (Quantity 999, harga 999.999 dan negatif) yang akan menyimpangkan KPI bila tidak dibersihkan.
2. **Satu transaksi terdiri dari beberapa baris produk** (rata-rata {kom(len(raw)/raw['Transaction_ID'].nunique(), 2)} baris per transaksi), sehingga transaksi harus dihitung dari Transaction_ID unik, bukan dari jumlah baris.
3. **Kategori terbesar adalah {kat_dom.index[0]}** ({kom(kat_dom.iloc[0]/kat_dom.sum()*100)}% penjualan) dan penjualan bulanan tertinggi terjadi pada {bul['Bulan']:%B %Y}.
4. Volume penjualan bulanan relatif stabil sepanjang 2025, sehingga peluang pertumbuhan lebih besar pada **nilai belanja per transaksi** (cross-selling) dan **retensi pelanggan** daripada pada musim.""")

# ---------------------------------------------------------------- DATA CLEANING
if halaman == MENU[3]:
    st.subheader("Laporan Data Quality: Sebelum dan Sesudah Cleaning")
    st.dataframe(tampil(tabel_dq()))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baris mentah", angka(len(raw)))
    c2.metric("Baris bersih", angka(len(bersih)), f"-{angka(len(raw) - len(bersih))}")
    c3.metric("Transaksi unik", angka(bersih["Transaction_ID"].nunique()))
    c4.metric("Pelanggan bertransaksi", angka(bersih["Customer_ID"].nunique()))

    st.subheader("Perbandingan Dataset Sebelum dan Sesudah")
    cmp = pd.DataFrame({
        "Ukuran": ["Baris", "Transaksi unik", "Pelanggan unik", "Total penjualan (Rp)", "Rata-rata Quantity", "Quantity maksimum",
                   "Harga satuan maksimum (Rp)", "Harga satuan minimum (Rp)", "Total sel kosong"],
        "Sebelum": [raw.shape[0], raw["Transaction_ID"].nunique(), raw["Customer_ID"].nunique(), raw["Line_Total"].sum(),
                    raw["Quantity"].mean(), raw["Quantity"].max(), raw["Unit_Price"].max(), raw["Unit_Price"].min(), int(raw.isna().sum().sum())],
        "Sesudah": [bersih.shape[0], bersih["Transaction_ID"].nunique(), bersih["Customer_ID"].nunique(), bersih["Line_Total"].sum(),
                    bersih["Quantity"].mean(), bersih["Quantity"].max(), bersih["Unit_Price"].max(), bersih["Unit_Price"].min(),
                    int(bersih.isna().sum().sum())]})
    st.dataframe(tampil(cmp.assign(Sebelum=cmp["Sebelum"].map(lambda v: kom(v, 2) if isinstance(v, float) and v < 100 else angka(v)),
                                   Sesudah=cmp["Sesudah"].map(lambda v: kom(v, 2) if isinstance(v, float) and v < 100 else angka(v)))))
    fh = go.Figure()
    fh.add_histogram(x=raw["Quantity"].clip(upper=20), name="Sebelum (dipotong 20)", marker_color="#E8590C", opacity=.6, nbinsx=21)
    fh.add_histogram(x=bersih["Quantity"], name="Sesudah", marker_color="#2563EB", opacity=.7, nbinsx=21)
    fh.update_layout(barmode="overlay", title="Distribusi Quantity Sebelum dan Sesudah Preprocessing")
    st.plotly_chart(rapikan(fh, 340))

    st.subheader("Aturan Cleaning yang Digunakan")
    st.dataframe(tampil(pd.DataFrame(log, columns=["Masalah", "Aturan / Keputusan", "Baris Terdampak"]), bulat=["Baris Terdampak"]))
    st.subheader("Data Transaksi Bersih")
    st.dataframe(tampil(bersih.drop(columns="Month"), rupiah=["Unit_Price", "Line_Total"]))
    st.download_button("Unduh seluruh hasil (Excel)", buat_excel(), "hasil_uts_bi.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------------------------------------------------- RFM & BASKET
if halaman == MENU[4]:
    st.subheader("basket_size per Transaksi")
    st.caption("basket_size = jumlah produk unik pada satu Transaction_ID.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Rata-rata basket_size", kom(bt["basket_size"].mean(), 2))
    m2.metric("basket_size maksimum", int(bt["basket_size"].max()))
    m3.metric("Transaksi 1 produk", f"{kom((bt['basket_size'] == 1).mean() * 100)}%")
    dist = bt["basket_size"].value_counts().sort_index().reset_index(); dist.columns = ["basket_size", "Transaksi"]
    fb = px.bar(dist, x="basket_size", y="Transaksi", text="Transaksi", title="Distribusi basket_size", color_discrete_sequence=["#2563EB"])
    st.plotly_chart(rapikan(fb, 320))
    st.dataframe(tampil(bt.head(500), rupiah=["Total_Belanja"]))

    st.subheader("Matriks Transaksi × Produk untuk Apriori")
    st.caption("Nilai 1 bila produk dibeli pada transaksi tersebut, 0 bila tidak. Satu baris mewakili satu Transaction_ID. Contoh 15 transaksi pertama:")
    mat = (bersih.assign(v=1).pivot_table(index="Transaction_ID", columns="Product_Name", values="v", aggfunc="max", fill_value=0))
    st.dataframe(mat.head(15))
    st.caption(f"Ukuran matriks lengkap: {angka(mat.shape[0])} transaksi × {mat.shape[1]} produk.")

    st.subheader("Tabel RFM per Pelanggan")
    st.caption(f"Tanggal acuan = {TGL_ACUAN:%d-%m-%Y} (satu hari setelah transaksi terakhir pada data bersih). Recency = hari sejak "
               "transaksi terakhir; Frequency = jumlah transaksi unik; Monetary = total penjualan bersih.")
    st.dataframe(tampil(rfm[["Customer_ID", "Region", "Customer_Type", "Last_Transaction", "Recency", "Frequency", "Monetary"]], rupiah=["Monetary"]))
    st.subheader("Log Transformation dan Standardization")
    st.caption("Log_Frequency = LN(Frequency + 1); Log_Monetary = LN(Monetary + 1); lalu Z-score untuk Recency, Log_Frequency, Log_Monetary. "
               "Standardisasi dipilih karena K-Means memakai jarak Euclidean sehingga fitur berskala besar (rupiah) akan mendominasi tanpa penyamaan skala. "
               "ID pelanggan tidak dimasukkan sebagai fitur.")
    st.dataframe(tampil(rfm[["Customer_ID", "Log_Frequency", "Log_Monetary", "Z_Recency", "Z_Log_Frequency", "Z_Log_Monetary"]],
                        desimal=["Log_Frequency", "Log_Monetary", "Z_Recency", "Z_Log_Frequency", "Z_Log_Monetary"]))
    z = rfm[["Z_Recency", "Z_Log_Frequency", "Z_Log_Monetary"]].agg(["mean", "std"]).T
    z.columns = ["Rata-rata", "Standar Deviasi"]; z.index.name = "Fitur"
    st.markdown("**Pemeriksaan standardisasi (rata-rata ≈ 0, standar deviasi ≈ 1)**")
    st.dataframe(tampil(z.reset_index(), desimal=["Rata-rata", "Standar Deviasi"]))
    st.markdown("**Strategi untuk dataset berukuran besar**")
    st.markdown("""- **Agregasi:** ringkas data transaksi menjadi RFM per pelanggan dan basket per transaksi sebelum memodelkan (seperti pada halaman ini).
- **Sampling:** gunakan sampel acak atau terstratifikasi untuk eksplorasi dan pemilihan k, lalu jalankan model akhir pada data penuh.
- **Chunking:** baca berkas besar bertahap (`read_csv(chunksize=...)`) dan akumulasikan agregat; gunakan `MiniBatchKMeans` untuk klaster.
- **Apriori:** naikkan minimum support atau batasi `max_len`, atau pakai FP-Growth yang lebih hemat memori.""")

# ---------------------------------------------------------------- APRIORI
if halaman == MENU[5]:
    st.subheader("Market Basket Analysis (Algoritma Apriori)")
    st.markdown(f"Basket = himpunan produk unik pada satu Transaction_ID ({angka(N_APR)} transaksi, {bersih['Product_Name'].nunique()} produk). "
                f"Parameter: minimum support **{kom(MIN_SUP*100, 2)}%**, minimum confidence **{kom(MIN_CONF*100, 0)}%**, minimum lift **{kom(MIN_LIFT, 1)}** "
                "(dapat diubah pada panel kiri).")
    st.info("**Alasan pemilihan parameter.** Dengan 30 produk dan rata-rata 3 produk per transaksi, peluang sebuah pasangan produk muncul bersamaan kecil "
            "(sebagian besar pasangan < 1% transaksi). Minimum support 1% (sekitar 60 dari 6.000 transaksi) cukup rendah untuk menangkap pasangan yang bermakna "
            "namun cukup tinggi agar tidak menangkap kebetulan. Confidence 20% menunjukkan 1 dari 5 pembeli produk A juga membeli B, dan lift > 1 memastikan "
            "hubungan lebih kuat daripada kebetulan.")
    k1, k2, k3 = st.columns(3)
    k1.metric("Frequent itemsets", angka(len(FI)))
    k2.metric("Aturan asosiasi", angka(len(RULES)))
    k3.metric("Lift tertinggi", kom(RULES["lift"].max(), 2) if len(RULES) else "-")
    if RULES.empty:
        st.warning("Tidak ada aturan pada parameter ini. Turunkan minimum support / confidence / lift pada panel kiri.")
    else:
        st.markdown("**Frequent itemsets (urut support)**")
        fis = FI[FI["Jumlah_Item"] >= 2].sort_values("support", ascending=False)[["Itemset", "Jumlah_Item", "support"]].copy()
        fis["support"] = (fis["support"] * 100).map(lambda v: kom(v, 2) + "%"); fis.columns = ["Itemset", "Jumlah Item", "Support"]
        st.dataframe(tampil(fis.head(30)))
        st.markdown("**10 aturan terbaik** — kriteria: lift tertinggi (kekuatan asosiasi), lalu confidence tertinggi sebagai penentu bila lift sama.")
        t = RULES.head(10)[["Aturan", "support", "confidence", "lift"]].copy()
        t["support"] = (t["support"] * 100).map(lambda v: kom(v, 2) + "%")
        t["confidence"] = (t["confidence"] * 100).map(lambda v: kom(v, 1) + "%")
        t["lift"] = t["lift"].map(lambda v: kom(v, 2)); t.columns = ["Aturan", "Support", "Confidence", "Lift"]
        st.dataframe(tampil(t))
        v1, v2 = st.columns(2)
        with v1:
            fg = grafik_network(RULES, KAT_MAP)
            if fg is not None: st.plotly_chart(rapikan(fg, 440))
        with v2:
            pr = RULES[(RULES["antecedents"].map(len) == 1) & (RULES["consequents"].map(len) == 1)]
            mat = pr.pivot_table(index="Antecedent", columns="Consequent", values="lift")
            fh = px.imshow(mat, color_continuous_scale="YlGnBu", aspect="auto", text_auto=".2f", title="Heatmap Lift (Antecedent → Consequent)")
            st.plotly_chart(rapikan(fh, 440))
        fs = px.scatter(RULES, x="support", y="confidence", size="lift", color="lift", hover_name="Aturan",
                        color_continuous_scale="Tealgrn", title="Support vs Confidence (ukuran dan warna = lift)")
        fs.update_xaxes(tickformat=".1%"); fs.update_yaxes(tickformat=".0%")
        st.plotly_chart(rapikan(fs, 380))

        st.subheader("Interpretasi Pola Pembelian")
        top = RULES.iloc[0]
        st.markdown(f"""**Cara membaca aturan teratas — {top['Aturan']}**
- **Support {kom(top['support']*100, 2)}%:** {kom(top['support']*100, 2)}% dari seluruh transaksi memuat kedua produk.
- **Confidence {kom(top['confidence']*100, 1)}%:** dari transaksi yang memuat {top['Antecedent']}, {kom(top['confidence']*100, 1)}% juga memuat {top['Consequent']}.
- **Lift {kom(top['lift'], 2)}:** peluang dibeli bersamaan {kom(top['lift'], 2)} kali lebih besar daripada jika kedua produk dibeli saling bebas (lift > 1 berarti asosiasi positif).
- Catatan: aturan asosiasi menunjukkan korelasi, **bukan** sebab-akibat.""")
        pair_u = RULES[(RULES["antecedents"].map(len) == 1) & (RULES["consequents"].map(len) == 1)].copy()
        pair_u["kunci"] = pair_u.apply(lambda r: " ↔ ".join(sorted([r["Antecedent"], r["Consequent"]])), axis=1)
        pair_u = pair_u.drop_duplicates("kunci").head(3)
        st.markdown("**Rekomendasi cross-selling / bundling dan penempatan rak (berdasarkan aturan, bukan perkiraan)**")
        for i, (_, r) in enumerate(pair_u.iterrows(), 1):
            st.markdown(f"""<div class="kartu" style="--c:#2563EB;margin-bottom:10px"><h4>{i}. {r['kunci']}</h4>
<small>support {kom(r['support']*100, 2)}% · confidence {kom(r['confidence']*100, 1)}% · lift {kom(r['lift'], 2)}</small><ul style="padding-left:18px;margin-top:6px">
<li><b>Bundling:</b> paket hemat {r['kunci'].replace(' ↔ ', ' + ')} dengan harga paket sedikit di bawah harga terpisah.</li>
<li><b>Penempatan rak:</b> letakkan kedua produk berdekatan atau pada satu etalase/endcap agar pembeli salah satunya melihat pasangannya.</li>
<li><b>Alasan:</b> lift {kom(r['lift'], 2)} menunjukkan keduanya dibeli bersamaan {kom(r['lift'], 1)}× lebih sering daripada kebetulan.</li></ul></div>""",
                        unsafe_allow_html=True)

# ---------------------------------------------------------------- ELBOW & K-MEANS
if halaman == MENU[6]:
    e_ = elbow.set_index("k")
    k_sil = int(e_["Silhouette"].idxmax())

    def turun(k):
        v = e_.at[k, "Turun (%)"]
        return "-" if pd.isna(v) else f"{kom(v, 2)}%"

    st.subheader("Fitur dan Pemodelan")
    st.markdown("Fitur K-Means: **Recency, Log_Frequency, Log_Monetary** (RFM per pelanggan; satu baris = satu pelanggan) yang distandardisasi dengan Z-score. "
                "Log dipakai karena Frequency dan Monetary condong ke kanan.")
    st.subheader("Elbow Method dan Silhouette (k = 2 sampai 6)")
    ca, cb = st.columns([1, 1.3])
    with ca:
        st.dataframe(tampil(elbow, desimal=["Inertia / WCSS", "Silhouette", "Turun (%)"]).assign(k=lambda x: x["k"].astype(int)))
    with cb:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=elbow["k"], y=elbow["Inertia / WCSS"], mode="lines+markers", name="Inertia",
                                 line=dict(color="#1F3A5F", width=3), marker=dict(size=9)), secondary_y=False)
        fig.add_trace(go.Bar(x=elbow["k"], y=elbow["Silhouette"], name="Silhouette", marker_color="rgba(20,184,166,.45)"), secondary_y=True)
        fig.add_vline(x=K, line_dash="dash", line_color="#E8590C", annotation_text=f"k dipilih = {K}")
        fig.update_layout(title="Grafik Elbow (garis) dan Silhouette (batang)", xaxis=dict(dtick=1))
        fig.update_yaxes(title_text="Inertia / WCSS", secondary_y=False); fig.update_yaxes(title_text="Silhouette", secondary_y=True)
        st.plotly_chart(rapikan(fig, 360))
    if K == K_REKOMENDASI:
        st.success(f"k yang digunakan = {K}. Silhouette tertinggi memang pada k = {k_sil} ({kom(e_.at[k_sil, 'Silhouette'], 4)}), tetapi hanya memisahkan "
                   f"pelanggan aktif dan tidak aktif sehingga kurang dapat ditindaklanjuti. Pada k = {K} silhouette {kom(e_.at[K, 'Silhouette'], 4)} "
                   f"(struktur klaster masih wajar) dan penurunan inertia melandai setelah k = {K} (k = {K}→{K+1}: {turun(K+1)}). Secara bisnis, "
                   f"k = {K} menghasilkan segmen yang berbeda strateginya: {', '.join(URUTAN)}.")
    else:
        st.warning(f"k yang dipilih = {K} (rekomendasi analisis k = {K_REKOMENDASI}). Pada k = {K}: inertia {kom(e_.at[K, 'Inertia / WCSS'], 2)}, "
                   f"penurunan {turun(K)}, silhouette {kom(e_.at[K, 'Silhouette'], 4)}. Label segmen dan strategi mengikuti nilai k.")

    st.subheader(f"Hasil K-Means (k = {K}, konvergen dalam {n_iter} iterasi)")
    st.dataframe(tampil(profil[["Cluster", "Segmen_Bisnis", "Jumlah_Pelanggan", "Avg_Recency", "Avg_Frequency", "Avg_Monetary",
                                "Avg_Log_Frequency", "Avg_Log_Monetary"]], rupiah=["Avg_Monetary"],
                        bulat=["Jumlah_Pelanggan", "Avg_Recency"], desimal=["Avg_Frequency", "Avg_Log_Frequency", "Avg_Log_Monetary"]))
    st.markdown("**Interpretasi dan label bisnis**")
    st.dataframe(tampil(pd.DataFrame({"Cluster": range(len(URUTAN)), "Label Bisnis": URUTAN,
                                      "Kriteria": [KRITERIA[dasar(sg)] for sg in URUTAN],
                                      "Karakteristik": [KARAKTER[sg] for sg in URUTAN],
                                      "Strategi": [STRATEGI[dasar(sg)][0] for sg in URUTAN]})))
    st.subheader("Visualisasi Cluster")
    v1, v2 = st.columns(2)
    with v1:
        sumbu = st.selectbox("Sumbu scatter", ["Recency vs Frequency", "Recency vs Monetary", "Frequency vs Monetary"])
        xa, ya = sumbu.split(" vs ")
        f1 = px.scatter(rfm, x=xa, y=ya, color="Segmen_Bisnis", color_discrete_map=WARNA, category_orders={"Segmen_Bisnis": URUTAN},
                        hover_data=["Customer_ID", "Recency", "Frequency", "Monetary"], opacity=.7, title=sumbu,
                        log_y=(ya == "Monetary"))
        st.plotly_chart(rapikan(f1, 420))
    with v2:
        f2 = px.scatter_3d(rfm, x="Z_Recency", y="Z_Log_Frequency", z="Z_Log_Monetary", color="Segmen_Bisnis", color_discrete_map=WARNA,
                           category_orders={"Segmen_Bisnis": URUTAN}, title="Ruang fitur terstandardisasi (3D)")
        f2.update_traces(marker=dict(size=3))
        st.plotly_chart(rapikan(f2, 460))

# ---------------------------------------------------------------- STRATEGI & REKOMENDASI
if halaman == MENU[7]:
    st.subheader(f"Strategi Marketing Berdasarkan Segmen (k = {K})")
    for mulai in range(0, len(URUTAN), 3):
        cols = st.columns(3)
        for col, sg in zip(cols, URUTAN[mulai:mulai + 3]):
            judul, aksi = STRATEGI[dasar(sg)]
            with col:
                st.markdown(
                    f"<div class='kartu' style='--c:{WARNA[sg]}'><h4>{sg}</h4><small>{judul}</small><hr style='margin:8px 0'>"
                    f"<b>{P.at[sg,'Jumlah_Pelanggan']:.0f} pelanggan · {kom(P.at[sg,'Persen_Revenue'])}% revenue</b><br>"
                    f"<small>{KARAKTER[sg]}</small><ul style='padding-left:18px;margin-top:8px'>"
                    + "".join(f"<li>{x}</li>" for x in aksi) + "</ul></div>", unsafe_allow_html=True)
        st.markdown("")
    # prioritas segmen: pendapatan besar tetapi recency tinggi
    risiko = [sg for sg in URUTAN if dasar(sg) in ("At Risk", "Churn Risk", "Lost")]
    prio = max(risiko, key=lambda s: P.at[s, "Total_Revenue"]) if risiko else None
    if prio:
        st.warning(f"**Segmen prioritas: {prio}.** Segmen ini sudah lama tidak bertransaksi (recency {P.at[prio,'Avg_Recency']:.0f} hari) namun "
                   f"masih menyumbang {kom(P.at[prio,'Persen_Revenue'])}% revenue dari {P.at[prio,'Jumlah_Pelanggan']:.0f} pelanggan, sehingga "
                   "kehilangan mereka berdampak paling besar di antara segmen berisiko.")

    st.subheader("Rekomendasi Keputusan Bisnis")
    st.markdown("Bedakan tiga jenis informasi: **FAKTA** = hasil hitungan data · **ESTIMASI** = proyeksi berbasis asumsi yang dapat diubah · "
                "**DAMPAK AKTUAL** = baru diketahui setelah implementasi dan uji (A/B test).")
    st.markdown("##### 1. Penempatan produk dan bundling (Apriori)")
    gain1 = 0
    if len(RULES):
        pair_u = RULES[(RULES["antecedents"].map(len) == 1) & (RULES["consequents"].map(len) == 1)].head(6)
        sets = bersih.groupby("Transaction_ID")["Product_Name"].apply(set)
        best = None
        for _, r in pair_u.iterrows():
            only_a = int(sets.map(lambda s: r["Antecedent"] in s and r["Consequent"] not in s).sum())
            avg_b = bersih.loc[bersih["Product_Name"] == r["Consequent"], "Line_Total"].mean()
            if best is None or only_a * avg_b > best[2]:
                best = (r, only_a, only_a * avg_b, avg_b)
        r, only_a, _, avg_b = best
        st.markdown(f"- **FAKTA:** {r['Aturan']} (support {kom(r['support']*100,2)}%, confidence {kom(r['confidence']*100,1)}%, lift {kom(r['lift'],2)}).")
        conv = st.slider("Asumsi: % pembeli yang ikut membeli produk pasangan setelah bundling / penempatan berdekatan", 1, 30, 10) / 100
        gain1 = only_a * conv * avg_b
        st.markdown(f"- **ESTIMASI:** {angka(only_a)} transaksi memuat {r['Antecedent']} tanpa {r['Consequent']}; rata-rata nilai {r['Consequent']} {rp(avg_b)}; "
                    f"konversi {kom(conv*100,0)}% → potensi tambahan penjualan **{rp(gain1)}** selama periode data (sebelum biaya).")
    st.markdown("##### 2. Retensi dan reaktivasi pelanggan berisiko (K-Means)")
    gain2 = 0
    if risiko:
        rr = rfm[rfm["Segmen_Bisnis"].isin(risiko)]
        aov = rr["Monetary"].sum() / rr["Frequency"].sum()
        st.markdown(f"- **FAKTA:** segmen berisiko ({', '.join(risiko)}) berjumlah **{angka(len(rr))} pelanggan** ({kom(len(rr)/len(rfm)*100)}%), "
                    f"menyumbang {kom(rr['Monetary'].sum()/rfm['Monetary'].sum()*100)}% revenue; rata-rata recency {rr['Recency'].mean():.0f} hari; "
                    f"nilai rata-rata per transaksi {rp(aov)}.")
        react = st.slider("Asumsi: % pelanggan berisiko yang kembali berbelanja satu kali tambahan", 1, 40, 10) / 100
        gain2 = len(rr) * react * aov
        st.markdown(f"- **ESTIMASI:** {kom(react*100,0)}% × {angka(len(rr))} pelanggan × {rp(aov)} = **{rp(gain2)}** (belum dikurangi biaya voucher).")
    st.markdown("##### 3. Program loyalitas untuk pelanggan bernilai tinggi")
    if "Loyal" in URUTAN:
        st.markdown(f"- **FAKTA:** segmen Loyal {P.at['Loyal','Jumlah_Pelanggan']:.0f} pelanggan ({kom(P.at['Loyal','Persen_Pelanggan'])}%) menyumbang "
                    f"{kom(P.at['Loyal','Persen_Revenue'])}% revenue. Program poin dan akses promo awal menjaga mereka agar tidak bergeser ke segmen berisiko.")
    st.markdown(f"**Total potensi (ESTIMASI, sebelum biaya): {rp(gain1 + gain2)}.** Dampak aktual baru dapat dinyatakan setelah kampanye diuji dengan kelompok kontrol.")

    st.markdown("##### Prioritas tindakan (skala 1–5; biaya 5 = paling murah)")
    pri = pd.DataFrame([["Penempatan rak dan bundling pasangan produk lift tertinggi", 4, 4, 5, 5],
                        ["Reaktivasi segmen berisiko dengan voucher personal", 5, 5, 3, 4],
                        ["Program poin untuk segmen Loyal", 3, 4, 3, 3],
                        ["Promosi pada bulan penjualan terendah", 3, 3, 4, 4]],
                       columns=["Rekomendasi", "Urgensi", "Potensi manfaat", "Biaya (murah = 5)", "Kemudahan"])
    pri["Skor"] = pri.iloc[:, 1:5].mean(axis=1)
    pri = pri.sort_values("Skor", ascending=False)
    pri["Skor"] = pri["Skor"].map(lambda v: kom(v, 2))
    st.dataframe(tampil(pri))
    st.caption("Skor merupakan penilaian analis (judgement), bukan hasil perhitungan data.")
    st.markdown("##### Rencana implementasi dan KPI evaluasi")
    st.dataframe(tampil(pd.DataFrame({
        "Horizon": ["Jangka pendek (0–1 bulan)", "Jangka menengah (1–3 bulan)", "Jangka menengah (3–6 bulan)"],
        "Tindakan": ["Pindahkan pasangan produk lift tinggi berdekatan; pasang paket bundling di kasir",
                     "Kampanye reaktivasi segmen berisiko dengan kelompok kontrol (A/B test)",
                     "Program poin untuk segmen Loyal; segmentasi ulang RFM setiap bulan"],
        "KPI evaluasi": ["Attach rate pasangan produk, basket_size, penjualan per paket",
                         "Reactivation rate, recency rata-rata, ROI voucher",
                         "Retention rate, frekuensi kunjungan, perpindahan antar-segmen"]})))

# ---------------------------------------------------------------- LAPORAN
if halaman == MENU[8]:
    esc = _html.escape
    st.markdown(
        """
<style>
.lap-sec{display:flex;align-items:center;gap:12px;margin:30px 0 12px 0;padding:12px 16px;border-radius:12px;
  background:linear-gradient(90deg,#1e3a8a,#2563eb);color:#fff}
.lap-sec .no{flex:none;width:34px;height:34px;border-radius:50%;background:#fff;color:#1e3a8a;font-weight:800;
  display:flex;align-items:center;justify-content:center}
.lap-sec .jd{font-size:1.08rem;font-weight:700;color:#fff}
.lap-sub{font-weight:700;margin:16px 0 6px 0;opacity:.9}
.lap-tbl{width:100%;border-collapse:separate;border-spacing:0;table-layout:fixed;font-size:.92rem;margin:6px 0 14px 0;
  border:1px solid rgba(128,128,128,.35);border-radius:10px;overflow:hidden}
.lap-tbl th{background:rgba(37,99,235,.16);text-align:left;padding:10px 12px;font-weight:700}
.lap-tbl td{padding:10px 12px;vertical-align:top;border-top:1px solid rgba(128,128,128,.25);line-height:1.55;overflow-wrap:anywhere}
.lap-tbl tr:nth-child(even) td{background:rgba(128,128,128,.06)}
.lap-tbl ul{margin:0;padding-left:18px}
.lap-badge{display:inline-block;padding:2px 10px;border-radius:999px;color:#fff;font-weight:700;font-size:.82rem}
.lap-q{border-left:5px solid #f59e0b;background:rgba(245,158,11,.10);border-radius:10px;padding:12px 16px;margin:10px 0 6px 0}
.lap-q .lbl{font-size:.72rem;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;color:#d97706}
.lap-q .tanya{font-weight:700;margin:2px 0 10px 0}
.lap-q .jawab{line-height:1.7;text-align:justify}
.lap-note{border-left:5px solid #2563eb;background:rgba(37,99,235,.09);border-radius:10px;padding:12px 16px;margin:8px 0 12px 0;line-height:1.65}
.lap-kes{display:flex;gap:12px;align-items:flex-start;padding:12px 16px;margin:8px 0;border-radius:10px;
  background:rgba(128,128,128,.10);border:1px solid rgba(128,128,128,.25);line-height:1.65}
.lap-kes .n{flex:none;width:28px;height:28px;border-radius:50%;background:#2563eb;color:#fff;font-weight:800;
  display:flex;align-items:center;justify-content:center;margin-top:1px}
</style>
""", unsafe_allow_html=True)

    def lap_judul(no, teks):
        st.markdown(f'<div class="lap-sec"><span class="no">{no}</span><span class="jd">{esc(teks)}</span></div>', unsafe_allow_html=True)

    def lap_sub(teks):
        st.markdown(f'<div class="lap-sub">{esc(teks)}</div>', unsafe_allow_html=True)

    def lap_tabel(header, baris, lebar, mentah=()):
        th = "".join(f'<th style="width:{w}%">{esc(h)}</th>' for h, w in zip(header, lebar))
        isi = "".join("<tr>" + "".join(f"<td>{c if i in mentah else esc(str(c))}</td>" for i, c in enumerate(row)) + "</tr>" for row in baris)
        st.markdown(f'<table class="lap-tbl"><thead><tr>{th}</tr></thead><tbody>{isi}</tbody></table>', unsafe_allow_html=True)

    def lap_tanya(tanya, jawab):
        st.markdown(f'<div class="lap-q"><div class="lbl">Pertanyaan</div><div class="tanya">{esc(tanya)}</div>'
                    f'<div class="lbl">Jawaban</div><div class="jawab">{esc(jawab)}</div></div>', unsafe_allow_html=True)

    def lap_catatan(teks):
        st.markdown(f'<div class="lap-note">{esc(teks)}</div>', unsafe_allow_html=True)

    def badge(sg):
        return f'<span class="lap-badge" style="background:{WARNA[sg]}">{esc(sg)}</span>'

    e = elbow.set_index("k")
    st.header("Laporan UTS: Integrasi Data Mining dan Business Intelligence pada Data Transaksi Toko")
    st.caption(f"{NAMA} · NPM {NPM} · Kelas {KELAS} · Universitas Timor")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baris mentah → bersih", f"{angka(len(raw))} → {angka(len(bersih))}")
    m2.metric("Pelanggan dianalisis", angka(len(rfm)))
    m3.metric("Jumlah cluster (k)", K)
    m4.metric("Aturan Apriori", angka(len(RULES)))

    lap_judul(1, "Business Understanding dan EDA")
    lap_tabel(["Aspek", "Uraian"], [
        ("Permasalahan utama", "Data transaksi hanya dipakai untuk laporan rutin; pola pembelian, segmen pelanggan, dan strategi pemasaran berbasis data belum dimanfaatkan, sementara data mentah masih kotor."),
        ("Pertanyaan bisnis", "(1) Produk apa yang dibeli bersamaan? (2) Segmen pelanggan mana yang paling bernilai dan berisiko? (3) Kapan penjualan tertinggi dan kategori apa yang dominan? (4) Tindakan apa yang diprioritaskan?"),
        ("Manfaat bagi manajemen", "Menaikkan penjualan lewat cross-selling dan bundling, serta mempertahankan pelanggan lewat retensi yang dibedakan per segmen."),
        ("Ukuran data", f"{angka(raw.shape[0])} baris × {raw.shape[1]} kolom; {raw['Transaction_ID'].nunique():,} transaksi; {raw['Customer_ID'].nunique():,} pelanggan; {raw['Product_ID'].nunique()} produk; "
                        f"{raw['Transaction_Date'].min():%d-%m-%Y} s.d. {raw['Transaction_Date'].max():%d-%m-%Y}.".replace(",", ".")),
    ], [24, 76])

    lap_judul(2, "Data Preprocessing")
    lap_tabel(["Pemeriksaan", "Sebelum", "Sesudah"], tabel_dq().values.tolist(), [60, 20, 20])
    lap_tabel(["Masalah", "Keputusan dan alasan", "Baris"], [(a, b_, angka(c)) for a, b_, c in log], [28, 62, 10])
    lap_catatan(f"Data mentah {angka(len(raw))} baris menjadi {angka(len(bersih))} baris bersih ({angka(len(raw) - len(bersih))} baris dihapus atau digabung). "
                f"Total penjualan berubah dari {rp(raw['Line_Total'].sum())} menjadi {rp(bersih['Line_Total'].sum())}.")
    lap_tabel(["Fitur", "Definisi"], [
        ("basket_size", f"Jumlah produk unik per Transaction_ID (rata-rata {kom(bt['basket_size'].mean(), 2)})."),
        ("Recency", f"Hari sejak transaksi terakhir terhadap tanggal acuan {TGL_ACUAN:%d-%m-%Y}."),
        ("Frequency", "Jumlah transaksi unik pelanggan."),
        ("Monetary", "Total penjualan bersih pelanggan."),
        ("Scaling", "log(x + 1) untuk Frequency dan Monetary lalu Z-score; K-Means memakai jarak Euclidean sehingga skala harus disamakan."),
    ], [24, 76])
    lap_tanya("Mengapa preprocessing harus dilakukan sebelum Apriori dan K-Means?",
              "Duplikat menggandakan basket dan Frequency, tanggal serta kuantitas yang salah merusak Recency dan Monetary, dan nilai ekstrem (999 unit, harga 999.999) menggeser pusat cluster. "
              "Tanpa pembersihan, aturan dan segmen mencerminkan kesalahan data, bukan perilaku pelanggan (garbage in, garbage out).")

    lap_judul(3, "Analisis Apriori")
    lap_catatan(f"Minimum support {kom(MIN_SUP*100, 2)}%, minimum confidence {kom(MIN_CONF*100, 0)}%, minimum lift {kom(MIN_LIFT, 1)}; {angka(len(FI))} frequent itemsets dan {angka(len(RULES))} aturan. "
                "Peringkat: lift tertinggi lalu confidence.")
    if len(RULES):
        lap_tabel(["No", "Aturan", "Support", "Confidence", "Lift"],
                  [(i, r["Aturan"], kom(r["support"] * 100, 2) + "%", kom(r["confidence"] * 100, 1) + "%", kom(r["lift"], 2))
                   for i, r in RULES.head(10).iterrows() for i in [i + 1]], [7, 53, 13, 14, 13])
    lap_tanya("Apa arti support, confidence, dan lift?",
              "Support adalah proporsi transaksi yang memuat kedua produk. Confidence adalah peluang produk B dibeli bila produk A dibeli. Lift membandingkan peluang tersebut dengan peluang kebetulan; "
              "lift di atas 1 berarti produk saling menguatkan. Aturan asosiasi menunjukkan korelasi, bukan sebab-akibat.")

    lap_judul(4, f"Segmentasi K-Means (k = {K})")
    lap_tabel(["k", "Inertia", "Silhouette", "Penurunan inertia"],
              [(int(k), kom(v["Inertia / WCSS"], 2), kom(v["Silhouette"], 4), "-" if pd.isna(v["Turun (%)"]) else kom(v["Turun (%)"], 2) + "%")
               for k, v in e.iterrows()], [15, 30, 27, 28])
    lap_catatan(f"Silhouette tertinggi pada k = {int(e['Silhouette'].idxmax())}, tetapi k = {K} dipilih karena struktur klaster masih wajar (silhouette {kom(e.at[K, 'Silhouette'], 4)}) "
                "dan menghasilkan segmen yang dapat dibedakan strateginya. Pemilihan k menggabungkan metrik dan interpretasi bisnis.")
    lap_tabel(["Cluster", "Label", "Pelanggan", "Recency", "Frequency", "Monetary", "Revenue"],
              [(int(rw["Cluster"]), badge(rw["Segmen_Bisnis"]), angka(rw["Jumlah_Pelanggan"]) + f" ({kom(rw['Persen_Pelanggan'])}%)",
                f"{rw['Avg_Recency']:.0f} hari", kom(rw["Avg_Frequency"]), rp(rw["Avg_Monetary"]), kom(rw["Persen_Revenue"]) + "%")
               for _, rw in profil.iterrows()], [8, 17, 17, 13, 13, 18, 14], mentah=(1,))
    lap_tabel(["Label", "Kriteria profil RFM", "Karakteristik hasil analisis"],
              [(badge(sg), KRITERIA[dasar(sg)], KARAKTER[sg]) for sg in URUTAN], [16, 30, 54], mentah=(0,))

    lap_judul(5, "Dashboard BI dan Strategi")
    lap_tabel(["Segmen", "Fokus strategi", "Tindakan yang disarankan"],
              [(badge(sg), STRATEGI[dasar(sg)][0], "<ul>" + "".join(f"<li>{esc(x)}</li>" for x in STRATEGI[dasar(sg)][1]) + "</ul>") for sg in URUTAN],
              [16, 24, 60], mentah=(0, 2))

    lap_judul(6, "Rekomendasi Keputusan Bisnis")
    lap_tabel(["Rekomendasi", "Bukti data (FAKTA)", "KPI keberhasilan"], [
        ("Penempatan rak dan bundling pasangan produk", RULES.iloc[0]["Aturan"] + f" (lift {kom(RULES.iloc[0]['lift'], 2)})" if len(RULES) else "-", "Attach rate pasangan, basket_size"),
        ("Reaktivasi segmen berisiko", f"{angka(int(P.loc[[s for s in URUTAN if dasar(s) in ('At Risk','Churn Risk','Lost')], 'Jumlah_Pelanggan'].sum()))} pelanggan berisiko", "Reactivation rate, ROI voucher"),
        ("Program loyalitas segmen Loyal", f"Loyal menyumbang {kom(P.at['Loyal','Persen_Revenue'])}% revenue" if "Loyal" in URUTAN else "-", "Retention rate, frekuensi kunjungan"),
    ], [30, 45, 25])
    lap_catatan("Angka estimasi dampak (lihat menu Strategi & Rekomendasi) berbasis asumsi yang dapat diubah; dampak aktual baru diketahui setelah uji A/B dengan kelompok kontrol.")

    lap_judul("✓", "Kesimpulan")
    atas = P["Persen_Revenue"].idxmax()
    poin = [
        f"Data mentah {angka(len(raw))} baris dibersihkan menjadi {angka(len(bersih))} baris valid dari {angka(bersih['Transaction_ID'].nunique())} transaksi.",
        (f"Apriori menemukan {angka(len(RULES))} aturan; pasangan terkuat {RULES.iloc[0]['Aturan']} (lift {kom(RULES.iloc[0]['lift'], 2)})." if len(RULES) else "Tidak ada aturan pada parameter saat ini."),
        f"K-Means (k = {K}) membentuk segmen {', '.join(URUTAN)}; segmen {atas} menyumbang revenue terbesar ({kom(P.at[atas, 'Persen_Revenue'])}%).",
        "Strategi sebaiknya dibedakan per segmen dan pola pembelian dimanfaatkan untuk bundling serta penataan rak, lalu diuji dengan A/B test.",
    ]
    for i, t in enumerate(poin, 1):
        st.markdown(f'<div class="lap-kes"><span class="n">{i}</span><span>{esc(t)}</span></div>', unsafe_allow_html=True)
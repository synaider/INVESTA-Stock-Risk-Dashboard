import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.interpolate import splrep, splev
from scipy import stats
from datetime import datetime, timedelta
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="INVESTA", page_icon="📊", layout="wide")

# ============================================================
# GLOBAL CSS — adaptive dark/light mode
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ---- Base font ---- */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ============================================================
   Gunakan CSS variables BAWAAN Streamlit — selalu tersedia
   dan otomatis berubah saat toggle dark/light:
   --background-color          : latar utama
   --secondary-background-color: latar kartu/sidebar
   --text-color                : warna teks utama
   --primary-color             : warna aksen
   ============================================================ */

/* ---- Alert boxes — menggunakan warna solid yang kontras di kedua mode ---- */
.box-success {
    padding: 0.75rem 1rem;
    background-color: color-mix(in srgb, #22c55e 12%, var(--background-color));
    border-left: 3px solid #22c55e;
    border-radius: 4px;
    margin: 0.75rem 0;
    color: var(--text-color);
    font-size: 0.9rem;
}
.box-warning {
    padding: 0.75rem 1rem;
    background-color: color-mix(in srgb, #eab308 12%, var(--background-color));
    border-left: 3px solid #eab308;
    border-radius: 4px;
    margin: 0.75rem 0;
    color: var(--text-color);
    font-size: 0.9rem;
}
.box-error {
    padding: 0.75rem 1rem;
    background-color: color-mix(in srgb, #ef4444 12%, var(--background-color));
    border-left: 3px solid #ef4444;
    border-radius: 4px;
    margin: 0.75rem 0;
    color: var(--text-color);
    font-size: 0.9rem;
}
.box-info {
    padding: 0.75rem 1rem;
    background-color: var(--secondary-background-color);
    border-left: 3px solid var(--primary-color);
    border-radius: 4px;
    margin: 0.75rem 0;
    color: var(--text-color);
    font-size: 0.9rem;
}

/* ---- Divider ---- */
.section-divider {
    border: none;
    border-top: 1px solid rgba(128,128,128,0.2);
    margin: 1.5rem 0;
}

/* ---- Main header ---- */
.main-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: var(--text-color);
    letter-spacing: 0.04em;
    margin-bottom: 0.2rem;
}
.sub-header {
    font-size: 0.95rem;
    color: var(--text-color);
    opacity: 0.6;
    margin-bottom: 2rem;
    letter-spacing: 0.02em;
}

/* ---- Feature cards (Tab 1) ---- */
.feature-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.feature-item {
    flex: 1;
    min-width: 160px;
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 6px;
    padding: 1rem;
}
.feature-step {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--primary-color);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.feature-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-color);
    margin-bottom: 0.3rem;
}
.feature-desc {
    font-size: 0.8rem;
    color: var(--text-color);
    opacity: 0.7;
    line-height: 1.5;
}

/* ---- Metric card ---- */
.metric-card {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 6px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-label {
    font-size: 0.78rem;
    color: var(--text-color);
    opacity: 0.6;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.35rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--text-color);
}
.metric-sub {
    font-size: 0.75rem;
    color: var(--text-color);
    opacity: 0.5;
    margin-top: 0.2rem;
}

/* ---- Tab styling ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid rgba(128,128,128,0.2);
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.85rem;
    color: var(--text-color);
    opacity: 0.6;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 0.6rem 1.2rem;
    transition: all 0.15s;
}
.stTabs [aria-selected="true"] {
    color: var(--text-color) !important;
    opacity: 1 !important;
    border-bottom: 2px solid var(--primary-color) !important;
    background: transparent !important;
}

/* ---- Streamlit default overrides ---- */
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
}

/* ---- Text classes — warna inherit dari Streamlit, bekerja di kedua mode ---- */
.text-body {
    font-size: 0.95rem;
    line-height: 1.7;
    color: var(--text-color);
    opacity: 0.85;
}
.text-body strong, .text-body b {
    color: var(--text-color);
    opacity: 1;
    font-weight: 600;
}
.text-caption {
    font-size: 0.82rem;
    color: var(--text-color);
    opacity: 0.7;
    line-height: 1.8;
}
.text-hint {
    font-size: 0.8rem;
    color: var(--text-color);
    opacity: 0.45;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 0.2rem;
}
.text-note {
    font-size: 0.88rem;
    color: var(--text-color);
    opacity: 0.75;
    margin-top: 0.5rem;
}
.text-disclaimer {
    font-size: 0.78rem;
    color: var(--text-color);
    opacity: 0.45;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# APP HEADER
# ============================================================
st.markdown('<p class="main-header">INVESTA</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Aplikasi Manajemen Risiko Investasi — '
    'Value at Risk dengan Variance-Covariance & Cornish-Fisher</p>',
    unsafe_allow_html=True
)

# ============================================================
# STOCK LISTS
# ============================================================
STOCK_LIST_IDN = {
    "ADRO": "Adaro Energy", "AKRA": "AKR Corporindo", "AMRT": "Sumber Alfaria Trijaya",
    "ANTM": "Aneka Tambang", "ASII": "Astra International", "BBCA": "Bank Central Asia",
    "BBNI": "Bank Negara Indonesia", "BBRI": "Bank Rakyat Indonesia", "BMRI": "Bank Mandiri",
    "BRPT": "Barito Pacific", "CPIN": "Charoen Pokphand Indonesia", "EXCL": "XL Axiata",
    "GOTO": "GoTo Gojek Tokopedia", "ICBP": "Indofood CBP", "INCO": "Vale Indonesia",
    "INDF": "Indofood Sukses Makmur", "INKP": "Indah Kiat Pulp & Paper", "ISAT": "Indosat Ooredoo",
    "ITMG": "Indo Tambangraya Megah", "JPFA": "Japfa Comfeed Indonesia", "KLBF": "Kalbe Farma",
    "MBMA": "Marga Abhinaya Abadi", "MDKA": "Merdeka Copper Gold", "MEDC": "Medco Energi",
    "PGAS": "Perusahaan Gas Negara", "PTBA": "Bukit Asam", "SMGR": "Semen Indonesia",
    "TLKM": "Telkom Indonesia", "UNTR": "United Tractors", "UNVR": "Unilever Indonesia"
}

STOCK_LIST_INTL = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "GOOGL": "Alphabet Inc. (Google)",
    "AMZN": "Amazon.com Inc.", "TSLA": "Tesla Inc.", "META": "Meta Platforms (Facebook)",
    "NVDA": "NVIDIA Corporation", "BRK-B": "Berkshire Hathaway", "JPM": "JPMorgan Chase",
    "JNJ": "Johnson & Johnson", "V": "Visa Inc.", "PG": "Procter & Gamble",
    "UNH": "UnitedHealth Group", "MA": "Mastercard", "HD": "Home Depot",
    "DIS": "Walt Disney Company", "BAC": "Bank of America", "NFLX": "Netflix Inc.",
    "ADBE": "Adobe Inc.", "CRM": "Salesforce Inc.", "CSCO": "Cisco Systems",
    "INTC": "Intel Corporation", "PFE": "Pfizer Inc.", "KO": "Coca-Cola Company",
    "PEP": "PepsiCo Inc.", "WMT": "Walmart Inc.", "NKE": "Nike Inc.",
    "MRK": "Merck & Co.", "T": "AT&T Inc.", "VZ": "Verizon Communications"
}

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Home",
    "Download Data Saham",
    "Upload & Uji Normalitas",
    "VaR — Variance-Covariance",
    "VaR — Cornish-Fisher",
    "Perbandingan & Backtesting",
])

# ============================================================
# TAB 1: HOME
# ============================================================
with tab1:

    st.markdown("#### Tentang Aplikasi")
    st.markdown(
        '<p class="text-body">'
        'INVESTA membantu Anda menghitung <strong>Value at Risk (VaR)</strong> '
        'dan <strong>Expected Shortfall (ES)</strong> dari data harga saham historis '
        'menggunakan dua pendekatan parametrik: '
        '<strong>Variance-Covariance</strong> dan <strong>Cornish-Fisher</strong>, '
        'dilengkapi dengan uji normalitas dan backtesting untuk validasi model.'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("#### Alur Kerja")
    st.markdown("""
    <div class="feature-row">
        <div class="feature-item">
            <div class="feature-step">Langkah 1</div>
            <div class="feature-title">Download / Upload Data</div>
            <div class="feature-desc">Ambil data historis dari Yahoo Finance atau upload file Excel sendiri.</div>
        </div>
        <div class="feature-item">
            <div class="feature-step">Langkah 2</div>
            <div class="feature-title">Uji Normalitas</div>
            <div class="feature-desc">Uji distribusi return dengan uji <strong>Lilliefors</strong> untuk menentukan metode VaR yang paling sesuai.</div>
        </div>
        <div class="feature-item">
            <div class="feature-step">Langkah 3</div>
            <div class="feature-title">Hitung VaR &amp; ES</div>
            <div class="feature-desc">Estimasi risiko dengan Variance-Covariance dan Cornish-Fisher, termasuk Expected Shortfall.</div>
        </div>
        <div class="feature-item">
            <div class="feature-step">Langkah 4</div>
            <div class="feature-title">Validasi Model</div>
            <div class="feature-desc">Bandingkan VaR &amp; ES kedua metode dan jalankan backtesting untuk evaluasi akurasi.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("#### Metode yang Digunakan")
    st.markdown("##### 1. Variance-Covariance (Normal VaR & ES)")
    st.markdown(
        '<p class="text-body">'
        'Mengasumsikan log return berdistribusi normal. VaR dan ES dihitung dari nilai posisi awal, '
        'standar deviasi historis, z-score tingkat kepercayaan, dan akar holding period. '
        'Rata-rata return diasumsikan relatif kecil sehingga diabaikan, '
        'dan rumus ditampilkan sebagai <strong>besar kerugian dalam satuan moneter</strong>.'
        '</p>',
        unsafe_allow_html=True
    )
    st.markdown('<p class="text-hint">— Rumus VaR —</p>', unsafe_allow_html=True)
    st.code("VaR(α) = V₀ × σ × Z(1−α) × √t", language=None)
    st.markdown('<p class="text-hint">— Rumus ES —</p>', unsafe_allow_html=True)
    st.code("ES(α) = V₀ × σ × φ(Z(1−α)) / α × √t", language=None)
    st.markdown(
        '<p class="text-caption">'
        'φ(Z<sub>1−α</sub>) = fungsi densitas normal standar pada titik Z<sub>1−α</sub><br>'
        'α = tingkat signifikansi (contoh: 0,05 untuk tingkat kepercayaan 95%)'
        '</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="text-note">'
        '✅ Mudah diinterpretasi &nbsp; ✅ Komputasi cepat<br>'
        '⚠️ Kurang akurat jika data tidak berdistribusi normal'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("##### 2. Cornish-Fisher (VaR & ES Adjusted)")
    st.markdown(
        '<p class="text-body">'
        'Menyesuaikan kuantil Z menggunakan skewness (S) dan excess kurtosis (K<sub>e</sub>) data aktual, '
        'sehingga lebih akurat untuk distribusi asimetris atau berekor tebal (<em>fat-tail</em>). '
        'Rata-rata return diasumsikan relatif kecil sehingga diabaikan, dan rumus ditampilkan sebagai '
        '<strong>besar kerugian dalam satuan moneter</strong>.'
        '</p>',
        unsafe_allow_html=True
    )
    st.markdown('<p class="text-hint">— Kuantil Cornish-Fisher (K_e = excess kurtosis = K − 3) —</p>', unsafe_allow_html=True)
    st.code(
        "Z_CF = Z(1−α) + (1/6)(Z²(1−α) − 1)·S\n"
        "             + (1/24)(Z³(1−α) − 3·Z(1−α))·K_e\n"
        "             − (1/36)(2·Z³(1−α) − 5·Z(1−α))·S²\n\n"
        "K_e  = excess kurtosis = K − 3",
        language=None
    )
    st.markdown('<p class="text-hint">— VaR &amp; ES Cornish-Fisher —</p>', unsafe_allow_html=True)
    st.code(
        "VaR_CF(α) = V₀ × σ × Z_CF × √t\n"
        "ES_CF(α)  = V₀ × σ × φ(Z_CF) / α × √t",
        language=None
    )
    st.markdown(
        '<p class="text-caption">'
        'φ(Z<sub>CF</sub>) = fungsi densitas normal standar pada titik Z<sub>CF</sub>'
        '</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="text-note">'
        '✅ Memperhitungkan skewness &amp; kurtosis &nbsp; ✅ Lebih akurat untuk return saham<br>'
        '⚠️ Kurang stabil untuk skewness yang sangat ekstrem'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("#### Notasi")
    st.markdown("""
    <div class="box-info">
    <table style="border-collapse:collapse; width:100%; font-size:0.88rem; line-height:2;">
        <tr><td style="padding-right:1.5rem; font-family:monospace; white-space:nowrap; font-weight:600;">V₀</td><td>= nilai investasi awal</td></tr>
        <tr><td style="font-family:monospace; white-space:nowrap; font-weight:600;">μ</td><td>= rata-rata return saham (mean) — diasumsikan ≈ 0 dan diabaikan dalam rumus</td></tr>
        <tr><td style="font-family:monospace; white-space:nowrap; font-weight:600;">σ</td><td>= standar deviasi return saham</td></tr>
        <tr><td style="font-family:monospace; white-space:nowrap; font-weight:600;">t</td><td>= holding period (hari)</td></tr>
        <tr><td style="font-family:monospace; white-space:nowrap; font-weight:600;">α</td><td>= tingkat signifikansi</td></tr>
        <tr><td style="font-family:monospace; white-space:nowrap; font-weight:600;">Z<sub>1−α</sub></td><td>= kuantil distribusi normal standar pada tingkat kepercayaan 1−α</td></tr>
        <tr><td style="font-family:monospace; white-space:nowrap; font-weight:600;">S</td><td>= skewness (kemencengan) return</td></tr>
        <tr><td style="font-family:monospace; white-space:nowrap; font-weight:600;">K</td><td>= kurtosis total (distribusi normal = 3)</td></tr>
        <tr><td style="font-family:monospace; white-space:nowrap; font-weight:600;">K<sub>e</sub> = K − 3</td><td>= excess kurtosis (distribusi normal = 0) — digunakan langsung dalam rumus CF</td></tr>
        <tr><td style="font-family:monospace; white-space:nowrap; font-weight:600;">Z<sub>CF</sub></td><td>= kuantil Cornish-Fisher yang sudah disesuaikan</td></tr>
        <tr><td style="font-family:monospace; white-space:nowrap; font-weight:600;">φ( · )</td><td>= fungsi densitas probabilitas (PDF) distribusi normal standar</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("#### Catatan Penggunaan")

    st.markdown("""
    <div class="box-warning">
        <strong>⚠️ Persyaratan Data</strong><br>
        Dalam penelitian ini, jumlah data yang digunakan minimal sebanyak
        <strong>250 observasi</strong> agar perhitungan return, Value at Risk (VaR), dan
        Expected Shortfall (ES) dapat dilakukan secara memadai. Namun, untuk memperoleh
        hasil backtesting yang lebih baik dan lebih stabil, disarankan menggunakan sekitar
        <strong>500 observasi</strong> atau lebih.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="box-info">
        <strong>ℹ️ Format File yang Didukung</strong><br>
        File Excel (<code>.xlsx</code> / <code>.xls</code>)
        dengan kolom tanggal (<code>Date</code> / <code>ref.date</code>)
        dan kolom harga penutupan (<code>Close</code> / <code>price.close</code>).
        Kolom OHLC dan Volume opsional, digunakan untuk candlestick chart.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="box-info">
        <strong>ℹ️ Interpretasi VaR &amp; ES</strong><br>
        VaR 95% dengan holding period 1 hari berarti terdapat 5% kemungkinan kerugian
        melebihi nilai VaR dalam satu hari. ES 95% menunjukkan rata-rata kerugian pada
        5% kasus terburuk tersebut — sehingga ES selalu ≥ VaR pada tingkat kepercayaan yang sama.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown('<p class="text-disclaimer">⚠️ Aplikasi ini dibuat untuk keperluan akademis dan edukasi.Selalu lakukan analisis mendalam sebelum mengambil keputusan investasi.</p>', unsafe_allow_html=True)
with tab2:
    st.header("Download Data Saham dari Yahoo Finance")

    st.markdown("""
    <div class="box-info">
        <strong>ℹ️ Informasi:</strong><br>
        • <strong>Saham Indonesia:</strong> Input kode saham <em>tanpa</em> .JK — sistem otomatis menambahkan (contoh: BBCA, TLKM)<br>
        • <strong>Saham Internasional:</strong> Gunakan ticker langsung (contoh: AAPL, MSFT, GOOGL)<br>
        • Atau pilih dari daftar preset saham populer yang tersedia
    </div>
    """, unsafe_allow_html=True)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    market_type = st.radio(
        "Pilih Jenis Pasar",
        ["🇮🇩 Saham Indonesia (IDX)", "🌐 Saham Internasional"],
        horizontal=True,
        key="market_type_tab2"
    )

    input_mode = st.radio(
        "Pilih Mode Input",
        ["Pilih dari Daftar Preset", "Input Manual Ticker"],
        horizontal=True,
        key="input_mode_tab2"
    )

    col1, col2 = st.columns(2)

    with col1:
        if input_mode == "Pilih dari Daftar Preset":
            if market_type == "🇮🇩 Saham Indonesia (IDX)":
                selected_stock = st.selectbox(
                    "Pilih Saham Indonesia",
                    options=list(STOCK_LIST_IDN.keys()),
                    format_func=lambda x: f"{x}.JK — {STOCK_LIST_IDN[x]}",
                    index=list(STOCK_LIST_IDN.keys()).index("JPFA"),
                    key="stock_select_idn_tab2"
                )
                ticker_symbol = f"{selected_stock}.JK"
                stock_name    = STOCK_LIST_IDN[selected_stock]
            else:
                selected_stock = st.selectbox(
                    "Pilih Saham Internasional",
                    options=list(STOCK_LIST_INTL.keys()),
                    format_func=lambda x: f"{x} — {STOCK_LIST_INTL[x]}",
                    index=list(STOCK_LIST_INTL.keys()).index("AAPL"),
                    key="stock_select_intl_tab2"
                )
                ticker_symbol = selected_stock
                stock_name    = STOCK_LIST_INTL[selected_stock]
        else:
            if market_type == "🇮🇩 Saham Indonesia (IDX)":
                manual_ticker = st.text_input(
                    "Masukkan Kode Saham Indonesia (tanpa .JK)",
                    value="BBCA",
                    help="Contoh: BBCA, TLKM, ASII, UNVR",
                    key="manual_ticker_idn_tab2"
                ).upper().strip()
                ticker_symbol = f"{manual_ticker}.JK"
                stock_name    = manual_ticker
                st.markdown(
                    f'<div class="box-info">Ticker yang akan diunduh: <strong>{ticker_symbol}</strong></div>',
                    unsafe_allow_html=True
                )
            else:
                manual_ticker = st.text_input(
                    "Masukkan Ticker Saham Internasional",
                    value="AAPL",
                    help="Contoh: AAPL, MSFT, TSLA, GOOGL, AMZN",
                    key="manual_ticker_intl_tab2"
                ).upper().strip()
                ticker_symbol = manual_ticker
                stock_name    = manual_ticker
                st.markdown(
                    f'<div class="box-info">Ticker yang akan diunduh: <strong>{ticker_symbol}</strong></div>',
                    unsafe_allow_html=True
                )

        start_date = st.date_input(
            "Tanggal Mulai",
            value=datetime.now() - timedelta(days=365),
            key="start_date_tab2"
        )

    with col2:
        end_date = st.date_input(
            "Tanggal Akhir",
            value=datetime.now(),
            key="end_date_tab2"
        )

    if st.button("Unduh Data Saham", type="primary", use_container_width=True, key="download_stock_tab2"):
        if not ticker_symbol or ticker_symbol.strip() in ("", ".JK"):
            st.markdown('<div class="box-error">❌ Ticker saham tidak boleh kosong.</div>', unsafe_allow_html=True)
        else:
            try:
                with st.spinner(f"Mengunduh data {ticker_symbol}..."):
                    data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)

                if data.empty:
                    st.markdown(f"""
                    <div class="box-error">
                        ❌ Data tidak ditemukan untuk ticker <strong>{ticker_symbol}</strong>.<br><br>
                        Contoh: BBCA · TLKM · ASII (IDX) &nbsp;|&nbsp; AAPL · MSFT · TSLA (Internasional)
                    </div>
                    """, unsafe_allow_html=True)
                    st.session_state.pop('tab2_close_price', None)
                else:
                    data = data.reset_index()
                    close_price = data[['Date','Open','High','Low','Close','Volume']].copy()
                    close_price.columns = ['Date','Open','High','Low','price.close','Volume']
                    close_price['Log Return'] = close_price['price.close'].pct_change().apply(
                        lambda x: np.log(1 + x) if pd.notna(x) and x != -1 else np.nan
                    )
                    st.session_state['tab2_close_price'] = close_price
                    st.session_state['tab2_ticker']      = ticker_symbol
                    st.session_state['tab2_stock_name']  = stock_name
                    st.session_state['tab2_currency']    = "Rp" if market_type == "🇮🇩 Saham Indonesia (IDX)" else "$"
                    st.session_state['tab2_market_type'] = market_type
                    st.session_state['downloaded_data']  = close_price
                    st.session_state['stock_ticker']     = ticker_symbol
                    st.session_state['stock_name']       = stock_name
                    st.session_state['market_type']      = market_type

            except Exception as e:
                st.markdown(f'<div class="box-error">❌ Error saat mengunduh: {str(e)}</div>', unsafe_allow_html=True)
                import traceback
                st.code(traceback.format_exc())

    if 'tab2_close_price' in st.session_state:
        close_price     = st.session_state['tab2_close_price']
        ticker_symbol   = st.session_state['tab2_ticker']
        stock_name      = st.session_state['tab2_stock_name']
        currency_symbol = st.session_state['tab2_currency']

        st.markdown(f"""
        <div class="box-success">
            ✅ <strong>Data tersedia</strong> — {ticker_symbol} · {stock_name} &nbsp;|&nbsp; {len(close_price)} baris
        </div>
        <div class="box-info">
            ➡️ Data ini sudah otomatis tersedia di <strong>Tab 3</strong> — pilih sumber
            <strong>"Gunakan Data dari Tab 2"</strong>, tidak perlu download &amp; upload ulang.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Preview Data")
        st.dataframe(close_price, use_container_width=True, height=360)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Statistik Deskriptif Log Return")

        log_returns     = close_price['Log Return'].dropna()
        clean_lr        = pd.Series(log_returns).replace([np.inf, -np.inf], np.nan).dropna()
        mean_ret        = clean_lr.mean()
        std_ret         = clean_lr.std(ddof=1)
        min_ret         = clean_lr.min()
        max_ret         = clean_lr.max()
        skew_ret        = stats.skew(clean_lr, bias=False)
        kurt_ret_excess = stats.kurtosis(clean_lr, fisher=True, bias=False)
        kurt_ret        = kurt_ret_excess + 3

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Mean",    f"{mean_ret:.6f}")
            st.metric("Std Dev", f"{std_ret:.6f}")
        with c2:
            st.metric("Min", f"{min_ret:.6f}")
            st.metric("Max", f"{max_ret:.6f}")
        with c3:
            st.metric("Skewness", f"{skew_ret:.6f}")
            st.metric("Kurtosis", f"{kurt_ret:.6f}")
        with c4:
            st.metric("Excess Kurtosis", f"{kurt_ret_excess:.6f}")
            st.metric("Jumlah Data",     f"{len(clean_lr)}")

        st.code(
            f"Kurtosis        = {kurt_ret:.4f}  (Distribusi Normal = 3)\n"
            f"Excess Kurtosis = {kurt_ret_excess:.4f}  (Distribusi Normal = 0)\n"
            "stats.kurtosis(fisher=True) → menghasilkan Excess Kurtosis",
            language=None
        )

        # ============================================================
        # Axis style — warna netral, terbaca di dark DAN light mode
        # ============================================================
        _axis = dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.15)',
            showline=True,
            linewidth=1,
            linecolor='rgba(128,128,128,0.3)',
            mirror=True,
            tickfont=dict(size=11),
            title_font=dict(size=12),
        )

        _range_sel = dict(
            buttons=[
                dict(count=7,    label="1 Minggu", step="day",   stepmode="backward"),
                dict(count=1,    label="1 Bulan",  step="month", stepmode="backward"),
                dict(count=3,    label="3 Bulan",  step="month", stepmode="backward"),
                dict(count=6,    label="6 Bulan",  step="month", stepmode="backward"),
                dict(count=1,    label="1 Tahun",  step="year",  stepmode="backward"),
                dict(count=3,    label="3 Tahun",  step="year",  stepmode="backward"),
                dict(count=5,    label="5 Tahun",  step="year",  stepmode="backward"),
                dict(step="all", label="Semua"),
            ],
            x=0, y=1.06, xanchor='left', yanchor='bottom'
        )

        # ============================================================
        # GRAFIK HARGA
        # ============================================================
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Grafik Harga Saham")

        chart_type = st.radio(
            "Pilih Jenis Chart",
            ["Candlestick", "Line Chart"],
            horizontal=True,
            key="chart_type_tab2"
        )

        close_price_plot = close_price[['Date','Open','High','Low','price.close','Volume']].copy()
        close_price_plot['Date'] = pd.to_datetime(close_price_plot['Date'])

        first_price      = close_price_plot['price.close'].iloc[0]
        last_price       = close_price_plot['price.close'].iloc[-1]
        price_change     = last_price - first_price
        price_change_pct = (price_change / first_price) * 100

        fig_price = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, row_heights=[0.7, 0.3]
        )

        if chart_type == "Candlestick":
            fig_price.add_trace(go.Candlestick(
                x=close_price_plot['Date'],
                open=close_price_plot['Open'],
                high=close_price_plot['High'],
                low=close_price_plot['Low'],
                close=close_price_plot['price.close'],
                name='OHLC',
                increasing=dict(line=dict(color='#16a34a'), fillcolor='#16a34a'),
                decreasing=dict(line=dict(color='#dc2626'), fillcolor='#dc2626'),
                hoverinfo="skip"
            ), row=1, col=1)
            fig_price.add_trace(go.Scatter(
                x=close_price_plot['Date'],
                y=close_price_plot['price.close'],
                mode='lines', line=dict(width=0), showlegend=False,
                hovertemplate=(
                    '<b>Tanggal</b>: %{x|%d/%m/%Y}<br>' +
                    f'<b>Open</b>: {currency_symbol} %{{customdata[0]:,.2f}}<br>' +
                    f'<b>High</b>: {currency_symbol} %{{customdata[1]:,.2f}}<br>' +
                    f'<b>Low</b>: {currency_symbol} %{{customdata[2]:,.2f}}<br>' +
                    f'<b>Close</b>: {currency_symbol} %{{customdata[3]:,.2f}}<br><extra></extra>'
                ),
                customdata=close_price_plot[['Open','High','Low','price.close']].values
            ), row=1, col=1)
        else:
            fig_price.add_trace(go.Scatter(
                x=close_price_plot['Date'],
                y=close_price_plot['price.close'],
                mode='lines', name='Harga Penutupan',
                line=dict(color='#3b82f6', width=1.8),
                hovertemplate=(
                    '<b>Tanggal</b>: %{x|%d/%m/%Y}<br>' +
                    f'<b>Close</b>: {currency_symbol} %{{y:,.2f}}<br><extra></extra>'
                )
            ), row=1, col=1)

        vol_colors = [
            '#dc2626' if close_price_plot['price.close'].iloc[i] < close_price_plot['Open'].iloc[i]
            else '#16a34a'
            for i in range(len(close_price_plot))
        ]
        fig_price.add_trace(go.Bar(
            x=close_price_plot['Date'], y=close_price_plot['Volume'],
            name='Volume', marker=dict(color=vol_colors, opacity=0.55),
            hovertemplate='<b>Tanggal</b>: %{x|%d/%m/%Y}<br><b>Volume</b>: %{y:,.0f}<br><extra></extra>'
        ), row=2, col=1)

        fig_price.update_layout(
            title=dict(
                text=(
                    f'<b>{ticker_symbol} — {stock_name}</b><br>'
                    f'<span style="font-size:12px;">'
                    f'Awal: {currency_symbol} {first_price:,.2f} &nbsp;|&nbsp; '
                    f'Akhir: {currency_symbol} {last_price:,.2f} &nbsp;|&nbsp; '
                    f'Perubahan: {price_change:+,.2f} ({price_change_pct:+.2f}%)</span>'
                ),
                x=0.5, xanchor='center', font=dict(size=15), y=0.97, yanchor='top'
            ),
            xaxis2=dict(title='Tanggal', **_axis),
            yaxis=dict(title=f'Harga ({currency_symbol})', **_axis),
            yaxis2=dict(title='Volume', **_axis),
            xaxis_rangeslider_visible=False,
            hovermode='x unified', height=680, showlegend=False,
            # ---- KUNCI: transparan → mengikuti background Streamlit ----
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=120, b=60, l=70, r=50)
        )
        fig_price.update_xaxes(**_axis)
        fig_price.update_yaxes(**_axis)
        fig_price.update_xaxes(rangeselector=_range_sel, row=2, col=1)
        st.plotly_chart(fig_price, use_container_width=True)

        # ============================================================
        # GRAFIK LOG RETURN
        # ============================================================
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Grafik Log Return")

        lr_plot = close_price[['Date','Log Return']].copy()
        lr_plot['Date'] = pd.to_datetime(lr_plot['Date'])
        lr_plot = lr_plot.dropna()

        mean_lr = lr_plot['Log Return'].mean()
        std_lr  = lr_plot['Log Return'].std()
        min_lr  = lr_plot['Log Return'].min()
        max_lr  = lr_plot['Log Return'].max()

        fig_ret = go.Figure()
        fig_ret.add_trace(go.Scatter(
            x=lr_plot['Date'], y=lr_plot['Log Return'],
            mode='lines', name='Log Return',
            line=dict(color='#6366f1', width=1.3),
            hovertemplate='<b>Tanggal</b>: %{x|%d/%m/%Y}<br><b>Return</b>: %{y:.6f}<br><extra></extra>'
        ))

        fig_ret.add_hline(y=0,              line_dash="dash", line_color="rgba(128,128,128,0.5)", line_width=1)
        fig_ret.add_hline(y=mean_lr,        line_dash="dot",  line_color="#f59e0b", line_width=1.5, opacity=0.9)
        fig_ret.add_hline(y=mean_lr+std_lr, line_dash="dot",  line_color="#f59e0b", line_width=1,   opacity=0.5)
        fig_ret.add_hline(y=mean_lr-std_lr, line_dash="dot",  line_color="#f59e0b", line_width=1,   opacity=0.5)

        for label, val in [
            (f"Mean: {mean_lr:.6f}",       mean_lr),
            (f"+1σ: {mean_lr+std_lr:.6f}", mean_lr+std_lr),
            (f"−1σ: {mean_lr-std_lr:.6f}", mean_lr-std_lr),
        ]:
            fig_ret.add_annotation(
                x=0.01, y=val, xref='paper', yref='y',
                text=label, showarrow=False,
                font=dict(color='#d97706', size=9),
                bgcolor='rgba(251,191,36,0.12)',
                bordercolor='#f59e0b', borderwidth=1,
                xanchor='left'
            )

        fig_ret.update_layout(
            title=dict(
                text=(
                    f'<b>Log Return — {ticker_symbol}</b><br>'
                    f'<span style="font-size:12px;">'
                    f'Mean: {mean_lr:.6f} &nbsp;|&nbsp; Std: {std_lr:.6f} &nbsp;|&nbsp; '
                    f'Min: {min_lr:.6f} &nbsp;|&nbsp; Max: {max_lr:.6f}</span>'
                ),
                x=0.5, xanchor='center', font=dict(size=15), y=0.97, yanchor='top'
            ),
            xaxis=dict(title='Tanggal', **_axis),
            yaxis=dict(title='Log Return', zeroline=True,
                       zerolinewidth=1, zerolinecolor='rgba(128,128,128,0.3)', **_axis),
            hovermode='x unified', height=500, showlegend=False,
            # ---- KUNCI: transparan ----
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=110, b=60, l=70, r=50)
        )
        fig_ret.update_xaxes(rangeselector=_range_sel)
        st.plotly_chart(fig_ret, use_container_width=True)

        # ---- Download Excel ----
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            close_price.to_excel(writer, index=False, sheet_name='Data')

        safe_filename = ticker_symbol.replace(".", "_")
        st.download_button(
            label="💾 Download ke Excel",
            data=output.getvalue(),
            file_name=f"{safe_filename}_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
with tab3:
    st.header("Data & Uji Normalitas")

    st.markdown("""
    <div class="box-info">
        <strong>ℹ️ Informasi:</strong><br>
        • Sumber data bisa <strong>langsung dari Tab 2</strong> (hasil unduhan Yahoo Finance) atau
          <strong>upload file Excel</strong> sendiri<br>
        • File Excel (.xlsx / .xls) dengan kolom tanggal (<code>Date</code> / <code>ref.date</code>) dan harga penutupan (<code>Close</code> / <code>price.close</code>)<br>
        • Kolom OHLC dan Volume bersifat opsional — digunakan untuk candlestick chart<br>
        • Return yang dianalisis adalah <strong>log return</strong> [ln(P<sub>t</sub>/P<sub>t−1</sub>)]<br>
        • Normalitas dievaluasi menggunakan uji <strong>Jarque–Bera</strong> — dipilih karena menilai kenormalan
          berdasarkan skewness dan kurtosis, relevan dengan pendekatan Cornish–Fisher
    </div>
    """, unsafe_allow_html=True)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # ============================================================
    # PILIH SUMBER DATA — dari Tab 2 atau upload Excel
    # ============================================================
    SRC_TAB2   = "📥 Gunakan Data dari Tab 2 (Yahoo Finance)"
    SRC_UPLOAD = "📂 Upload File Excel"

    tab2_available = 'tab2_close_price' in st.session_state

    data_source = st.radio(
        "Sumber Data",
        [SRC_TAB2, SRC_UPLOAD],
        index=0 if tab2_available else 1,
        horizontal=True,
        key="data_source_tab3"
    )
    use_tab2 = (data_source == SRC_TAB2)

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        alpha_test = st.selectbox(
            "Taraf Signifikansi (α)",
            options=[0.01, 0.05, 0.10],
            format_func=lambda x: f"{int(x*100)}%  (α = {x})",
            index=1,
            key="alpha_test_tab3"
        )
    with col_cfg2:
        if use_tab2 and tab2_available:
            # Jenis pasar mengikuti pilihan di Tab 2 agar mata uang konsisten
            market_type = st.session_state.get('tab2_market_type', "🇮🇩 Saham Indonesia (IDX)")
            st.markdown(
                f'<p class="text-hint">Jenis Pasar (mengikuti Tab 2)</p>'
                f'<p class="text-body"><strong>{market_type}</strong></p>',
                unsafe_allow_html=True
            )
        else:
            market_type = st.radio(
                "Jenis Pasar",
                ["🇮🇩 Saham Indonesia (IDX)", "🌐 Saham Internasional"],
                horizontal=True,
                key="market_type_tab3"
            )

    df_source     = None   # DataFrame dari Tab 2
    uploaded_file = None   # file Excel dari uploader

    if use_tab2:
        if tab2_available:
            df_source = (st.session_state['tab2_close_price']
                         .drop(columns=['Log Return'], errors='ignore')
                         .copy())
            _t2_ticker = st.session_state.get('tab2_ticker', '-')
            _t2_name   = st.session_state.get('tab2_stock_name', '-')
            _t2_dates  = pd.to_datetime(df_source['Date'], errors='coerce').dropna()
            _t2_period = (f"{_t2_dates.min():%d/%m/%Y} – {_t2_dates.max():%d/%m/%Y}"
                          if not _t2_dates.empty else "-")
            st.markdown(
                f'<div class="box-success">✅ Menggunakan data dari Tab 2 — '
                f'<strong>{_t2_ticker}</strong> · {_t2_name} &nbsp;|&nbsp; '
                f'{len(df_source)} baris &nbsp;|&nbsp; Periode {_t2_period}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="box-warning">⚠️ Belum ada data di Tab 2. Silakan unduh data saham di '
                '<strong>Tab 2 (Download Data Saham)</strong> terlebih dahulu, atau pilih '
                '<strong>Upload File Excel</strong>.</div>',
                unsafe_allow_html=True
            )
    else:
        uploaded_file = st.file_uploader(
            "Upload File Excel Data Saham",
            type=['xlsx', 'xls'],
            key="upload_file_tab3"
        )

    if df_source is not None or uploaded_file is not None:
        try:
            if uploaded_file is not None:
                df = pd.read_excel(uploaded_file)
            else:
                df = df_source

            # ---- Deteksi kolom tanggal ----
            date_col = None
            for name in ['Date','date','DATE','ref.date','Ref.Date','REF.DATE',
                         'Tanggal','tanggal','TANGGAL','Time','time','datetime']:
                if name in df.columns:
                    date_col = name
                    break

            if date_col is None:
                st.markdown(
                    '<div class="box-warning">⚠️ Kolom tanggal tidak ditemukan. Data diproses tanpa index tanggal.<br>'
                    'Kolom tersedia: ' + ", ".join(df.columns.tolist()) + '</div>',
                    unsafe_allow_html=True
                )
            else:
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    df = df.dropna(subset=[date_col])
                    df = df.set_index(date_col)
                    st.markdown(
                        f'<div class="box-success">✅ Kolom tanggal ditemukan: <strong>{date_col}</strong></div>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.markdown(
                        f'<div class="box-warning">⚠️ Tidak dapat parse kolom tanggal: {str(e)}</div>',
                        unsafe_allow_html=True
                    )

            # ---- Deteksi kolom close ----
            close_col = None
            for name in ['Close','close','Close Price','close price','price.close',
                         'CLOSE','Closing Price','closing price','PRICE.CLOSE',
                         'Adj Close','adj close','ADJ CLOSE']:
                if name in df.columns:
                    close_col = name
                    break

            if close_col is None:
                st.markdown(
                    '<div class="box-error">❌ Kolom harga penutupan tidak ditemukan.<br>'
                    'Kolom tersedia: ' + ", ".join(df.columns.tolist()) + '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="box-success">✅ Kolom harga ditemukan: <strong>{close_col}</strong></div>',
                    unsafe_allow_html=True
                )

                # ---- Hitung Log Return & bersihkan ----
                df['Log Return'] = np.log(df[close_col] / df[close_col].shift(1))
                log_returns   = df['Log Return'].dropna()
                clean_returns = pd.Series(log_returns).replace([np.inf, -np.inf], np.nan).dropna()
                n             = len(clean_returns)

                st.markdown(
                    f'<div class="box-success">✅ File berhasil diproses — <strong>{n}</strong> observasi log return valid</div>',
                    unsafe_allow_html=True
                )

                currency_symbol = "Rp" if market_type == "🇮🇩 Saham Indonesia (IDX)" else "$"

                # _ax dan _rs didefinisikan DI SINI (sebelum blok candlestick)
                # agar selalu tersedia meski file tidak punya OHLC/tanggal
                _ax = dict(
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.15)',
                    showline=True, linewidth=1,
                    linecolor='rgba(128,128,128,0.3)',
                    mirror=True,
                    tickfont=dict(size=11),
                    title_font=dict(size=12),
                )
                _rs = dict(
                    buttons=[
                        dict(count=7,  label="1 Minggu", step="day",   stepmode="backward"),
                        dict(count=1,  label="1 Bulan",  step="month", stepmode="backward"),
                        dict(count=3,  label="3 Bulan",  step="month", stepmode="backward"),
                        dict(count=6,  label="6 Bulan",  step="month", stepmode="backward"),
                        dict(count=1,  label="1 Tahun",  step="year",  stepmode="backward"),
                        dict(count=3,  label="3 Tahun",  step="year",  stepmode="backward"),
                        dict(step="all", label="Semua")
                    ],
                    x=0, y=1.06, xanchor='left', yanchor='bottom'
                )

                # ============================================================
                # PREVIEW TABEL
                # ============================================================
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("#### Preview Data")

                display_cols = [c for c in [close_col, 'Log Return'] if c in df.columns]
                df_display   = df[display_cols].reset_index() if (date_col and isinstance(df.index, pd.DatetimeIndex)) else df[display_cols]
                st.dataframe(df_display, use_container_width=True, height=400)

                # ============================================================
                # GRAFIK HARGA — toggle candlestick / line
                # ============================================================
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("#### Grafik Harga Saham")

                has_ohlc = all(c in df.columns for c in ['Open','High','Low', close_col])

                if has_ohlc and isinstance(df.index, pd.DatetimeIndex):
                    chart_type_t3 = st.radio(
                        "Pilih Jenis Chart",
                        ["Candlestick", "Line Chart"],
                        horizontal=True,
                        key="chart_type_tab3"
                    )

                    cd = df[[close_col,'Open','High','Low']].copy()
                    if 'Volume' in df.columns:
                        cd['Volume'] = df['Volume']
                    cd = cd.reset_index()
                    cd['Date'] = pd.to_datetime(cd[date_col])

                    first_p = cd[close_col].iloc[0]
                    last_p  = cd[close_col].iloc[-1]
                    chg     = last_p - first_p
                    chg_pct = (chg / first_p) * 100

                    has_vol  = 'Volume' in cd.columns
                    row_h    = [0.7, 0.3] if has_vol else [1.0]
                    n_rows   = 2 if has_vol else 1

                    fig_p = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                                          vertical_spacing=0.05, row_heights=row_h)

                    if chart_type_t3 == "Candlestick":
                        fig_p.add_trace(go.Candlestick(
                            x=cd['Date'], open=cd['Open'], high=cd['High'],
                            low=cd['Low'], close=cd[close_col], name='OHLC',
                            increasing=dict(line=dict(color='#26A69A')),
                            decreasing=dict(line=dict(color='#EF5350')),
                            hoverinfo="skip"
                        ), row=1, col=1)
                        fig_p.add_trace(go.Scatter(
                            x=cd['Date'], y=cd[close_col], mode='lines',
                            line=dict(width=0), showlegend=False,
                            hovertemplate=(
                                '<b>Tanggal</b>: %{x|%d/%m/%Y}<br>' +
                                f'<b>Open</b>: {currency_symbol} %{{customdata[0]:,.2f}}<br>' +
                                f'<b>High</b>: {currency_symbol} %{{customdata[1]:,.2f}}<br>' +
                                f'<b>Low</b>: {currency_symbol} %{{customdata[2]:,.2f}}<br>' +
                                f'<b>Close</b>: {currency_symbol} %{{customdata[3]:,.2f}}<br><extra></extra>'
                            ),
                            customdata=cd[['Open','High','Low',close_col]].values
                        ), row=1, col=1)
                    else:
                        fig_p.add_trace(go.Scatter(
                            x=cd['Date'], y=cd[close_col], mode='lines',
                            name='Harga Penutupan', line=dict(color='#3b82f6', width=1.8),
                            hovertemplate=(
                                '<b>Tanggal</b>: %{x|%d/%m/%Y}<br>' +
                                f'<b>Close</b>: {currency_symbol} %{{y:,.2f}}<br><extra></extra>'
                            )
                        ), row=1, col=1)

                    if has_vol:
                        vol_c = ['#EF5350' if cd[close_col].iloc[i] < cd['Open'].iloc[i] else '#26A69A'
                                 for i in range(len(cd))]
                        fig_p.add_trace(go.Bar(
                            x=cd['Date'], y=cd['Volume'], name='Volume',
                            marker=dict(color=vol_c, opacity=0.55),
                            hovertemplate='<b>Tanggal</b>: %{x|%d/%m/%Y}<br><b>Volume</b>: %{y:,.0f}<br><extra></extra>'
                        ), row=2, col=1)

                    fig_p.update_layout(
                        title=dict(
                            text=(f'<b>Grafik Harga Saham (Harga Penutupan)</b><br>'
                                  f'<span style="font-size:12px;">'
                                  f'Awal: {currency_symbol} {first_p:,.2f} &nbsp;|&nbsp; '
                                  f'Akhir: {currency_symbol} {last_p:,.2f} &nbsp;|&nbsp; '
                                  f'Perubahan: {chg:+,.2f} ({chg_pct:+.2f}%)</span>'),
                            x=0.5, xanchor='center',
                            font=dict(size=15), y=0.97, yanchor='top'
                        ),
                        yaxis=dict(title=f'Harga Penutupan ({currency_symbol})', **_ax),
                        **({"xaxis2": dict(title='Tanggal', **_ax),
                            "yaxis2": dict(title='Volume', **_ax)} if has_vol else
                           {"xaxis": dict(title='Tanggal', **_ax)}),
                        xaxis_rangeslider_visible=False,
                        hovermode='x unified', height=660 if has_vol else 460,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=110, b=60, l=70, r=50)
                    )
                    fig_p.update_xaxes(**_ax)
                    fig_p.update_yaxes(**_ax)
                    if has_vol:
                        fig_p.update_xaxes(rangeselector=_rs, row=2, col=1)
                    else:
                        fig_p.update_xaxes(rangeselector=_rs, row=1, col=1)
                    st.plotly_chart(fig_p, use_container_width=True)
                else:
                    st.markdown(
                        '<div class="box-info">ℹ️ Candlestick chart memerlukan kolom Open, High, Low, Close, dan Date.</div>',
                        unsafe_allow_html=True
                    )

                # ============================================================
                # HISTOGRAM FREKUENSI — Harga Penutupan & Log Return
                # ============================================================
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("#### Histogram Distribusi Data")

                fig_hist, (ax_h1, ax_h2) = plt.subplots(1, 2, figsize=(14, 5))
                fig_hist.patch.set_alpha(0.0)

                # ---- Histogram Harga Penutupan (frekuensi) ----
                close_vals_h = df[close_col].dropna().values
                ax_h1.set_facecolor('none')
                ax_h1.patch.set_alpha(0.0)
                ax_h1.hist(close_vals_h, bins=30, color='#22c55e', edgecolor=(0,0,0,0.15),
                           alpha=0.8, label='Harga Penutupan')
                ax_h1.set_xlabel(f'Harga Penutupan ({currency_symbol})', fontsize=10)
                ax_h1.set_ylabel('Frekuensi', fontsize=10)
                ax_h1.set_title('Histogram Harga Penutupan', fontsize=12, fontweight='bold')
                ax_h1.tick_params(labelsize=9)
                for sp in ax_h1.spines.values():
                    sp.set_color('gray'); sp.set_alpha(0.4)
                ax_h1.legend(fontsize=9)
                ax_h1.grid(True, alpha=0.2, axis='y')

                # ---- Histogram Log Return (frekuensi) ----
                ax_h2.set_facecolor('none')
                ax_h2.patch.set_alpha(0.0)
                ax_h2.hist(clean_returns.values, bins=30, color='#6366f1', edgecolor=(0,0,0,0.15),
                           alpha=0.8, label='Log Return')
                ax_h2.set_xlabel('Return  [ln(Pₜ/Pₜ₋₁)]', fontsize=10)
                ax_h2.set_ylabel('Frekuensi', fontsize=10)
                ax_h2.set_title('Histogram Return', fontsize=12, fontweight='bold')
                ax_h2.tick_params(labelsize=9)
                for sp in ax_h2.spines.values():
                    sp.set_color('gray'); sp.set_alpha(0.4)
                ax_h2.legend(fontsize=9)
                ax_h2.grid(True, alpha=0.2, axis='y')

                plt.tight_layout()
                st.pyplot(fig_hist, transparent=True)
                plt.close()

                # ============================================================
                # STATISTIK DESKRIPTIF DATA HISTORIS SAHAM
                # ============================================================
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("#### Statistik Deskriptif Data Historis Saham")

                st.markdown(
                    '<div class="box-info">ℹ️ Statistik berikut dihitung dari seluruh data harga penutupan '
                    'dan log return historis. Nilai skewness dan kurtosis menggunakan estimator '
                    '<strong>bias-corrected</strong> (<code>bias=False</code>) yang cocok untuk analisis '
                    'sampel. Nilai yang dipakai dalam rumus Jarque–Bera menggunakan estimator yang konsisten '
                    'dengan implementasi SciPy dan ditampilkan terpisah di bagian uji normalitas.</div>',
                    unsafe_allow_html=True
                )

                # Hitung statistik harga penutupan
                close_s = pd.Series(close_vals_h)
                close_log_ret = clean_returns

                col_stat1, col_stat2 = st.columns(2)

                with col_stat1:
                    st.markdown("**Harga Penutupan**")
                    df_stat_close = pd.DataFrame([
                        {"Statistik": "Jumlah Observasi", "Nilai": f"{len(close_s):,}"},
                        {"Statistik": "Minimum",          "Nilai": f"{currency_symbol} {close_s.min():,.2f}"},
                        {"Statistik": "Maksimum",         "Nilai": f"{currency_symbol} {close_s.max():,.2f}"},
                        {"Statistik": "Mean",             "Nilai": f"{currency_symbol} {close_s.mean():,.2f}"},
                        {"Statistik": "Median",           "Nilai": f"{currency_symbol} {close_s.median():,.2f}"},
                        {"Statistik": "Std Dev",          "Nilai": f"{currency_symbol} {close_s.std(ddof=1):,.2f}"},
                        {"Statistik": "Skewness (S)",     "Nilai": f"{float(stats.skew(close_s, bias=False)):.6f}"},
                        {"Statistik": "Kurtosis (K)",     "Nilai": f"{float(stats.kurtosis(close_s, fisher=True, bias=False)) + 3:.6f}"},
                        {"Statistik": "Excess Kurtosis",  "Nilai": f"{float(stats.kurtosis(close_s, fisher=True, bias=False)):.6f}"},
                    ])
                    st.dataframe(df_stat_close, use_container_width=True, hide_index=True)

                with col_stat2:
                    st.markdown("**Log Return**")
                    df_stat_lr = pd.DataFrame([
                        {"Statistik": "Jumlah Observasi", "Nilai": f"{len(close_log_ret):,}"},
                        {"Statistik": "Minimum",          "Nilai": f"{close_log_ret.min():.6f}"},
                        {"Statistik": "Maksimum",         "Nilai": f"{close_log_ret.max():.6f}"},
                        {"Statistik": "Mean",             "Nilai": f"{close_log_ret.mean():.6f}"},
                        {"Statistik": "Median",           "Nilai": f"{close_log_ret.median():.6f}"},
                        {"Statistik": "Std Dev",          "Nilai": f"{close_log_ret.std(ddof=1):.6f}"},
                        {"Statistik": "Skewness (S)",     "Nilai": f"{float(stats.skew(close_log_ret, bias=False)):.6f}"},
                        {"Statistik": "Kurtosis (K)",     "Nilai": f"{float(stats.kurtosis(close_log_ret, fisher=True, bias=False)) + 3:.6f}"},
                        {"Statistik": "Excess Kurtosis",  "Nilai": f"{float(stats.kurtosis(close_log_ret, fisher=True, bias=False)):.6f}"},
                    ])
                    st.dataframe(df_stat_lr, use_container_width=True, hide_index=True)

                st.markdown(
                    '<p class="text-hint">— Catatan estimator kurtosis —</p>',
                    unsafe_allow_html=True
                )
                _skew_bc  = float(stats.skew(clean_returns, bias=False))
                _ek_bc    = float(stats.kurtosis(clean_returns, fisher=True, bias=False))
                _skew_raw = float(stats.skew(clean_returns, bias=True))
                _ek_raw   = float(stats.kurtosis(clean_returns, fisher=True, bias=True))
                st.code(
                    f"Tabel di atas memakai bias=False (bias-corrected):\n"
                    f"  S   = {_skew_bc:.6f},  K_e = {_ek_bc:.6f}\n\n"
                    f"Rumus Jarque–Bera memakai bias=True (konsisten dengan stats.jarque_bera):\n"
                    f"  S   = {_skew_raw:.6f},  K_e = {_ek_raw:.6f}\n\n"
                    f"Perbedaan kecil ini wajar — keduanya adalah estimator yang valid\n"
                    f"untuk momen yang sama, dengan koreksi bias yang berbeda.",
                    language=None
                )
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("#### Grafik Log Return")

                lr_plot  = log_returns.copy()
                mean_lr  = lr_plot.mean()
                std_lr   = lr_plot.std()
                min_lr   = lr_plot.min()
                max_lr   = lr_plot.max()

                fig_lr = go.Figure()
                fig_lr.add_trace(go.Scatter(
                    x=lr_plot.index if isinstance(lr_plot.index, pd.DatetimeIndex) else list(range(len(lr_plot))),
                    y=lr_plot.values, mode='lines', name='Log Return',
                    line=dict(color='#818cf8', width=1.3),
                    hovertemplate=(
                        '<b>Tanggal</b>: %{x|%d/%m/%Y}<br><b>Return</b>: %{y:.6f}<br><extra></extra>'
                        if isinstance(lr_plot.index, pd.DatetimeIndex) else
                        '<b>Index</b>: %{x}<br><b>Return</b>: %{y:.6f}<br><extra></extra>'
                    )
                ))
                fig_lr.add_hline(y=0,              line_dash="dash", line_color="#475569", line_width=1,   opacity=0.8)
                fig_lr.add_hline(y=mean_lr,        line_dash="dot",  line_color="#f59e0b", line_width=1.5, opacity=0.9)
                fig_lr.add_hline(y=mean_lr+std_lr, line_dash="dot",  line_color="#f59e0b", line_width=1,   opacity=0.5)
                fig_lr.add_hline(y=mean_lr-std_lr, line_dash="dot",  line_color="#f59e0b", line_width=1,   opacity=0.5)
                for lbl, val in [(f"Mean: {mean_lr:.6f}", mean_lr),
                                  (f"+1σ: {mean_lr+std_lr:.6f}", mean_lr+std_lr),
                                  (f"−1σ: {mean_lr-std_lr:.6f}", mean_lr-std_lr)]:
                    fig_lr.add_annotation(x=0.01, y=val, xref='paper', yref='y',
                        text=lbl, showarrow=False, font=dict(color='#d97706', size=9),
                        bgcolor='rgba(251,191,36,0.12)', bordercolor='#f59e0b',
                        borderwidth=1, xanchor='left')
                fig_lr.update_layout(
                    title=dict(
                        text=(f'<b>Grafik Log Return Harga Saham</b><br>'
                              f'<span style="font-size:12px;">'
                              f'Mean: {mean_lr:.6f} &nbsp;|&nbsp; Std: {std_lr:.6f} &nbsp;|&nbsp; '
                              f'Min: {min_lr:.6f} &nbsp;|&nbsp; Max: {max_lr:.6f}</span>'),
                        x=0.5, xanchor='center', font=dict(size=15),
                        y=0.97, yanchor='top'
                    ),
                    xaxis=dict(title='Tanggal' if isinstance(lr_plot.index, pd.DatetimeIndex) else 'Index', **_ax),
                    yaxis=dict(title='Log Return', zeroline=True, zerolinewidth=1,
                               zerolinecolor='rgba(128,128,128,0.3)', **_ax),
                    hovermode='x unified', height=480, showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=110, b=60, l=70, r=50)
                )
                if isinstance(lr_plot.index, pd.DatetimeIndex):
                    fig_lr.update_xaxes(rangeselector=_rs)
                st.plotly_chart(fig_lr, use_container_width=True)

                # ============================================================
                # STATISTIK DESKRIPTIF
                # ============================================================
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("#### Statistik Deskriptif")

                mean_ret        = clean_returns.mean()
                std_ret         = clean_returns.std(ddof=1)
                skew_ret        = stats.skew(clean_returns, bias=False)
                kurt_ret_excess = stats.kurtosis(clean_returns, fisher=True, bias=False)
                kurt_ret        = kurt_ret_excess + 3

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Mean",    f"{mean_ret:.7f}")
                    st.metric("Median",  f"{clean_returns.median():.7f}")
                with c2:
                    st.metric("Std Dev",   f"{std_ret:.7f}")
                    st.metric("Variance",  f"{clean_returns.var(ddof=1):.7f}")
                with c3:
                    st.metric("Skewness (S)", f"{skew_ret:.7f}")
                    st.metric("Kurtosis (K)", f"{kurt_ret:.7f}")
                with c4:
                    st.metric("Excess Kurtosis", f"{kurt_ret_excess:.7f}")
                    st.metric("Min / Max",
                              f"{clean_returns.min():.4f} / {clean_returns.max():.4f}")

                st.markdown(
                    '<p style="color:#475569; font-size:0.8rem; margin-bottom:0.2rem; font-family:monospace;">— Catatan Kurtosis —</p>',
                    unsafe_allow_html=True
                )
                # PATCH 1: baris Jarque-Bera dihapus
                st.code(
                    f"Kurtosis        = {kurt_ret:.4f}  (Distribusi Normal = 3)\n"
                    f"Excess Kurtosis = {kurt_ret_excess:.4f}  (Distribusi Normal = 0)\n"
                    "stats.kurtosis(fisher=True) → menghasilkan Excess Kurtosis",
                    language=None
                )

                # ============================================================
                # UJI NORMALITAS — JARQUE–BERA
                # ============================================================
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("#### Uji Normalitas")

                st.markdown("""
                <div class="box-info">
                    <strong>ℹ️ Latar Belakang Pengujian Normalitas</strong><br>
                    Pada tahap ini dilakukan pengujian normalitas data return saham untuk mengetahui apakah
                    distribusi return cenderung mengikuti distribusi normal atau tidak. Pengujian ini penting
                    karena metode <strong>Variance-Covariance</strong> mengasumsikan return berdistribusi normal,
                    sedangkan metode <strong>Cornish-Fisher</strong> merupakan penyesuaian ketika data menunjukkan
                    penyimpangan dari normalitas — terutama akibat skewness dan kurtosis.<br><br>
                    <strong>Hipotesis:</strong> H₀: data return berdistribusi normal &nbsp;|&nbsp;
                    H₁: data return tidak berdistribusi normal<br>
                    <strong>Kriteria:</strong> Tolak H₀ jika p-value ≤ α.
                </div>
                """, unsafe_allow_html=True)

                # is_normal_overall: None = belum diketahui, True = normal, False = tidak normal
                is_normal_overall = None
                jb_stat, jb_pval  = None, None

                if n < 3:
                    st.markdown(
                        '<div class="box-error">❌ Data terlalu sedikit untuk uji normalitas (minimal 3 observasi).</div>',
                        unsafe_allow_html=True
                    )
                else:
                    import math as _math

                    def _fmt_pval_dual(p):
                        """Kembalikan (pval_decimal, pval_scientific_html, pval_sentence)."""
                        if p is None: return "—", "—", "p-value tidak tersedia"
                        if p == 0.0: return "0", "0", "p-value ≈ 0"
                        if p >= 0.0001:
                            dec_digits = max(4, -int(_math.floor(_math.log10(p))) + 3)
                            dec_digits = min(dec_digits, 15)
                            dec   = f"{p:.{dec_digits}f}"
                            exp   = int(_math.floor(_math.log10(p)))
                            coeff = p / (10 ** exp)
                            sci   = f"{coeff:.2f} × 10<sup>{exp}</sup>"
                            sent  = f"p-value = {p:.6f}"
                        else:
                            dec   = f"{p:.20f}".rstrip('0').rstrip('.')
                            exp   = int(_math.floor(_math.log10(p)))
                            coeff = p / (10 ** exp)
                            exp_str = str(exp).replace('-', '&#8722;')
                            sci   = f"{coeff:.2f} × 10<sup>{exp_str}</sup>"
                            sent  = f"p-value = {dec}"
                        return dec, sci, sent

                    # ---- Jarque–Bera ----
                    try:
                        jb_stat, jb_pval = stats.jarque_bera(clean_returns)
                    except Exception as e:
                        st.markdown(f'<div class="box-error">❌ Jarque–Bera error: {str(e)}</div>',
                                    unsafe_allow_html=True)

                    # Keputusan dari JB — fallback None jika JB gagal
                    is_normal_overall = (jb_pval > alpha_test) if jb_pval is not None else None

                    # ---- Rumus JB — gunakan bias=True agar konsisten dengan stats.jarque_bera ----
                    skew_jb = float(stats.skew(clean_returns, bias=True))
                    kurt_excess_jb = float(stats.kurtosis(clean_returns, fisher=True, bias=True))
                    jb_calc = n * (skew_jb**2 / 6 + kurt_excess_jb**2 / 24)
                    st.markdown(
                        '<p class="text-hint">— Rumus Jarque–Bera —</p>',
                        unsafe_allow_html=True
                    )
                    st.code(
                        f"JB = n × (S² / 6  +  K_e² / 24)\n"
                        f"   = {n} × ({skew_jb:.6f}² / 6  +  {kurt_excess_jb:.6f}² / 24)\n"
                        f"   = {jb_calc:.6f}  (konsisten dengan stats.jarque_bera)\n\n"
                        f"n   = {n}  (jumlah observasi)\n"
                        f"S   = {skew_jb:.6f}  (skewness, bias=True)\n"
                        f"K_e = {kurt_excess_jb:.6f}  (excess kurtosis, bias=True)\n\n"
                        f"Catatan: statistik deskriptif di atas memakai bias=False (bias-corrected).\n"
                        f"Untuk konsistensi rumus JB, di sini dipakai bias=True sesuai implementasi SciPy.\n"
                        f"JB ~ χ²(2) secara asimtotik",
                        language=None
                    )

                    st.markdown("""
                    <div class="box-info">
                        <strong>🎯 Mengapa Jarque–Bera?</strong><br><br>
                        Dalam penelitian ini, uji normalitas dilakukan dengan metode <strong>Jarque–Bera</strong>.
                        Uji ini dipilih karena menilai kenormalan data berdasarkan <strong>skewness</strong> dan
                        <strong>kurtosis</strong>, sehingga sesuai dengan karakteristik data return saham serta
                        relevan dengan pendekatan <strong>Cornish–Fisher</strong> yang juga mempertimbangkan
                        kedua ukuran tersebut.
                    </div>
                    """, unsafe_allow_html=True)

                    # ---- Tabel hasil uji ----
                    _SUP_MAP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

                    def _pval_sci_unicode(p):
                        if p is None: return "—"
                        if p == 0.0: return "≈ 0"
                        try:
                            exp   = int(_math.floor(_math.log10(p)))
                            coeff = p / (10 ** exp)
                            sup   = str(exp).translate(_SUP_MAP)
                            return f"{coeff:.2f} × 10{sup}"
                        except Exception:
                            return f"{p:.2e}"

                    def _pval_str(p):
                        if p is None: return "—"
                        d, _, _ = _fmt_pval_dual(p)
                        return d

                    jb_dec, jb_sci, jb_sent = _fmt_pval_dual(jb_pval)
                    df_norm = pd.DataFrame([{
                        'Uji Normalitas'      : 'Jarque–Bera',
                        'Statistik'           : f"{jb_stat:.6f}" if jb_stat is not None else "—",
                        'P-Value (desimal)'   : _pval_str(jb_pval),
                        'P-Value (saintifik)' : _pval_sci_unicode(jb_pval),
                        'α'                   : f"{alpha_test}",
                        'Keputusan'           : ('✅ Gagal Tolak H₀ — Normal'
                                                  if (jb_pval is not None and jb_pval > alpha_test)
                                                  else ('❌ Tolak H₀ — Tidak Normal'
                                                        if jb_pval is not None else '—')),
                    }])
                    st.dataframe(df_norm, use_container_width=True, hide_index=True)

                    # Detail p-value
                    if jb_pval is not None:
                        st.markdown(
                            f'<div class="box-info"><strong>Detail p-value:</strong><br>'
                            f'• <strong>Jarque–Bera</strong>: desimal = <code>{jb_dec}</code>'
                            f' &nbsp;|&nbsp; saintifik = {jb_sci}</div>',
                            unsafe_allow_html=True
                        )

                    # ---- Kesimpulan ----
                    if jb_pval is not None:
                        if is_normal_overall is True:
                            st.markdown(
                                f'<div class="box-success">✅ <strong>Kesimpulan (Jarque–Bera): '
                                f'Data berdistribusi normal</strong><br>'
                                f'({jb_sent} &gt; α = {alpha_test}). '
                                f'Data return saham dapat dianggap berdistribusi normal, '
                                f'sehingga asumsi dasar metode Variance-Covariance relatif terpenuhi.</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="box-warning">⚠️ <strong>Kesimpulan (Jarque–Bera): '
                                f'Data tidak berdistribusi normal</strong><br>'
                                f'({jb_sent}; α = {alpha_test}). '
                                f'Data return saham tidak berdistribusi normal, sehingga pendekatan '
                                f'Cornish-Fisher menjadi penting sebagai penyesuaian terhadap skewness '
                                f'dan kurtosis data. Pemilihan metode akhir tetap dievaluasi melalui '
                                f'backtesting di Tab 6.</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown(
                            '<div class="box-warning">⚠️ Uji Jarque–Bera tidak berhasil dihitung, sehingga '
                            'kesimpulan normalitas belum dapat ditetapkan secara otomatis.</div>',
                            unsafe_allow_html=True
                        )

                # ============================================================
                # VISUALISASI DISTRIBUSI — 3 panel: Harga Penutupan | Log Return | Q-Q Plot
                # ============================================================
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("#### Visualisasi Distribusi")

                st.markdown(
                    '<div class="box-info">'
                    'ℹ️ <strong>Panduan membaca visualisasi:</strong><br>'
                    '• <strong>Kiri</strong>: Histogram <em>harga penutupan</em> — menunjukkan distribusi harga saham mentah.<br>'
                    '• <strong>Tengah</strong>: Histogram <em>log return</em> [ln(Pₜ/Pₜ₋₁)] — ini yang diuji normalitasnya, '
                    'karena model VaR berbasis pada distribusi return, bukan harga.<br>'
                    '• <strong>Kanan</strong>: Q-Q Plot log return vs distribusi normal — '
                    'semakin dekat titik ke garis, semakin normal distribusinya.'
                    '</div>',
                    unsafe_allow_html=True
                )

                fig_dist, axes_dist = plt.subplots(1, 3, figsize=(18, 5))
                fig_dist.patch.set_alpha(0.0)

                # pakai hasil keputusan utama dari Jarque-Bera
                # normal_flag = bool murni, aman dipakai di if/else
                normal_flag = (is_normal_overall is True)
                hist_color = '#3b82f6' if normal_flag else '#f87171'

                # ---- Panel 1: Histogram Harga Penutupan ----
                close_vals = df[close_col].dropna().values
                ax0 = axes_dist[0]
                ax0.set_facecolor('none')
                ax0.patch.set_alpha(0.0)
                ax0.hist(close_vals, bins=50, density=True, alpha=0.7,
                         color='#22c55e', edgecolor=(0, 0, 0, 0.15),
                         label='Harga Penutupan')
                close_mean = close_vals.mean()
                close_std  = close_vals.std()
                x_close = np.linspace(close_vals.min(), close_vals.max(), 200)
                ax0.plot(x_close, stats.norm.pdf(x_close, close_mean, close_std),
                         color='#f59e0b', linewidth=2, linestyle='--', label='Kurva Normal (ref.)')
                ax0.set_xlabel(f'Harga Penutupan ({currency_symbol})', fontsize=10)
                ax0.set_ylabel('Density', fontsize=10)
                ax0.set_title('Histogram Harga Penutupan', fontsize=12, fontweight='bold')
                ax0.tick_params(labelsize=9)
                for sp in ax0.spines.values():
                    sp.set_color('gray')
                    sp.set_alpha(0.4)
                ax0.legend(fontsize=9)
                ax0.grid(True, alpha=0.2)

                # ---- Panel 2: Histogram Log Return ----
                ax1 = axes_dist[1]
                ax1.set_facecolor('none')
                ax1.patch.set_alpha(0.0)
                ax1.hist(clean_returns, bins=50, density=True, alpha=0.7,
                         color=hist_color, edgecolor=(0, 0, 0, 0.15), label='Log Return')
                x_range = np.linspace(clean_returns.min(), clean_returns.max(), 200)
                ax1.plot(x_range, stats.norm.pdf(x_range, mean_ret, std_ret),
                         color='#f59e0b', linewidth=2, linestyle='--',
                         label='Kurva Normal' + (' (fit baik)' if normal_flag else ' (referensi)'))
                ax1.set_xlabel('Log Return  [ln(Pₜ/Pₜ₋₁)]', fontsize=10)
                ax1.set_ylabel('Density', fontsize=10)
                ax1.set_title('Histogram Distribusi Log Return', fontsize=12, fontweight='bold')
                ax1.tick_params(labelsize=9)
                for sp in ax1.spines.values():
                    sp.set_color('gray')
                    sp.set_alpha(0.4)
                ax1.legend(fontsize=9)
                ax1.grid(True, alpha=0.2)

                # ---- Panel 3: Q-Q Plot Log Return ----
                ax2 = axes_dist[2]
                ax2.set_facecolor('none')
                ax2.patch.set_alpha(0.0)
                (osm, osr), (slope, intercept, r) = stats.probplot(clean_returns, dist="norm")
                ax2.scatter(osm, osr, s=8, alpha=0.5,
                            color='#6366f1' if normal_flag else '#f87171', label='Log Return')
                line_x = np.array([min(osm), max(osm)])
                ax2.plot(line_x, slope * line_x + intercept,
                         color='#f59e0b', linewidth=1.8, linestyle='--', label='Garis Normal')
                ax2.set_xlabel('Theoretical Quantiles', fontsize=10)
                ax2.set_ylabel('Sample Quantiles (Log Return)', fontsize=10)
                ax2.set_title('Q-Q Plot: Log Return vs Normal', fontsize=12, fontweight='bold')
                ax2.tick_params(labelsize=9)
                for sp in ax2.spines.values():
                    sp.set_color('gray')
                    sp.set_alpha(0.4)
                ax2.legend(fontsize=9)
                ax2.grid(True, alpha=0.2)

                if is_normal_overall is False:
                    ax2.text(0.05, 0.95,
                             'Penyimpangan dari garis\nmengindikasikan non-normalitas',
                             transform=ax2.transAxes, fontsize=8.5,
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='#fef9c3',
                                       edgecolor='#f59e0b', alpha=0.8))

                plt.tight_layout()
                st.pyplot(fig_dist, transparent=True)
                plt.close()

                st.markdown(
                    '<p style="color:#475569; font-size:0.8rem; margin-bottom:0.2rem; font-family:monospace;">— Interpretasi Visualisasi —</p>',
                    unsafe_allow_html=True
                )
                if is_normal_overall is True:
                    st.code(
                        "Histogram : data mengikuti kurva normal dengan baik\n"
                        "Q-Q Plot  : titik-titik data dekat dengan garis referensi\n"
                        "Kesimpulan: distribusi return mendekati normal",
                        language=None
                    )
                elif is_normal_overall is False:
                    st.code(
                        "Histogram : terdapat penyimpangan dari kurva normal (ekor tebal / asimetri)\n"
                        "Q-Q Plot  : titik-titik menyimpang dari garis, terutama di ekor distribusi\n"
                        "Kesimpulan: distribusi return tidak normal — pendekatan Cornish-Fisher relevan\n"
                        "            dipertimbangkan, namun penilaian akhir metode tetap dilihat dari backtesting",
                        language=None
                    )
                else:
                    st.code(
                        "Kesimpulan visual belum dapat ditentukan otomatis karena hasil uji utama tidak tersedia.",
                        language=None
                    )

                # ============================================================
                # ANALISIS SKEWNESS & KURTOSIS
                # ============================================================
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("#### Analisis Skewness & Kurtosis")

                if abs(skew_ret) < 0.5:
                    skew_label = "Hampir Simetris"
                    skew_box   = "box-success"
                    skew_note  = "Distribusi mendekati simetris — tidak ada kemencengan signifikan."
                elif skew_ret >= 0.5:
                    skew_label = "Condong Kanan (Right-skewed)"
                    skew_box   = "box-warning"
                    skew_note  = "Ekor distribusi lebih panjang di sisi kanan — lebih banyak nilai ekstrem positif."
                else:
                    skew_label = "Condong Kiri (Left-skewed)"
                    skew_box   = "box-warning"
                    skew_note  = "Ekor distribusi lebih panjang di sisi kiri — lebih banyak nilai ekstrem negatif."

                st.markdown(
                    f'<div class="{skew_box}"><strong>Skewness = {skew_ret:.4f} — {skew_label}</strong><br>'
                    f'{skew_note}</div>',
                    unsafe_allow_html=True
                )

                if kurt_ret <= 3:
                    kurt_label = "Platykurtic (K ≤ 3) — ekor lebih tipis dari normal"
                    kurt_box   = "box-info"
                elif kurt_ret <= 5:
                    kurt_label = "Leptokurtic ringan (3 < K ≤ 5) — ekor sedikit lebih tebal dari normal"
                    kurt_box   = "box-warning"
                else:
                    kurt_label = "Leptokurtic kuat (K > 5) — fat-tail signifikan"
                    kurt_box   = "box-error"

                st.markdown(
                    f'<div class="{kurt_box}"><strong>Kurtosis = {kurt_ret:.4f} — {kurt_label}</strong><br>'
                    f'Excess Kurtosis = {kurt_ret_excess:.4f} '
                    f'(nilai positif → ekor lebih tebal, risiko ekstrem lebih tinggi dari asumsi normal)</div>',
                    unsafe_allow_html=True
                )

                # ============================================================
                # ANALISIS OUTLIER (IQR)
                # ============================================================
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("#### Analisis Outlier (IQR)")

                q1  = clean_returns.quantile(0.25)
                q2  = clean_returns.median()
                q3  = clean_returns.quantile(0.75)
                iqr = q3 - q1
                n_out   = len(clean_returns[(clean_returns < q1 - 1.5*iqr) |
                                             (clean_returns > q3 + 1.5*iqr)])
                out_pct = (n_out / n) * 100

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Q1 (25%)",    f"{q1:.6f}")
                c2.metric("Median (Q2)", f"{q2:.6f}")
                c3.metric("Q3 (75%)",    f"{q3:.6f}")
                c4.metric("IQR",         f"{iqr:.6f}")

                o1, o2, o3 = st.columns(3)
                o1.metric("Jumlah Outliers",     f"{n_out}")
                o2.metric("Total Observasi",     f"{n}")
                o3.metric("Persentase Outliers", f"{out_pct:.2f}%")

                if out_pct < 5:
                    st.markdown(
                        f'<div class="box-success">✅ Outliers rendah ({out_pct:.2f}%) — data relatif konsisten</div>',
                        unsafe_allow_html=True
                    )
                elif out_pct < 10:
                    st.markdown(
                        f'<div class="box-warning">⚠️ Outliers sedang ({out_pct:.2f}%) — perhatikan nilai-nilai ekstrem</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="box-error">❌ Outliers tinggi ({out_pct:.2f}%) — data memiliki banyak nilai ekstrem</div>',
                        unsafe_allow_html=True
                    )

                # ---- Box Plot Return ----
                st.markdown("##### Box Plot Log Return")

                fig_box, ax_box = plt.subplots(figsize=(10, 3))
                fig_box.patch.set_alpha(0.0)
                ax_box.set_facecolor('none')
                ax_box.patch.set_alpha(0.0)

                bp = ax_box.boxplot(
                    clean_returns.values,
                    vert=False,
                    patch_artist=True,
                    widths=0.5,
                    boxprops=dict(facecolor='#6366f1', alpha=0.6, linewidth=1.2),
                    medianprops=dict(color='#f59e0b', linewidth=2),
                    whiskerprops=dict(color='gray', linewidth=1.2, linestyle='--'),
                    capprops=dict(color='gray', linewidth=1.5),
                    flierprops=dict(
                        marker='o', markerfacecolor='#ef4444',
                        markeredgecolor='#ef4444', markersize=4, alpha=0.6
                    )
                )

                # Tambahkan garis mean
                ax_box.axvline(clean_returns.mean(), color='#22c55e', linewidth=1.5,
                               linestyle=':', label=f'Mean = {clean_returns.mean():.5f}')

                # Label batas IQR
                ax_box.axvline(q1 - 1.5*iqr, color='#ef4444', linewidth=1,
                               linestyle=':', alpha=0.5, label=f'Batas bawah = {q1 - 1.5*iqr:.5f}')
                ax_box.axvline(q3 + 1.5*iqr, color='#ef4444', linewidth=1,
                               linestyle=':', alpha=0.5, label=f'Batas atas = {q3 + 1.5*iqr:.5f}')

                ax_box.set_xlabel('Log Return  [ln(Pₜ/Pₜ₋₁)]', fontsize=10)
                ax_box.set_yticks([])
                ax_box.set_title(
                    f'Box Plot Log Return  —  {n_out} outlier dari {n} observasi ({out_pct:.2f}%)',
                    fontsize=11, fontweight='bold'
                )
                ax_box.tick_params(labelsize=9)
                for sp in ax_box.spines.values():
                    sp.set_color('gray'); sp.set_alpha(0.4)
                ax_box.legend(fontsize=8.5, loc='upper right')
                ax_box.grid(True, alpha=0.2, axis='x')

                plt.tight_layout()
                st.pyplot(fig_box, transparent=True)
                plt.close()

                st.markdown(
                    '<p class="text-hint">— Cara membaca box plot —</p>',
                    unsafe_allow_html=True
                )
                st.code(
                    f"Kotak (box)  : rentang interkuartil Q1–Q3  →  [{q1:.5f}, {q3:.5f}]\n"
                    f"Garis kuning : median  =  {q2:.5f}\n"
                    f"Garis hijau  : mean    =  {clean_returns.mean():.5f}\n"
                    f"Whisker      : rentang normal  →  [Q1 − 1.5×IQR, Q3 + 1.5×IQR]\n"
                    f"             = [{q1 - 1.5*iqr:.5f}, {q3 + 1.5*iqr:.5f}]\n"
                    f"Titik merah  : outlier di luar whisker  →  {n_out} observasi ({out_pct:.2f}%)",
                    language=None
                )

                # Perbandingan Mean vs Median
                if abs(skew_ret) > 0.1:
                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                    st.markdown("#### Perbandingan Mean vs Median")
                    m1, m2 = st.columns(2)
                    m1.metric("Mean",   f"{mean_ret:.6f}")
                    m2.metric("Median", f"{q2:.6f}")
                    if mean_ret > q2:
                        st.markdown(
                            '<div class="box-info">Mean &gt; Median — data condong ke kanan (right-skewed)</div>',
                            unsafe_allow_html=True
                        )
                    elif mean_ret < q2:
                        st.markdown(
                            '<div class="box-info">Mean &lt; Median — data condong ke kiri (left-skewed)</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="box-success">Mean ≈ Median — distribusi simetris</div>',
                            unsafe_allow_html=True
                        )

                # ---- Simpan ke session state ----
                st.session_state['uploaded_data']          = df
                st.session_state['log_returns']            = clean_returns
                st.session_state['is_normal']              = is_normal_overall   # dari JB
                st.session_state['market_type']            = market_type
                st.session_state['normality_primary_test'] = 'Jarque-Bera'
                st.session_state['distribution_tests']     = {
                    'Jarque-Bera': {
                        'statistic': jb_stat,
                        'p_value'  : jb_pval,
                        'is_normal': (jb_pval > alpha_test) if jb_pval is not None else None
                    }
                }

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                # PATCH 2: pesan akhir diperbarui
                st.markdown(
                    '<div class="box-success">✅ Analisis selesai. Data siap untuk perhitungan VaR di Tab 4 dan Tab 5.</div>',
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.markdown(
                f'<div class="box-error">❌ Error memproses data: {str(e)}</div>',
                unsafe_allow_html=True
            )
            import traceback
            st.code(traceback.format_exc())
with tab4:
    st.header("VaR — Variance-Covariance")

    if 'log_returns' not in st.session_state:
        st.markdown("""
        <div class="box-warning">
            ⚠️ Data belum tersedia. Silakan upload data dan lakukan uji normalitas di
            <strong>Tab 3 (Upload &amp; Uji Normalitas)</strong> terlebih dahulu.
        </div>
        """, unsafe_allow_html=True)
    else:
        log_returns = st.session_state['log_returns']
        is_normal   = st.session_state.get('is_normal', None)  # None = belum diketahui

        market_t3 = (
            st.session_state.get('market_type')
            or st.session_state.get('market_type_tab3')
            or '🇮🇩 Saham Indonesia (IDX)'
        )
        currency_symbol = "Rp" if "IDX" in market_t3 else "$"

        st.markdown("""
        <div class="box-info">
            <strong>ℹ️ Metode Variance-Covariance (Parametrik Normal)</strong><br>
            Mengasumsikan log return berdistribusi normal. Pada tab ini, VaR dan ES dihitung
            secara analitik menggunakan standar deviasi (σ), z-score tingkat kepercayaan,
            dan akar holding period. Rata-rata return (μ) tetap ditampilkan sebagai informasi
            statistik, tetapi dalam perhitungan utama diasumsikan relatif kecil sehingga diabaikan.
        </div>
        """, unsafe_allow_html=True)

        if is_normal is False:
            st.markdown("""
            <div class="box-warning">
                ⚠️ Data tidak berdistribusi normal berdasarkan uji <strong>Jarque–Bera</strong>.
                Variance-Covariance mungkin <strong>underestimate</strong> atau
                <strong>overestimate</strong> risiko — bandingkan dengan Cornish-Fisher di Tab 5.
            </div>
            """, unsafe_allow_html=True)
        elif is_normal is None:
            st.markdown("""
            <div class="box-info">
                ℹ️ Status normalitas data belum tersedia. Pastikan uji normalitas di
                Tab 3 sudah dijalankan agar peringatan ini dapat ditampilkan dengan akurat.
            </div>
            """, unsafe_allow_html=True)

        # ============================================================
        # RUMUS REFERENSI — FIX 2: dua tingkat (return & moneter)
        # ============================================================
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Rumus yang Digunakan")

        st.markdown(
            '<p style="color:#475569; font-size:0.8rem; margin-bottom:0.2rem; font-family:monospace;">'
            '— Dalam satuan return (batas ekor kiri distribusi) —</p>',
            unsafe_allow_html=True
        )
        st.code(
            "VaR_t = −σ × Z(1−α) × √t\n"
            "ES_t  = −σ × φ(Z(1−α)) / α × √t",
            language=None
        )

        st.markdown(
            '<p style="color:#475569; font-size:0.8rem; margin-bottom:0.2rem; font-family:monospace;">'
            '— Dalam satuan moneter (besar kerugian, positif) —</p>',
            unsafe_allow_html=True
        )
        st.code(
            "VaR_moneter = V₀ × σ × Z(1−α) × √t\n"
            "ES_moneter  = V₀ × σ × φ(Z(1−α)) / α × √t",
            language=None
        )

        st.markdown(
            '<p style="color:#94a3b8; font-size:0.82rem; margin-top:0.3rem;">'
            'φ(Z<sub>1−α</sub>) = PDF normal standar pada titik Z<sub>1−α</sub> &nbsp;|&nbsp; '
            'μ diasumsikan ≈ 0 dan diabaikan dalam rumus di atas'
            '</p>',
            unsafe_allow_html=True
        )

        # ============================================================
        # PARAMETER INPUT
        # ============================================================
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Parameter Perhitungan")

        col1, col2 = st.columns(2)
        with col1:
            v0 = st.number_input(
                f"Nilai Investasi Awal (V₀) [{currency_symbol}]",
                min_value=1,
                max_value=9_007_199_254_740_991,
                value=100_000_000,
                step=1_000_000,
                format="%d"
            )
            st.markdown(
                f'<p style="color:#64748b; font-size:0.82rem;">V₀ = {currency_symbol} {v0:,.0f}</p>',
                unsafe_allow_html=True
            )

        with col2:
            input_mode_t4 = st.radio(
                "Mode Input Parameter",
                ["Preset Standar", "Input Manual"],
                horizontal=True,
                key="input_mode_tab4"
            )

        if input_mode_t4 == "Preset Standar":
            col3, col4 = st.columns(2)
            with col3:
                conf_preset = st.selectbox(
                    "Confidence Level",
                    options=[99, 95, 90],
                    format_func=lambda x: f"{x}%  (α = {100 - x}%)",
                    index=1,
                    key="conf_tab4"
                )
                alpha_sel = (100 - conf_preset) / 100
                conf_sel  = conf_preset
            with col4:
                hp_option = st.selectbox(
                    "Holding Period",
                    ["1 hari", "5 hari (1 minggu kerja)",
                     "20 hari (1 bulan kerja)", "252 hari (1 tahun kerja)"],
                    index=0,
                    key="hp_tab4"
                )
                hp_map   = {"1 hari": 1, "5 hari (1 minggu kerja)": 5,
                             "20 hari (1 bulan kerja)": 20, "252 hari (1 tahun kerja)": 252}
                hp_sel   = hp_map[hp_option]
                hp_label = hp_option
        else:
            col3, col4 = st.columns(2)
            with col3:
                conf_sel = st.slider(
                    "Confidence Level (%)", min_value=80, max_value=99,
                    value=95, step=1, key="conf_slider_tab4"
                )
                alpha_sel = (100 - conf_sel) / 100
                st.markdown(
                    f'<p style="color:#64748b; font-size:0.82rem;">α = {alpha_sel:.4f}</p>',
                    unsafe_allow_html=True
                )
            with col4:
                hp_sel = st.number_input(
                    "Holding Period (hari)", min_value=1, max_value=504,
                    value=1, step=1, key="hp_input_tab4"
                )
                hp_label = f"{hp_sel} hari"

        st.markdown(
            f'<div class="box-info">'
            f'<strong>Parameter Terpilih</strong><br>'
            f'Confidence Level: <strong>{conf_sel}%</strong> (α = {alpha_sel:.4f}) &nbsp;|&nbsp; '
            f'Holding Period: <strong>{hp_label}</strong> &nbsp;|&nbsp; '
            f'V₀ = {currency_symbol} {v0:,.0f}'
            f'</div>',
            unsafe_allow_html=True
        )

        # ============================================================
        # TOMBOL HITUNG
        # ============================================================
        if st.button("Hitung VaR & ES — Variance-Covariance",
                     type="primary", use_container_width=True, key="calc_vc_tab4"):

            clean_lr   = pd.Series(log_returns).replace([np.inf, -np.inf], np.nan).dropna()
            mu         = clean_lr.mean()
            sigma      = clean_lr.std(ddof=1)

            if len(clean_lr) < 2 or pd.isna(sigma) or sigma <= 0:
                st.markdown(
                    '<div class="box-error">❌ Standar deviasi tidak valid. '
                    'Pastikan data log return memiliki variasi dan jumlah observasi mencukupi.</div>',
                    unsafe_allow_html=True
                )
                st.stop()

            z_score    = stats.norm.ppf(1 - alpha_sel)
            phi_z      = stats.norm.pdf(z_score)
            var_pct    = sigma * z_score * np.sqrt(hp_sel)   # besar kerugian, positif
            es_pct     = sigma * (phi_z / alpha_sel) * np.sqrt(hp_sel)
            var_amount = v0 * var_pct
            es_amount  = v0 * es_pct

            st.markdown(
                '<div class="box-success">✅ Perhitungan Variance-Covariance selesai.</div>',
                unsafe_allow_html=True
            )

            # ============================================================
            # PARAMETER STATISTIK
            # ============================================================
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### Parameter Statistik Data")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Mean (μ)",    f"{mu:.7f}")
            s2.metric("Std Dev (σ)", f"{sigma:.7f}")
            s3.metric("Z-score",     f"{z_score:.6f}")
            s4.metric("φ(Z)",        f"{phi_z:.6f}")

            st.markdown(
                '<p style="color:#475569; font-size:0.8rem; margin-top:0.6rem; margin-bottom:0.15rem; font-family:monospace;">μ dan σ</p>',
                unsafe_allow_html=True
            )
            # FIX 1 lanjutan: keterangan μ ditambahkan "diabaikan dalam rumus utama"
            st.code(
                f"μ  = {mu:.8f}\n"
                f"   rata-rata log return harian dari data historis\n"
                f"   ditampilkan sebagai statistik; dalam rumus utama diasumsikan ≈ 0 dan diabaikan",
                language=None
            )
            st.code(
                f"σ  = {sigma:.8f}\n"
                f"   ukuran volatilitas return — semakin besar σ, semakin berisiko",
                language=None
            )

            st.markdown(
                '<p style="color:#475569; font-size:0.8rem; margin-top:0.6rem; margin-bottom:0.15rem; font-family:monospace;">Z-score dan φ(Z)</p>',
                unsafe_allow_html=True
            )
            st.code(
                f"Z(1−α) = Z({1-alpha_sel:.2f}) = {z_score:.6f}\n"
                f"   kuantil distribusi normal standar pada tingkat kepercayaan {conf_sel}%\n"
                f"   nilai ini menetapkan seberapa jauh ekor kiri distribusi yang diperhitungkan",
                language=None
            )
            st.code(
                f"φ(Z(1−α)) = {phi_z:.6f}\n"
                f"   nilai PDF normal standar pada titik Z — digunakan sebagai faktor skala ES",
                language=None
            )

            st.markdown(
                '<p style="color:#475569; font-size:0.8rem; margin-top:0.6rem; margin-bottom:0.15rem; font-family:monospace;">α dan t</p>',
                unsafe_allow_html=True
            )
            st.code(
                f"α  = {alpha_sel:.4f}\n"
                f"   tingkat signifikansi — probabilitas kerugian melebihi VaR ({int(alpha_sel*100)}%)",
                language=None
            )
            st.code(
                f"t  = {hp_sel} hari  (holding period)\n"
                f"   jangka waktu kepemilikan; VaR dan ES diskala dengan faktor √t",
                language=None
            )

            # ============================================================
            # HASIL VaR & ES
            # ============================================================
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### Hasil Perhitungan")

            r1, r2 = st.columns(2)
            with r1:
                st.markdown(
                    '<p style="color:#475569; font-size:0.8rem; margin-bottom:0.2rem; font-family:monospace;">— Value at Risk —</p>',
                    unsafe_allow_html=True
                )
                st.code(
                    f"VaR_moneter = V₀ × σ × Z(1−α) × √t\n"
                    f"           = {v0:,.0f} × {sigma:.6f} × {z_score:.6f} × √{hp_sel}\n"
                    f"           = {currency_symbol} {var_amount:,.2f}",
                    language=None
                )
                va1, va2 = st.columns(2)
                # FIX 3: label metric diubah agar tidak ambigu
                va1.metric(f"VaR {conf_sel}% (besar kerugian, absolut)", f"{var_pct:.6f}")
                va2.metric(f"VaR {conf_sel}% ({currency_symbol})", f"{currency_symbol} {var_amount:,.0f}")

            with r2:
                st.markdown(
                    '<p style="color:#475569; font-size:0.8rem; margin-bottom:0.2rem; font-family:monospace;">— Expected Shortfall —</p>',
                    unsafe_allow_html=True
                )
                st.code(
                    f"ES_moneter = V₀ × σ × φ(Z(1−α)) / α × √t\n"
                    f"          = {v0:,.0f} × {sigma:.6f} × {phi_z:.6f} / {alpha_sel:.4f} × √{hp_sel}\n"
                    f"          = {currency_symbol} {es_amount:,.2f}",
                    language=None
                )
                ea1, ea2 = st.columns(2)
                # FIX 3: label metric diubah agar tidak ambigu
                ea1.metric(f"ES {conf_sel}% (besar kerugian, absolut)", f"{es_pct:.6f}")
                ea2.metric(f"ES {conf_sel}% ({currency_symbol})", f"{currency_symbol} {es_amount:,.0f}")

            # Tabel ringkasan
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### Tabel Ringkasan Hasil")

            df_res = pd.DataFrame([
                {"Parameter": "Metode",                                      "Nilai": "Variance-Covariance (Normal)"},
                {"Parameter": "Confidence Level",                            "Nilai": f"{conf_sel}%"},
                {"Parameter": "Alpha (α)",                                   "Nilai": f"{alpha_sel:.4f}"},
                {"Parameter": "Holding Period",                              "Nilai": hp_label},
                {"Parameter": "Nilai Investasi (V₀)",                        "Nilai": f"{currency_symbol} {v0:,.0f}"},
                {"Parameter": "Mean (μ) — informasi statistik",              "Nilai": f"{mu:.8f}"},
                {"Parameter": "Std Dev (σ)",                                 "Nilai": f"{sigma:.8f}"},
                {"Parameter": "Z-score Z(1−α)",                              "Nilai": f"{z_score:.6f}"},
                {"Parameter": "φ(Z(1−α))",                                  "Nilai": f"{phi_z:.6f}"},
                # FIX 3: label tabel diubah
                {"Parameter": f"VaR {conf_sel}% (besar kerugian, absolut)", "Nilai": f"{var_pct:.8f}"},
                {"Parameter": f"VaR {conf_sel}% ({currency_symbol})",       "Nilai": f"{currency_symbol} {var_amount:,.2f}"},
                {"Parameter": f"VaR {conf_sel}% (% investasi)",             "Nilai": f"{(var_amount/v0)*100:.4f}%"},
                {"Parameter": f"ES {conf_sel}% (besar kerugian, absolut)",  "Nilai": f"{es_pct:.8f}"},
                {"Parameter": f"ES {conf_sel}% ({currency_symbol})",        "Nilai": f"{currency_symbol} {es_amount:,.2f}"},
                {"Parameter": f"ES {conf_sel}% (% investasi)",              "Nilai": f"{(es_amount/v0)*100:.4f}%"},
            ])
            st.dataframe(df_res, use_container_width=True, hide_index=True)

            # ============================================================
            # VISUALISASI
            # ============================================================
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### Visualisasi")

            skew_data = stats.skew(clean_lr, bias=False)
            kurt_data = stats.kurtosis(clean_lr, fisher=True, bias=False) + 3

            fig_vc, axes_vc = plt.subplots(1, 2, figsize=(14, 5))
            fig_vc.patch.set_alpha(0.0)

            ax1 = axes_vc[0]
            ax1.set_facecolor('none')
            ax1.patch.set_alpha(0.0)
            ax1.hist(clean_lr, bins=50, density=True, alpha=0.65,
                     color='#3b82f6', edgecolor=(0,0,0,0.1), label='Log Return Historis')
            x_range = np.linspace(clean_lr.min(), clean_lr.max(), 300)
            ax1.plot(x_range, stats.norm.pdf(x_range, mu, sigma),
                     color='#f59e0b', linewidth=2, linestyle='--', label='Kurva Normal')
            ax1.axvline(-var_pct, color='#ef4444', linewidth=2, linestyle='--',
                        label=f'VaR {conf_sel}% batas: {-var_pct:.5f}')
            ax1.axvline(-es_pct, color='#f97316', linewidth=2, linestyle=':',
                        label=f'ES {conf_sel}% batas: {-es_pct:.5f}')
            ax1.set_xlabel('Log Return', fontsize=10)
            ax1.set_ylabel('Density',    fontsize=10)
            ax1.set_title('Distribusi Return & Batas VaR/ES', fontsize=12, fontweight='bold')
            ax1.tick_params(labelsize=9)
            for sp in ax1.spines.values():
                sp.set_color('gray'); sp.set_alpha(0.4)
            ax1.legend(fontsize=8.5)
            ax1.grid(True, alpha=0.2)

            ax2 = axes_vc[1]
            ax2.set_facecolor('none')
            ax2.patch.set_alpha(0.0)
            x_cdf = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
            y_cdf = stats.norm.cdf(x_cdf, mu, sigma)
            ax2.plot(x_cdf, y_cdf, color='#6366f1', linewidth=2, label='CDF Normal')
            ax2.axvline(-var_pct, color='#ef4444', linewidth=2, linestyle='--',
                        label=f'VaR {conf_sel}%')
            ax2.axhline(alpha_sel, color='#ef4444', linewidth=1, linestyle=':', alpha=0.6,
                        label=f'α = {alpha_sel:.4f}')
            x_tail = np.linspace(mu - 4*sigma, -var_pct, 200)
            y_tail = stats.norm.cdf(x_tail, mu, sigma)
            ax2.fill_between(x_tail, 0, y_tail, alpha=0.25, color='#ef4444',
                             label=f'Tail ({int(alpha_sel*100)}%)')
            ax2.set_xlabel('Log Return',             fontsize=10)
            ax2.set_ylabel('Probabilitas Kumulatif', fontsize=10)
            ax2.set_title('CDF Normal & Area Tail',  fontsize=12, fontweight='bold')
            ax2.tick_params(labelsize=9)
            for sp in ax2.spines.values():
                sp.set_color('gray'); sp.set_alpha(0.4)
            ax2.legend(fontsize=8.5)
            ax2.grid(True, alpha=0.2)

            plt.tight_layout()
            st.pyplot(fig_vc, transparent=True)
            plt.close()

            # Interpretasi dinamis
            if abs(skew_data) < 0.5:
                shape_desc = "terkonsentrasi di sekitar nol dan relatif simetris"
            elif skew_data > 0:
                shape_desc = "sedikit miring ke kanan dengan lebih banyak nilai ekstrem positif"
            else:
                shape_desc = "sedikit miring ke kiri dengan lebih banyak nilai ekstrem negatif"

            if kurt_data > 5:
                tail_desc = "Ekor distribusi tebal (fat-tail) — tail risk lebih tinggi dari asumsi normal."
            elif kurt_data > 3:
                tail_desc = "Ekor distribusi sedikit lebih tebal dari normal — risiko ekstrem moderat."
            else:
                tail_desc = "Ekor distribusi lebih tipis dari normal — asumsi normalitas relatif terpenuhi."

            var_pos = "jauh ke sisi kiri" if var_pct > 2 * sigma else "tidak terlalu jauh dari pusat"

            st.markdown(
                '<p style="color:#475569; font-size:0.8rem; margin-top:0.5rem; margin-bottom:0.15rem; font-family:monospace;">— Interpretasi: Histogram Distribusi Return (kiri) —</p>',
                unsafe_allow_html=True
            )
            st.code(
                f"Distribusi log return historis {shape_desc}.\n"
                f"{tail_desc}\n"
                f"Garis VaR {conf_sel}% (batas return = {-var_pct:.5f}) berada {var_pos} di sisi kiri,\n"
                f"menandai batas kerugian return pada tingkat kepercayaan {conf_sel}%.\n"
                f"Kerugian moneter setara {currency_symbol} {var_amount:,.0f} dari nilai investasi.\n"
                f"Garis ES {conf_sel}% (batas return = {-es_pct:.5f}) menggambarkan\n"
                f"rata-rata kerugian ketika VaR telah terlampaui.",
                language=None
            )

            st.markdown(
                '<p style="color:#475569; font-size:0.8rem; margin-top:0.5rem; margin-bottom:0.15rem; font-family:monospace;">— Interpretasi: CDF Normal & Area Tail (kanan) —</p>',
                unsafe_allow_html=True
            )
            st.code(
                f"Kurva CDF menggambarkan probabilitas kumulatif return.\n"
                f"Area tail {int(alpha_sel*100)}% (diarsir merah) di sisi kiri merepresentasikan\n"
                f"peluang terjadinya kerugian ekstrem pada α = {alpha_sel:.4f}.\n"
                f"Garis VaR {conf_sel}% menandai titik batas return minimum —\n"
                f"kerugian di luar titik ini termasuk wilayah risiko ekstrem.",
                language=None
            )

            # ============================================================
            # INTERPRETASI HASIL & TINGKAT RISIKO
            # ============================================================
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### Interpretasi Hasil")

            var_pct_inv = (var_amount / v0) * 100
            es_pct_inv  = (es_amount  / v0) * 100

            if var_pct_inv < 5:
                risk_label, risk_box = "RENDAH",        "box-success"
            elif var_pct_inv < 10:
                risk_label, risk_box = "SEDANG",        "box-warning"
            elif var_pct_inv < 20:
                risk_label, risk_box = "TINGGI",        "box-error"
            else:
                risk_label, risk_box = "SANGAT TINGGI", "box-error"

            st.markdown(
                f'<div class="{risk_box}">'
                f'<strong>Tingkat Risiko: {risk_label}</strong><br>'
                f'VaR {conf_sel}% ({hp_label}): '
                f'<strong>{currency_symbol} {var_amount:,.0f}</strong> '
                f'({var_pct_inv:.2f}% dari investasi)<br>'
                f'ES {conf_sel}% ({hp_label}): '
                f'<strong>{currency_symbol} {es_amount:,.0f}</strong> '
                f'({es_pct_inv:.2f}% dari investasi)'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<p style="color:#475569; font-size:0.8rem; margin-bottom:0.15rem; font-family:monospace;">— Interpretasi VaR —</p>',
                unsafe_allow_html=True
            )
            st.code(
                f"Dengan tingkat kepercayaan {conf_sel}%, kerugian dalam {hp_label}\n"
                f"tidak akan melebihi {currency_symbol} {var_amount:,.0f} ({var_pct_inv:.2f}% dari investasi).\n"
                f"Terdapat {int(alpha_sel*100)}% kemungkinan kerugian melebihi batas ini.",
                language=None
            )

            st.markdown(
                '<p style="color:#475569; font-size:0.8rem; margin-bottom:0.15rem; font-family:monospace;">— Interpretasi ES —</p>',
                unsafe_allow_html=True
            )
            st.code(
                f"Jika kerugian melampaui VaR (pada {int(alpha_sel*100)}% kasus terburuk),\n"
                f"rata-rata kerugian adalah {currency_symbol} {es_amount:,.0f} ({es_pct_inv:.2f}% dari investasi).\n"
                f"ES selalu ≥ VaR — ES lebih konservatif dan informatif untuk tail risk.",
                language=None
            )

            if is_normal is False:
                st.markdown(
                    '<div class="box-warning">⚠️ Data tidak normal berdasarkan uji Jarque–Bera — '
                    'hasil VaR ini mungkin kurang akurat. '
                    'Bandingkan dengan Cornish-Fisher di Tab 5.</div>',
                    unsafe_allow_html=True
                )

            # ============================================================
            # SIMPAN & DOWNLOAD
            # ============================================================
            st.session_state['var_vc_results'] = {
                'method'              : 'Variance-Covariance',
                'confidence_level'    : conf_sel,
                'alpha'               : alpha_sel,
                'holding_period'      : hp_sel,
                'holding_period_label': hp_label,
                'v0'                  : v0,
                'currency'            : currency_symbol,
                'mu'                  : mu,
                'sigma'               : sigma,
                'z_score'             : z_score,
                'phi_z'               : phi_z,
                'var_pct'             : var_pct,
                'var_amount'          : var_amount,
                'es_pct'              : es_pct,
                'es_amount'           : es_amount,
                'risk_level'          : risk_label,
                'is_normal'           : is_normal,
            }

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            output_vc = io.BytesIO()
            with pd.ExcelWriter(output_vc, engine='openpyxl') as writer:
                df_res.to_excel(writer, index=False, sheet_name='Hasil VaR VC')
                pd.DataFrame([{
                    'mu': mu, 'sigma': sigma, 'z_score': z_score, 'phi_z': phi_z,
                    'alpha': alpha_sel, 'holding_period': hp_sel,
                    'v0': v0, 'currency': currency_symbol,
                    'var_pct': var_pct, 'var_amount': var_amount,
                    'es_pct': es_pct, 'es_amount': es_amount
                }]).to_excel(writer, index=False, sheet_name='Parameter')

            st.download_button(
                label="💾 Download Hasil VaR Variance-Covariance",
                data=output_vc.getvalue(),
                file_name=f"var_vc_{conf_sel}pct_{hp_sel}days.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

with tab5:
    st.header("VaR — Cornish-Fisher")

    if 'log_returns' not in st.session_state:
        st.markdown("""
        <div class="box-warning">
            ⚠️ Data belum tersedia. Silakan upload data dan lakukan uji normalitas di
            <strong>Tab 3 (Upload &amp; Uji Normalitas)</strong> terlebih dahulu.
        </div>
        """, unsafe_allow_html=True)
    else:
        log_returns = st.session_state['log_returns']
        is_normal   = st.session_state.get('is_normal', None)  # None = belum diketahui

        market_t3 = (
            st.session_state.get('market_type')
            or st.session_state.get('market_type_tab3')
            or '🇮🇩 Saham Indonesia (IDX)'
        )
        currency_symbol = "Rp" if "IDX" in market_t3 else "$"

        st.markdown("""
        <div class="box-info">
            <strong>ℹ️ Metode Cornish-Fisher</strong><br>
            Metode Cornish-Fisher digunakan untuk menyesuaikan kuantil normal dengan
            mempertimbangkan skewness dan excess kurtosis data return. Nilai VaR dihitung
            menggunakan kuantil Cornish-Fisher, sedangkan Expected Shortfall (ES) dihitung
            sebagai rata-rata kerugian pada observasi yang melampaui batas VaR Cornish-Fisher.
        </div>
        """, unsafe_allow_html=True)

        if is_normal is True:
            st.markdown("""
            <div class="box-info">
                ℹ️ Data berdistribusi normal berdasarkan uji <strong>Jarque–Bera</strong> — koreksi Cornish-Fisher
                akan minimal. Perbandingan dengan Variance-Covariance tersedia di Tab 6.
            </div>
            """, unsafe_allow_html=True)
        elif is_normal is None:
            st.markdown("""
            <div class="box-info">
                ℹ️ Status normalitas data belum tersedia. Pastikan uji normalitas di
                Tab 3 sudah dijalankan agar informasi ini dapat ditampilkan dengan akurat.
            </div>
            """, unsafe_allow_html=True)

        # ============================================================
        # RUMUS REFERENSI
        # ============================================================
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Rumus yang Digunakan")

        st.markdown(
            '<p style="color:#475569; font-size:0.8rem; margin-bottom:0.2rem; font-family:monospace;">'
            '1. Kuantil Cornish-Fisher  (K_e = excess kurtosis = stats.kurtosis(fisher=True))</p>',
            unsafe_allow_html=True
        )
        st.code(
            "Z_CF = Z_(1-α)\n"
            "     + (1/6)(Z_(1-α)^2 - 1) · S\n"
            "     + (1/24)(Z_(1-α)^3 - 3·Z_(1-α)) · K_e\n"
            "     - (1/36)(2·Z_(1-α)^3 - 5·Z_(1-α)) · S²\n\n"
            "Catatan: K_e adalah excess kurtosis.\n"
            "         stats.kurtosis(fisher=True) sudah menghasilkan excess kurtosis.",
            language=None
        )

        st.markdown(
            '<p style="color:#475569; font-size:0.8rem; margin-bottom:0.2rem; font-family:monospace;">'
            '2. VaR Cornish-Fisher</p>',
            unsafe_allow_html=True
        )
        st.code(
            "VaR_CF = V₀ × σ × Z_CF × √t\n\n"
            "Asumsi: Z_CF dihitung dari kuantil Z_(1-α) (kuantil kanan, bernilai positif),\n"
            "        sehingga VaR_CF dinyatakan sebagai besar kerugian positif.\n"
            "        Mean return diasumsikan ≈ 0 dan diabaikan.",
            language=None
        )

        st.markdown(
            '<p style="color:#475569; font-size:0.8rem; margin-bottom:0.2rem; font-family:monospace;">'
            '3. Expected Shortfall</p>',
            unsafe_allow_html=True
        )
        st.code(
            "ES_CF = (1/m) × Σ L_i,  untuk seluruh L_i > VaR_CF\n\n"
            "dengan:\n"
            "L_i = kerugian observasi ke-i dalam nilai uang (rupiah/dolar)\n"
            "m   = jumlah observasi tail yang melebihi VaR_CF\n\n"
            "Dalam implementasi ini, kerugian observasi dibentuk langsung\n"
            "dalam skala moneter yang konsisten dengan V₀ dan horizon t,\n"
            "sehingga ES nominal diperoleh langsung dari rata-rata tail loss\n"
            "tanpa pengalian ulang oleh V₀ maupun √t.",
            language=None
        )

        # ============================================================
        # PARAMETER INPUT
        # ============================================================
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("#### Parameter Perhitungan")

        col1, col2 = st.columns(2)
        with col1:
            v0_cf = st.number_input(
                f"Nilai Investasi Awal (V₀) [{currency_symbol}]",
                min_value=1,
                max_value=9_007_199_254_740_991,
                value=100_000_000,
                step=1_000_000,
                format="%d",
                key="v0_cf_tab5"
            )
            st.markdown(
                f'<p style="color:#64748b; font-size:0.82rem;">V₀ = {currency_symbol} {v0_cf:,.0f}</p>',
                unsafe_allow_html=True
            )

        with col2:
            input_mode_cf = st.radio(
                "Mode Input Parameter",
                ["Preset Standar", "Input Manual"],
                horizontal=True,
                key="input_mode_tab5"
            )

        if input_mode_cf == "Preset Standar":
            col3, col4 = st.columns(2)
            with col3:
                conf_preset_cf = st.selectbox(
                    "Confidence Level",
                    options=[99, 95, 90],
                    format_func=lambda x: f"{x}%  (α = {100 - x}%)",
                    index=1,
                    key="conf_tab5"
                )
                alpha_cf = (100 - conf_preset_cf) / 100
                conf_cf  = conf_preset_cf
            with col4:
                hp_option_cf = st.selectbox(
                    "Holding Period",
                    ["1 hari", "5 hari (1 minggu kerja)",
                     "20 hari (1 bulan kerja)", "252 hari (1 tahun kerja)"],
                    index=0,
                    key="hp_tab5"
                )
                hp_map_cf = {"1 hari": 1, "5 hari (1 minggu kerja)": 5,
                              "20 hari (1 bulan kerja)": 20, "252 hari (1 tahun kerja)": 252}
                hp_cf  = hp_map_cf[hp_option_cf]
                hp_lbl = hp_option_cf
        else:
            col3, col4 = st.columns(2)
            with col3:
                conf_cf = st.slider(
                    "Confidence Level (%)", min_value=80, max_value=99,
                    value=95, step=1, key="conf_slider_tab5"
                )
                alpha_cf = (100 - conf_cf) / 100
                st.markdown(
                    f'<p style="color:#64748b; font-size:0.82rem;">α = {alpha_cf:.4f}</p>',
                    unsafe_allow_html=True
                )
            with col4:
                hp_cf = st.number_input(
                    "Holding Period (hari)", min_value=1, max_value=504,
                    value=1, step=1, key="hp_input_tab5"
                )
                hp_lbl = f"{hp_cf} hari"

        st.markdown(
            f'<div class="box-info">'
            f'<strong>Parameter Terpilih</strong><br>'
            f'Confidence Level: <strong>{conf_cf}%</strong> (α = {alpha_cf:.4f}) &nbsp;|&nbsp; '
            f'Holding Period: <strong>{hp_lbl}</strong> &nbsp;|&nbsp; '
            f'V₀ = {currency_symbol} {v0_cf:,.0f}'
            f'</div>',
            unsafe_allow_html=True
        )

        # ============================================================
        # TOMBOL HITUNG
        # ============================================================
        if st.button("Hitung VaR & ES — Cornish-Fisher",
                     type="primary", use_container_width=True, key="calc_cf_tab5"):

            clean_lr = pd.Series(log_returns).replace([np.inf, -np.inf], np.nan).dropna()
            mu    = clean_lr.mean()
            sigma = clean_lr.std(ddof=1)

            if len(clean_lr) < 2 or pd.isna(sigma) or sigma <= 0:
                st.markdown(
                    '<div class="box-error">❌ Standar deviasi tidak valid. '
                    'Pastikan data log return memiliki variasi dan jumlah observasi mencukupi.</div>',
                    unsafe_allow_html=True
                )
                st.stop()

            S     = stats.skew(clean_lr, bias=False)
            EK    = stats.kurtosis(clean_lr, fisher=True, bias=False)
            K     = EK + 3

            z_norm = stats.norm.ppf(1 - alpha_cf)
            z_cf   = (z_norm
                      + (1/6)  * (z_norm**2 - 1) * S
                      + (1/24) * (z_norm**3 - 3*z_norm) * EK
                      - (1/36) * (2*z_norm**3 - 5*z_norm) * S**2)
            phi_cf = stats.norm.pdf(z_cf)

            # ---- VaR ----
            var_1day_cf = sigma * z_cf
            var_pct_cf  = var_1day_cf * np.sqrt(hp_cf)
            var_amt_cf  = v0_cf * var_pct_cf

            # ---- Horizon returns ----
            if hp_cf == 1:
                horizon_returns = clean_lr.copy()
            else:
                horizon_returns = clean_lr.rolling(window=hp_cf).sum().dropna()

            # ---- ES: tail loss nominal langsung ----
            losses_nominal     = -v0_cf * horizon_returns
            tail_losses_nominal = losses_nominal[losses_nominal > var_amt_cf]

            if len(tail_losses_nominal) >= 3:
                es_amt_cf = tail_losses_nominal.mean()
                es_method = f"empiris nominal ({len(tail_losses_nominal)} observasi tail)"
            else:
                es_amt_cf = v0_cf * sigma * (phi_cf / alpha_cf) * np.sqrt(hp_cf)
                es_method = "parametrik nominal (fallback: data tail < 3 observasi)"

            es_pct_cf = es_amt_cf / v0_cf   # proporsi, hanya untuk tampilan

            st.markdown(
                '<div class="box-success">✅ Perhitungan Cornish-Fisher selesai.</div>',
                unsafe_allow_html=True
            )

            # ============================================================
            # PARAMETER STATISTIK
            # ============================================================
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### Parameter Statistik Data")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Mean (μ)",        f"{mu:.7f}")
            s2.metric("Std Dev (σ)",     f"{sigma:.7f}")
            s3.metric("Skewness (S)",    f"{S:.7f}")
            s4.metric("Kurtosis (K)",    f"{K:.7f}")

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Excess Kurtosis (EK)", f"{EK:.7f}")
            e2.metric("Z normal",             f"{z_norm:.6f}")
            e3.metric("Z_CF",                 f"{z_cf:.6f}")
            e4.metric("Koreksi CF",           f"{z_cf - z_norm:+.6f}")

            st.markdown(
                '<p style="color:#475569; font-size:0.8rem; margin-top:0.6rem; margin-bottom:0.15rem; font-family:monospace;">μ dan σ</p>',
                unsafe_allow_html=True
            )
            st.code(
                f"μ  = {mu:.8f}\n"
                f"   rata-rata log return harian dari data historis\n"
                f"   (diasumsikan ≈ 0 dalam perhitungan VaR)",
                language=None
            )
            st.code(
                f"σ  = {sigma:.8f}\n"
                f"   ukuran volatilitas return — semakin besar σ, semakin berisiko",
                language=None
            )

            st.markdown(
                '<p style="color:#475569; font-size:0.8rem; margin-top:0.6rem; margin-bottom:0.15rem; font-family:monospace;">Skewness dan Kurtosis</p>',
                unsafe_allow_html=True
            )
            st.code(
                f"S  (skewness)          = {S:.8f}\n"
                f"   kemencengan distribusi; S > 0 = condong kanan, S < 0 = condong kiri",
                language=None
            )
            st.code(
                f"K  (kurtosis total)    = {K:.8f}  (distribusi normal = 3)\n"
                f"EK (excess kurtosis)   = {EK:.8f}  (distribusi normal = 0)\n"
                f"   EK digunakan langsung dalam rumus CF\n"
                f"   EK > 0 → fat-tail → risiko ekstrem lebih tinggi dari asumsi normal",
                language=None
            )

            st.markdown(
                '<p style="color:#475569; font-size:0.8rem; margin-top:0.6rem; margin-bottom:0.15rem; font-family:monospace;">Z-score dan φ(Z_CF)</p>',
                unsafe_allow_html=True
            )
            st.code(
                f"Z(1−α) = Z({1-alpha_cf:.2f}) = {z_norm:.6f}\n"
                f"   kuantil distribusi normal standar pada tingkat kepercayaan {conf_cf}%",
                language=None
            )
            st.code(
                f"Z_CF = {z_cf:.6f}\n"
                f"   kuantil Cornish-Fisher setelah disesuaikan dengan S dan EK\n"
                f"   koreksi dari Z normal: {z_cf - z_norm:+.6f}  "
                f"({'Z_CF > Z normal → VaR lebih konservatif' if z_cf > z_norm else 'Z_CF < Z normal → VaR lebih kecil dari VC'})",
                language=None
            )

            st.markdown(
                '<p style="color:#475569; font-size:0.8rem; margin-top:0.6rem; margin-bottom:0.15rem; font-family:monospace;">α dan t</p>',
                unsafe_allow_html=True
            )
            st.code(
                f"α  = {alpha_cf:.4f}\n"
                f"   tingkat signifikansi — probabilitas kerugian melebihi VaR ({int(alpha_cf*100)}%)",
                language=None
            )
            st.code(
                f"t  = {hp_cf} hari  (holding period)\n"
                f"   jangka waktu kepemilikan; VaR diskala dengan faktor √t",
                language=None
            )

            # ============================================================
            # HASIL VaR & ES
            # ============================================================
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### Hasil Perhitungan")

            r1, r2 = st.columns(2)
            with r1:
                st.markdown(
                    '<p style="color:#475569; font-size:0.8rem; margin-bottom:0.2rem; font-family:monospace;">— VaR Cornish-Fisher —</p>',
                    unsafe_allow_html=True
                )
                st.code(
                    f"VaR_CF = V₀ × σ × Z_CF × √t\n"
                    f"       = {v0_cf:,.0f} × {sigma:.6f} × {z_cf:.6f} × √{hp_cf}\n"
                    f"       = {currency_symbol} {var_amt_cf:,.2f}",
                    language=None
                )
                va1, va2 = st.columns(2)
                va1.metric(f"VaR_CF {conf_cf}% (% investasi)",
                           f"{(var_amt_cf/v0_cf)*100:.4f}%")
                va2.metric(f"VaR_CF {conf_cf}% ({currency_symbol})",
                           f"{currency_symbol} {var_amt_cf:,.0f}")

            with r2:
                st.markdown(
                    '<p style="color:#475569; font-size:0.8rem; margin-bottom:0.2rem; font-family:monospace;">— Expected Shortfall Cornish-Fisher —</p>',
                    unsafe_allow_html=True
                )
                # Hitung beberapa angka intermediat untuk ditampilkan
                n_tail_disp   = len(tail_losses_nominal)
                sum_tail_disp = tail_losses_nominal.sum() if n_tail_disp > 0 else 0.0
                st.code(
                    f"Langkah 1: Bentuk kerugian nominal tiap observasi horizon {hp_lbl}\n"
                    f"   L_i = −V₀ × r_i\n"
                    f"       = −{v0_cf:,.0f} × r_i\n\n"
                    f"Langkah 2: Saring observasi tail  (L_i > VaR_CF)\n"
                    f"   VaR_CF         = {currency_symbol} {var_amt_cf:,.2f}\n"
                    f"   Jumlah observasi tail (m) = {n_tail_disp}\n\n"
                    f"Langkah 3: Hitung rata-rata tail loss\n"
                    f"   ES_CF = (1/m) × Σ L_i\n"
                    f"         = (1/{n_tail_disp}) × {currency_symbol} {sum_tail_disp:,.2f}\n"
                    f"         = {currency_symbol} {es_amt_cf:,.2f}\n\n"
                    f"[{es_method}]",
                    language=None
                )
                ea1, ea2 = st.columns(2)
                ea1.metric(f"ES_CF {conf_cf}% (% investasi)",
                           f"{(es_amt_cf/v0_cf)*100:.4f}%")
                ea2.metric(f"ES_CF {conf_cf}% ({currency_symbol})",
                           f"{currency_symbol} {es_amt_cf:,.0f}")

            # Tabel ringkasan
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### Tabel Ringkasan Hasil")

            df_res_cf = pd.DataFrame([
                {"Parameter": "Metode",                                "Nilai": "Cornish-Fisher"},
                {"Parameter": "Confidence Level",                      "Nilai": f"{conf_cf}%"},
                {"Parameter": "Alpha (α)",                             "Nilai": f"{alpha_cf:.4f}"},
                {"Parameter": "Holding Period",                        "Nilai": hp_lbl},
                {"Parameter": "Nilai Investasi (V₀)",                  "Nilai": f"{currency_symbol} {v0_cf:,.0f}"},
                {"Parameter": "Mean (μ) — informasi statistik",        "Nilai": f"{mu:.8f}"},
                {"Parameter": "Std Dev (σ)",                           "Nilai": f"{sigma:.8f}"},
                {"Parameter": "Skewness (S)",                          "Nilai": f"{S:.8f}"},
                {"Parameter": "Kurtosis Total (K)",                    "Nilai": f"{K:.8f}"},
                {"Parameter": "Excess Kurtosis (EK = K−3)",            "Nilai": f"{EK:.8f}"},
                {"Parameter": "Z normal Z(1−α)",                       "Nilai": f"{z_norm:.6f}"},
                {"Parameter": "Z_CF",                                  "Nilai": f"{z_cf:.6f}"},
                {"Parameter": "Koreksi CF (Z_CF − Z normal)",          "Nilai": f"{z_cf - z_norm:+.6f}"},
                {"Parameter": "Metode ES",                             "Nilai": es_method},
                {"Parameter": "Jumlah Observasi Tail",                 "Nilai": f"{len(tail_losses_nominal)}"},
                {"Parameter": f"VaR_CF {conf_cf}% ({currency_symbol})", "Nilai": f"{currency_symbol} {var_amt_cf:,.2f}"},
                {"Parameter": f"VaR_CF {conf_cf}% (% investasi)",      "Nilai": f"{(var_amt_cf/v0_cf)*100:.4f}%"},
                {"Parameter": f"ES_CF {conf_cf}% ({currency_symbol})", "Nilai": f"{currency_symbol} {es_amt_cf:,.2f}"},
                {"Parameter": f"ES_CF {conf_cf}% (% investasi)",       "Nilai": f"{(es_amt_cf/v0_cf)*100:.4f}%"},
            ])
            st.dataframe(df_res_cf, use_container_width=True, hide_index=True)

            # ============================================================
            # VISUALISASI
            # ============================================================
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### Visualisasi")

            # Gunakan horizon returns untuk plot
            plot_returns = horizon_returns.copy()
            var_plot_cf  = var_pct_cf       # batas VaR dalam satuan return horizon-t
            es_plot_cf   = es_pct_cf        # proporsi, dipakai sebagai batas return untuk ES

            x_label_plot = 'Log Return (1 hari)' if hp_cf == 1 else f'Log Return (horizon {hp_lbl})'

            fig_cf, axes_cf = plt.subplots(1, 2, figsize=(14, 5))
            fig_cf.patch.set_alpha(0.0)

            ax1 = axes_cf[0]
            ax1.set_facecolor('none')
            ax1.patch.set_alpha(0.0)
            ax1.hist(plot_returns, bins=50, density=True, alpha=0.65,
                     color='#3b82f6', edgecolor=(0,0,0,0.1), label='Log Return Historis')
            x_range = np.linspace(plot_returns.min(), plot_returns.max(), 300)
            mu_h    = plot_returns.mean()
            sigma_h = plot_returns.std()
            ax1.plot(x_range, stats.norm.pdf(x_range, mu_h, sigma_h),
                     color='#f59e0b', linewidth=2, linestyle='--', label='Kurva Normal (ref.)')
            ax1.axvline(-var_plot_cf, color='#ef4444', linewidth=2, linestyle='--',
                        label=f'VaR_CF batas: {-var_plot_cf:.5f}')
            ax1.axvline(-es_plot_cf, color='#f97316', linewidth=2, linestyle=':',
                        label=f'ES_CF ekuivalen: {-es_plot_cf:.5f}')
            x_tail_hist = x_range[x_range < -var_plot_cf]
            if len(x_tail_hist) > 0:
                ax1.fill_between(x_tail_hist,
                                 stats.norm.pdf(x_tail_hist, mu_h, sigma_h),
                                 alpha=0.2, color='#ef4444')
            ax1.set_xlabel(x_label_plot, fontsize=10)
            ax1.set_ylabel('Density',    fontsize=10)
            ax1.set_title('Distribusi Return & Batas VaR_CF / ES_CF',
                          fontsize=12, fontweight='bold')
            ax1.tick_params(labelsize=9)
            for sp in ax1.spines.values():
                sp.set_color('gray'); sp.set_alpha(0.4)
            ax1.legend(fontsize=8.5)
            ax1.grid(True, alpha=0.2)

            ax2 = axes_cf[1]
            ax2.set_facecolor('none')
            ax2.patch.set_alpha(0.0)
            x_cdf = np.linspace(mu_h - 4*sigma_h, mu_h + 4*sigma_h, 500)
            y_cdf = stats.norm.cdf(x_cdf, mu_h, sigma_h)
            ax2.plot(x_cdf, y_cdf, color='#6366f1', linewidth=2, label='CDF Normal (ref.)')
            ax2.axvline(-var_plot_cf, color='#ef4444', linewidth=2, linestyle='--',
                        label=f'VaR_CF {conf_cf}%')
            ax2.axhline(alpha_cf, color='#ef4444', linewidth=1, linestyle=':', alpha=0.6,
                        label=f'α = {alpha_cf:.4f}')
            x_fill = np.linspace(mu_h - 4*sigma_h, -var_plot_cf, 200)
            y_fill = stats.norm.cdf(x_fill, mu_h, sigma_h)
            ax2.fill_between(x_fill, 0, y_fill, alpha=0.25, color='#ef4444',
                             label=f'Tail ({int(alpha_cf*100)}%)')
            ax2.set_xlabel(x_label_plot,             fontsize=10)
            ax2.set_ylabel('Probabilitas Kumulatif', fontsize=10)
            ax2.set_title('CDF & Area Tail (CF)',    fontsize=12, fontweight='bold')
            ax2.tick_params(labelsize=9)
            for sp in ax2.spines.values():
                sp.set_color('gray'); sp.set_alpha(0.4)
            ax2.legend(fontsize=8.5)
            ax2.grid(True, alpha=0.2)

            plt.tight_layout()
            st.pyplot(fig_cf, transparent=True)
            plt.close()

            # Interpretasi
            if K > 5:
                tail_desc = "Ekor distribusi sangat tebal (fat-tail kuat) — risiko ekstrem jauh lebih tinggi dari asumsi normal."
            elif K > 3:
                tail_desc = "Ekor distribusi sedikit lebih tebal dari normal — Cornish-Fisher memperhitungkan ini melalui EK."
            else:
                tail_desc = "Ekor distribusi lebih tipis dari normal — koreksi CF minimal."

            st.markdown(
                '<p style="color:#475569; font-size:0.8rem; margin-top:0.5rem; margin-bottom:0.15rem; font-family:monospace;">— Interpretasi —</p>',
                unsafe_allow_html=True
            )
            st.code(
                f"Distribusi log return historis dianalisis dengan koreksi Cornish-Fisher\n"
                f"melalui skewness (S = {S:.4f}) dan excess kurtosis (EK = {EK:.4f}).\n"
                f"{tail_desc}\n"
                f"Z_CF = {z_cf:.4f} dibanding Z normal = {z_norm:.4f}.\n"
                f"VaR_CF pada horizon {hp_lbl} sebesar {currency_symbol} {var_amt_cf:,.0f}.\n"
                f"ES_CF dihitung sebagai rata-rata seluruh kerugian nominal yang melampaui VaR_CF.\n"
                f"Dari {len(tail_losses_nominal)} observasi tail, rata-rata kerugian tersebut adalah\n"
                f"{currency_symbol} {es_amt_cf:,.0f} atau {(es_amt_cf/v0_cf)*100:.2f}% dari investasi.",
                language=None
            )

            # ============================================================
            # INTERPRETASI HASIL & TINGKAT RISIKO
            # ============================================================
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### Interpretasi Hasil")

            var_pct_inv = (var_amt_cf / v0_cf) * 100
            es_pct_inv  = (es_amt_cf  / v0_cf) * 100

            if var_pct_inv < 5:
                risk_label, risk_box = "RENDAH",        "box-success"
            elif var_pct_inv < 10:
                risk_label, risk_box = "SEDANG",        "box-warning"
            elif var_pct_inv < 20:
                risk_label, risk_box = "TINGGI",        "box-error"
            else:
                risk_label, risk_box = "SANGAT TINGGI", "box-error"

            st.markdown(
                f'<div class="{risk_box}">'
                f'<strong>Tingkat Risiko: {risk_label}</strong><br>'
                f'VaR_CF {conf_cf}% ({hp_lbl}): '
                f'<strong>{currency_symbol} {var_amt_cf:,.0f}</strong> '
                f'({var_pct_inv:.2f}% dari investasi)<br>'
                f'ES_CF {conf_cf}% ({hp_lbl}): '
                f'<strong>{currency_symbol} {es_amt_cf:,.0f}</strong> '
                f'({es_pct_inv:.2f}% dari investasi)'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<p style="color:#475569; font-size:0.8rem; margin-bottom:0.15rem; font-family:monospace;">— Interpretasi VaR_CF —</p>',
                unsafe_allow_html=True
            )
            st.code(
                f"Dengan tingkat kepercayaan {conf_cf}%, kerugian dalam {hp_lbl}\n"
                f"tidak akan melebihi {currency_symbol} {var_amt_cf:,.0f} ({var_pct_inv:.2f}% dari investasi).\n"
                f"Hasil ini sudah disesuaikan dengan skewness (S = {S:.4f})\n"
                f"dan excess kurtosis (EK = {EK:.4f}) data aktual.\n"
                f"Terdapat {int(alpha_cf*100)}% kemungkinan kerugian melebihi batas ini.",
                language=None
            )

            st.markdown(
                '<p style="color:#475569; font-size:0.8rem; margin-bottom:0.15rem; font-family:monospace;">— Interpretasi ES_CF —</p>',
                unsafe_allow_html=True
            )
            st.code(
                f"ES_CF dihitung langsung sebagai rata-rata kerugian nominal\n"
                f"dari {len(tail_losses_nominal)} observasi yang melampaui VaR_CF ({es_method}).\n"
                f"Nilai ES_CF = {currency_symbol} {es_amt_cf:,.0f} ({es_pct_inv:.2f}% dari investasi).\n"
                f"ES_CF selalu ≥ VaR_CF — lebih konservatif dan informatif untuk tail risk.",
                language=None
            )

            if is_normal is False:
                st.markdown(
                    '<div class="box-info">ℹ️ Data tidak normal berdasarkan uji Jarque–Bera — '
                    'Cornish-Fisher lebih tepat dibanding Variance-Covariance karena memperhitungkan '
                    'skewness dan kurtosis. Perbandingan keduanya tersedia di Tab 6.</div>',
                    unsafe_allow_html=True
                )

            # ============================================================
            # SIMPAN & DOWNLOAD
            # ============================================================
            st.session_state['var_cf_results'] = {
                'method'              : 'Cornish-Fisher',
                'confidence_level'    : conf_cf,
                'alpha'               : alpha_cf,
                'holding_period'      : hp_cf,
                'holding_period_label': hp_lbl,
                'v0'                  : v0_cf,
                'currency'            : currency_symbol,
                'mu'                  : mu,
                'sigma'               : sigma,
                'S'                   : S,
                'K'                   : K,
                'EK'                  : EK,
                'z_norm'              : z_norm,
                'z_cf'                : z_cf,
                'phi_cf'              : phi_cf,
                'var_pct'             : var_pct_cf,
                'var_amount'          : var_amt_cf,
                'es_pct'              : es_pct_cf,
                'es_amount'           : es_amt_cf,
                'es_method'           : es_method,
                'n_tail'              : len(tail_losses_nominal),
                'risk_level'          : risk_label,
                'is_normal'           : is_normal,
            }

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            output_cf = io.BytesIO()
            with pd.ExcelWriter(output_cf, engine='openpyxl') as writer:
                df_res_cf.to_excel(writer, index=False, sheet_name='Hasil VaR CF')
                pd.DataFrame([{
                    'mu': mu, 'sigma': sigma, 'S': S, 'K': K, 'EK': EK,
                    'z_norm': z_norm, 'z_cf': z_cf, 'phi_cf': phi_cf,
                    'alpha': alpha_cf, 'holding_period': hp_cf,
                    'v0': v0_cf, 'currency': currency_symbol,
                    'var_pct': var_pct_cf, 'var_amount': var_amt_cf,
                    'es_pct': es_pct_cf,   'es_amount': es_amt_cf,
                    'es_method': es_method, 'n_tail': len(tail_losses_nominal)
                }]).to_excel(writer, index=False, sheet_name='Parameter')

            st.download_button(
                label="💾 Download Hasil VaR Cornish-Fisher",
                data=output_cf.getvalue(),
                file_name=f"var_cf_{conf_cf}pct_{hp_cf}days.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
with tab6:
    st.header("Perbandingan & Backtesting")

    if 'var_vc_results' not in st.session_state or 'var_cf_results' not in st.session_state:
        st.markdown("""
        <div class="box-warning">
            ⚠️ Hitung VaR di <strong>Tab 4 (Variance-Covariance)</strong> dan
            <strong>Tab 5 (Cornish-Fisher)</strong> terlebih dahulu.
        </div>
        """, unsafe_allow_html=True)
    else:
        vc_results  = st.session_state['var_vc_results']
        cf_results  = st.session_state['var_cf_results']
        log_returns = st.session_state['log_returns']

        currency_symbol = vc_results.get('currency', cf_results.get('currency', 'Rp'))

        # ============================================================
        # 6.1 PERBANDINGAN NILAI VaR DAN ES
        # ============================================================
        st.markdown("#### 6.1 Perbandingan Nilai VaR dan ES")

        st.info(
            "Bagian ini menampilkan perbandingan estimasi Value at Risk (VaR) dan "
            "Expected Shortfall (ES) menggunakan metode Variance-Covariance dan Cornish-Fisher."
        )

        vc_var    = vc_results['var_amount']
        vc_es     = vc_results['es_amount']
        cf_var    = cf_results['var_amount']
        cf_es     = cf_results['es_amount']
        vc_conf   = vc_results['confidence_level']
        cf_conf   = cf_results['confidence_level']
        vc_hp     = vc_results['holding_period']
        cf_hp     = cf_results['holding_period']
        vc_hp_lbl = vc_results['holding_period_label']
        cf_hp_lbl = cf_results['holding_period_label']

        # FIX 3: cek kesamaan parameter Tab 4 dan Tab 5
        same_conf = vc_conf == cf_conf
        same_hp   = vc_hp   == cf_hp
        if not (same_conf and same_hp):
            st.markdown(
                '<div class="box-warning">'
                '⚠️ Hasil Tab 4 dan Tab 5 belum menggunakan parameter yang sama. '
                'Perbandingan nominal tetap ditampilkan, tetapi untuk interpretasi yang adil '
                'sebaiknya hitung ulang keduanya dengan confidence level dan holding period yang sama.'
                '</div>',
                unsafe_allow_html=True
            )

        # FIX 2: label "% investasi" menggantikan "return"
        df_compare = pd.DataFrame([
            {
                'Metode'           : 'Variance-Covariance',
                f'VaR ({currency_symbol})' : f"{currency_symbol} {vc_var:,.2f}",
                f'ES ({currency_symbol})'  : f"{currency_symbol} {vc_es:,.2f}",
                'VaR (% investasi)': f"{vc_results['var_pct'] * 100:.4f}%",
                'ES (% investasi)' : f"{vc_results['es_pct'] * 100:.4f}%",
                'Confidence Level' : f"{vc_conf}%",
                'Holding Period'   : vc_hp_lbl,
            },
            {
                'Metode'           : 'Cornish-Fisher',
                f'VaR ({currency_symbol})' : f"{currency_symbol} {cf_var:,.2f}",
                f'ES ({currency_symbol})'  : f"{currency_symbol} {cf_es:,.2f}",
                'VaR (% investasi)': f"{cf_results['var_pct'] * 100:.4f}%",
                'ES (% investasi)' : f"{cf_results['es_pct'] * 100:.4f}%",
                'Confidence Level' : f"{cf_conf}%",
                'Holding Period'   : cf_hp_lbl,
            },
        ])
        st.dataframe(df_compare, use_container_width=True, hide_index=True)

        diff_var = vc_var - cf_var
        diff_es  = vc_es  - cf_es
        st.markdown(
            f'<div class="box-info">'
            f'Selisih VaR (VC − CF): <strong>{currency_symbol} {diff_var:+,.2f}</strong> &nbsp;|&nbsp; '
            f'Selisih ES (VC − CF): <strong>{currency_symbol} {diff_es:+,.2f}</strong>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Visualisasi bar chart dark theme
        fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(13, 5))
        fig_cmp.patch.set_alpha(0.0)
        for ax, vals, title, ylabel in [
            (axes_cmp[0], [vc_var, cf_var], f'Perbandingan VaR ({vc_conf}%, {vc_hp_lbl})', f'VaR ({currency_symbol})'),
            (axes_cmp[1], [vc_es,  cf_es],  f'Perbandingan ES ({vc_conf}%, {vc_hp_lbl})',  f'ES ({currency_symbol})'),
        ]:
            ax.set_facecolor('none')
            ax.patch.set_alpha(0.0)
            bars = ax.bar(['Variance-Covariance', 'Cornish-Fisher'], vals,
                          color=['#3b82f6', '#f87171'], alpha=0.85, width=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{currency_symbol} {val:,.0f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.tick_params(labelsize=9)
            for sp in ax.spines.values():
                sp.set_color('gray'); sp.set_alpha(0.4)
            ax.yaxis.grid(True, alpha=0.2)
            ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig_cmp, transparent=True)
        plt.close()

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ============================================================
        # 6.2 PARAMETER BACKTESTING
        # ============================================================
        st.markdown("#### 6.2 Parameter Backtesting")

        backtest_mode = st.radio(
            "Mode Backtesting",
            ["Gunakan Parameter dari Tab 4 & 5", "Input Manual"],
            horizontal=True,
            key="backtest_mode_tab6"
        )

        if backtest_mode == "Gunakan Parameter dari Tab 4 & 5":
            backtest_conf   = vc_conf
            backtest_alpha  = vc_results['alpha']
            backtest_hp     = vc_results['holding_period']
            backtest_hp_lbl = vc_results['holding_period_label']
            st.markdown(
                f'<div class="box-info">'
                f'Confidence Level: <strong>{backtest_conf}%</strong> (α = {backtest_alpha:.4f}) &nbsp;|&nbsp; '
                f'Holding Period: <strong>{backtest_hp_lbl}</strong>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                backtest_conf = st.selectbox(
                    "Confidence Level",
                    options=[99, 95, 90],
                    format_func=lambda x: f"{x}%  (α = {100-x}%)",
                    index=1,
                    key="backtest_conf_tab6"
                )
                backtest_alpha = (100 - backtest_conf) / 100
            with col_b2:
                hp_opt = st.selectbox(
                    "Holding Period",
                    ["1 hari", "5 hari (1 minggu kerja)",
                     "20 hari (1 bulan kerja)", "252 hari (1 tahun kerja)"],
                    index=0,
                    key="backtest_hp_tab6"
                )
                hp_map_bt   = {"1 hari": 1, "5 hari (1 minggu kerja)": 5,
                                "20 hari (1 bulan kerja)": 20, "252 hari (1 tahun kerja)": 252}
                backtest_hp     = hp_map_bt[hp_opt]
                backtest_hp_lbl = hp_opt

        backtest_method = st.selectbox(
            "Pilih Metode untuk Backtesting",
            ['Variance-Covariance', 'Cornish-Fisher', 'Keduanya'],
            key="backtest_method_tab6"
        )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ============================================================
        # TOMBOL JALANKAN BACKTESTING
        # ============================================================
        if st.button("Jalankan Backtesting", type="primary",
                     use_container_width=True, key="run_backtest_tab6"):

            clean_lr = pd.Series(log_returns).replace([np.inf, -np.inf], np.nan).dropna()
            n_total  = len(clean_lr)

            # ---- 6.3 Skema Data Training & Testing ----
            st.markdown("#### 6.3 Skema Data Training dan Testing")

            if n_total >= 260:
                TE      = 250
                mode_te = "Standar Jorion (250 hari)"
            elif 200 <= n_total < 260:
                TE      = 200
                mode_te = "Kompromi (200 hari — data terbatas)"
            else:
                TE      = int(0.7 * n_total)
                mode_te = "Proporsional 70% — data sangat terbatas"

            TT             = n_total - TE
            n_observations = max(TT - backtest_hp + 1, 0)

            st.markdown(
                f'<div class="box-info">'
                f'<strong>Total Data:</strong> {n_total} hari<br>'
                f'<strong>Training Window (TE):</strong> {TE} hari ({mode_te})<br>'
                f'<strong>Testing Window (TT):</strong> {TT} hari<br>'
                f'<strong>Holding Period:</strong> {backtest_hp_lbl}<br>'
                f'<strong>Jumlah Jendela Uji Efektif:</strong> {n_observations}'
                f'</div>',
                unsafe_allow_html=True
            )

            if TT < 30:
                st.markdown(
                    f'<div class="box-warning">⚠️ Observasi uji (TT = {TT}) sangat kecil. '
                    f'Ideal TT ≥ 100. Hasil backtesting harus ditafsirkan dengan hati-hati.</div>',
                    unsafe_allow_html=True
                )

            if n_observations <= 0:
                st.markdown(
                    f'<div class="box-error">❌ Data testing tidak cukup untuk holding period {backtest_hp} hari. '
                    f'Gunakan holding period lebih kecil atau tambah data.</div>',
                    unsafe_allow_html=True
                )
            else:
                # ---- 6.4 Perhitungan Backtesting ----
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("#### 6.4 Perhitungan Backtesting VaR")

                train_lr = clean_lr.iloc[:TE]
                mean_r   = train_lr.mean()
                std_r    = train_lr.std(ddof=1)
                skew_r   = stats.skew(train_lr, bias=False)
                ek_r     = stats.kurtosis(train_lr, fisher=True, bias=False)

                st.markdown(
                    '<p style="color:#475569; font-size:0.8rem; margin-bottom:0.15rem; font-family:monospace;">— Parameter dari data training —</p>',
                    unsafe_allow_html=True
                )
                st.code(
                    f"μ  = {mean_r:.8f}   (dari {TE} observasi training)\n"
                    f"σ  = {std_r:.8f}\n"
                    f"S  = {skew_r:.8f}\n"
                    f"EK = {ek_r:.8f}",
                    language=None
                )

                # FIX 1: threshold negatif (return ekor kiri)
                z_vc_bt   = stats.norm.ppf(1 - backtest_alpha)
                var_vc_bt = -std_r * z_vc_bt * np.sqrt(backtest_hp)

                z_norm_bt = stats.norm.ppf(1 - backtest_alpha)
                z_cf_bt   = (z_norm_bt
                             + (1/6)  * (z_norm_bt**2 - 1) * skew_r
                             + (1/24) * (z_norm_bt**3 - 3*z_norm_bt) * ek_r
                             - (1/36) * (2*z_norm_bt**3 - 5*z_norm_bt) * skew_r**2)
                var_cf_bt = -std_r * z_cf_bt * np.sqrt(backtest_hp)

                # FIX 4: redaksi "return threshold"
                st.markdown(
                    '<p style="color:#475569; font-size:0.8rem; margin-bottom:0.15rem; font-family:monospace;">— VaR yang digunakan sebagai batas backtesting —</p>',
                    unsafe_allow_html=True
                )
                st.code(
                    f"VaR_VC (return threshold, {backtest_hp}-hari) = {var_vc_bt:.8f}\n"
                    f"VaR_CF (return threshold, {backtest_hp}-hari) = {var_cf_bt:.8f}\n\n"
                    f"Pelanggaran VC: r_t < VaR_VC_t  →  I_t = 1\n"
                    f"Pelanggaran CF: r_t < VaR_CF_t  →  I_t = 1",
                    language=None
                )

                # Rolling window
                backtest_data_vc = []
                backtest_data_cf = []
                violations_vc    = 0
                violations_cf    = 0

                for i in range(n_observations):
                    ws  = TE + i
                    we  = ws + backtest_hp
                    r_t = clean_lr.iloc[ws:we].sum()

                    viol_vc = 1 if r_t < var_vc_bt else 0
                    viol_cf = 1 if r_t < var_cf_bt else 0
                    violations_vc += viol_vc
                    violations_cf += viol_cf

                    if backtest_method in ['Variance-Covariance', 'Keduanya']:
                        backtest_data_vc.append({
                            'Periode'          : f"VaR({i+1})",
                            'Batas VaR_t (VC)' : round(var_vc_bt, 8),  # FIX 5: nama kolom
                            'r_t'              : round(r_t, 8),
                            'I_t'              : viol_vc
                        })
                    if backtest_method in ['Cornish-Fisher', 'Keduanya']:
                        backtest_data_cf.append({
                            'Periode'          : f"VaR({i+1})",
                            'Batas VaR_t (CF)' : round(var_cf_bt, 8),  # FIX 5: nama kolom
                            'r_t'              : round(r_t, 8),
                            'I_t'              : viol_cf
                        })

                st.markdown(
                    '<div class="box-success">✅ Backtesting selesai.</div>',
                    unsafe_allow_html=True
                )

                # ---- 6.5 Tabel Pelanggaran ----
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown(f"#### 6.5 Tabel Pelanggaran VaR — Holding Period {backtest_hp_lbl}")

                if backtest_method in ['Variance-Covariance', 'Keduanya']:
                    st.markdown("**Tabel Backtesting Variance-Covariance**")
                    st.dataframe(pd.DataFrame(backtest_data_vc),
                                 use_container_width=True, hide_index=True, height=360)
                    st.markdown(
                        f'<p style="color:#64748b; font-size:0.82rem;">{len(backtest_data_vc)} observasi ditampilkan</p>',
                        unsafe_allow_html=True
                    )

                if backtest_method in ['Cornish-Fisher', 'Keduanya']:
                    st.markdown("**Tabel Backtesting Cornish-Fisher**")
                    st.dataframe(pd.DataFrame(backtest_data_cf),
                                 use_container_width=True, hide_index=True, height=360)
                    st.markdown(
                        f'<p style="color:#64748b; font-size:0.82rem;">{len(backtest_data_cf)} observasi ditampilkan</p>',
                        unsafe_allow_html=True
                    )

                # ---- 6.6 Uji Kupiec ----
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("#### 6.6 Hasil Uji Kupiec")

                def kupiec_test(x, n, alpha):
                    if n <= 0:
                        return None, None, None
                    p_hat      = x / n
                    eps        = 1e-10
                    p_hat_safe = min(max(p_hat, eps), 1 - eps)
                    alpha_safe = min(max(alpha, eps), 1 - eps)
                    lr_uc = -2 * np.log(
                        ((1 - alpha_safe)**(n - x) * alpha_safe**x) /
                        ((1 - p_hat_safe)**(n - x) * p_hat_safe**x)
                    )
                    p_value = 1 - stats.chi2.cdf(lr_uc, df=1)
                    return lr_uc, p_value, p_hat

                def fmt_pval(p):
                    """Format p-value sampai digit signifikan pertama (1-9) muncul.
                    Contoh: 0.0000123 → '0.0000123', 1.23e-12 → '1.23 × 10⁻¹²'"""
                    if p is None:
                        return "N/A"
                    if p == 0.0:
                        return "< 1 × 10⁻¹⁰"
                    if p >= 0.0001:
                        # Temukan jumlah desimal yang diperlukan
                        import math
                        decimals = max(4, -int(math.floor(math.log10(p))) + 2)
                        decimals = min(decimals, 10)
                        return f"{p:.{decimals}f}"
                    else:
                        # Notasi ilmiah yang rapi
                        exp   = int(np.floor(np.log10(p)))
                        coeff = p / (10 ** exp)
                        sup   = str(exp).replace('-', '⁻').translate(
                            str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')
                        )
                        return f"{coeff:.2f} × 10{sup}"

                expected_violations = backtest_alpha * n_observations
                lr_vc = pval_vc = phat_vc = None
                lr_cf = pval_cf = phat_cf = None

                col_k1, col_k2 = st.columns(2)

                if backtest_method in ['Variance-Covariance', 'Keduanya']:
                    with col_k1:
                        lr_vc, pval_vc, phat_vc = kupiec_test(
                            violations_vc, n_observations, backtest_alpha
                        )
                        status_vc = "✅ Diterima" if (pval_vc is not None and pval_vc > 0.05) else "❌ Ditolak"
                        box_vc    = "box-success" if (pval_vc is not None and pval_vc > 0.05) else "box-error"
                        st.markdown(
                            f'<div class="{box_vc}">'
                            f'<strong>Variance-Covariance</strong><br>'
                            f'Jumlah Observasi (n): {n_observations}<br>'
                            f'Jumlah Pelanggaran (x): {violations_vc}<br>'
                            f'Expected Violations (n × α): {expected_violations:.2f}<br>'
                            f'Proporsi Aktual (p̂): {phat_vc:.4f}<br>'
                            f'LR Kupiec: {lr_vc:.4f}<br>'
                            f'p-value: {fmt_pval(pval_vc)}<br>'
                            f'Status Model: <strong>{status_vc}</strong>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                if backtest_method in ['Cornish-Fisher', 'Keduanya']:
                    with col_k2:
                        lr_cf, pval_cf, phat_cf = kupiec_test(
                            violations_cf, n_observations, backtest_alpha
                        )
                        status_cf = "✅ Diterima" if (pval_cf is not None and pval_cf > 0.05) else "❌ Ditolak"
                        box_cf    = "box-success" if (pval_cf is not None and pval_cf > 0.05) else "box-error"
                        st.markdown(
                            f'<div class="{box_cf}">'
                            f'<strong>Cornish-Fisher</strong><br>'
                            f'Jumlah Observasi (n): {n_observations}<br>'
                            f'Jumlah Pelanggaran (x): {violations_cf}<br>'
                            f'Expected Violations (n × α): {expected_violations:.2f}<br>'
                            f'Proporsi Aktual (p̂): {phat_cf:.4f}<br>'
                            f'LR Kupiec: {lr_cf:.4f}<br>'
                            f'p-value: {fmt_pval(pval_cf)}<br>'
                            f'Status Model: <strong>{status_cf}</strong>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                # ---- 6.7 Visualisasi ----
                if backtest_method == 'Keduanya':
                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                    st.markdown("#### 6.7 Visualisasi Backtesting")

                    fig_bt, ax_bt = plt.subplots(figsize=(9, 5))
                    fig_bt.patch.set_alpha(0.0)
                    ax_bt.set_facecolor('none')
                    ax_bt.patch.set_alpha(0.0)

                    methods_viol     = ['Variance-Covariance', 'Cornish-Fisher', 'Expected']
                    violation_counts = [violations_vc, violations_cf, expected_violations]
                    colors_bt        = ['#3b82f6', '#f87171', '#f59e0b']

                    bars_bt = ax_bt.bar(methods_viol, violation_counts,
                                        color=colors_bt, alpha=0.85, width=0.5)
                    for bar, val in zip(bars_bt, violation_counts):
                        ax_bt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                   f'{val:.1f}', ha='center', va='bottom',
                                   fontsize=10, fontweight='bold')
                    ax_bt.set_ylabel('Jumlah Pelanggaran', fontsize=10)
                    ax_bt.set_title(
                        f'Perbandingan Violations: Aktual vs Expected\n'
                        f'({backtest_conf}%, {backtest_hp_lbl}, n = {n_observations})',
                        fontsize=11, fontweight='bold'
                    )
                    ax_bt.tick_params(labelsize=9)
                    for sp in ax_bt.spines.values():
                        sp.set_color('gray'); sp.set_alpha(0.4)
                    ax_bt.yaxis.grid(True, alpha=0.2)
                    ax_bt.set_axisbelow(True)
                    plt.tight_layout()
                    st.pyplot(fig_bt, transparent=True)
                    plt.close()

                # ---- 6.8 Interpretasi ----
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("#### 6.8 Interpretasi Hasil")

                vc_narasi = f"{violations_vc} dari {n_observations} observasi" \
                    if backtest_method in ['Variance-Covariance', 'Keduanya'] else "—"
                cf_narasi = f"{violations_cf} dari {n_observations} observasi" \
                    if backtest_method in ['Cornish-Fisher', 'Keduanya'] else "—"

                # Narasi dasar
                st.markdown(
                    f'<div class="box-info">'
                    f'Berdasarkan hasil backtesting, metode Variance-Covariance menghasilkan '
                    f'jumlah pelanggaran sebesar <strong>{vc_narasi}</strong>, '
                    f'sedangkan metode Cornish-Fisher menghasilkan jumlah pelanggaran sebesar '
                    f'<strong>{cf_narasi}</strong>. '
                    f'Model VaR dinyatakan diterima apabila p-value Uji Kupiec lebih besar dari 0,05.'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # ---- Interpretasi perbandingan dinamis (hanya jika Keduanya) ----
                if backtest_method == 'Keduanya' and pval_vc is not None and pval_cf is not None:

                    vc_accepted = pval_vc > 0.05
                    cf_accepted = pval_cf > 0.05

                    # Selisih proporsi pelanggaran terhadap alpha
                    vc_dev = abs(phat_vc - backtest_alpha)
                    cf_dev = abs(phat_cf - backtest_alpha)

                    # Tentukan metode yang lebih baik
                    if vc_accepted and not cf_accepted:
                        winner        = "Variance-Covariance"
                        loser         = "Cornish-Fisher"
                        winner_pval   = pval_vc
                        loser_pval    = pval_cf
                        winner_viol   = violations_vc
                        loser_viol    = violations_cf
                        winner_phat   = phat_vc
                        loser_phat    = phat_cf
                        winner_status = "diterima"
                        loser_status  = "ditolak"
                        result_box    = "box-success"

                    elif cf_accepted and not vc_accepted:
                        winner        = "Cornish-Fisher"
                        loser         = "Variance-Covariance"
                        winner_pval   = pval_cf
                        loser_pval    = pval_vc
                        winner_viol   = violations_cf
                        loser_viol    = violations_vc
                        winner_phat   = phat_cf
                        loser_phat    = phat_vc
                        winner_status = "diterima"
                        loser_status  = "ditolak"
                        result_box    = "box-success"

                    elif vc_accepted and cf_accepted:
                        # Kedua diterima — bandingkan kedekatan proporsi ke alpha
                        if vc_dev <= cf_dev:
                            winner        = "Variance-Covariance"
                            loser         = "Cornish-Fisher"
                            winner_pval   = pval_vc
                            loser_pval    = pval_cf
                            winner_viol   = violations_vc
                            loser_viol    = violations_cf
                            winner_phat   = phat_vc
                            loser_phat    = phat_cf
                        else:
                            winner        = "Cornish-Fisher"
                            loser         = "Variance-Covariance"
                            winner_pval   = pval_cf
                            loser_pval    = pval_vc
                            winner_viol   = violations_cf
                            loser_viol    = violations_vc
                            winner_phat   = phat_cf
                            loser_phat    = phat_vc
                        winner_status = "diterima"
                        loser_status  = "diterima"
                        result_box    = "box-success"

                    else:
                        # Kedua ditolak
                        if vc_dev <= cf_dev:
                            winner        = "Variance-Covariance"
                            loser         = "Cornish-Fisher"
                            winner_pval   = pval_vc
                            loser_pval    = pval_cf
                            winner_viol   = violations_vc
                            loser_viol    = violations_cf
                            winner_phat   = phat_vc
                            loser_phat    = phat_cf
                        else:
                            winner        = "Cornish-Fisher"
                            loser         = "Variance-Covariance"
                            winner_pval   = pval_cf
                            loser_pval    = pval_vc
                            winner_viol   = violations_cf
                            loser_viol    = violations_vc
                            winner_phat   = phat_cf
                            loser_phat    = phat_vc
                        winner_status = "ditolak"
                        loser_status  = "ditolak"
                        result_box    = "box-warning"

                    # Teks alasan keputusan
                    is_normal_data = st.session_state.get('is_normal', None)

                    if winner_status == "diterima" and loser_status == "ditolak":
                        alasan = (
                            f"Metode <strong>{winner}</strong> merupakan satu-satunya model yang "
                            f"diterima oleh Uji Kupiec (p-value = {fmt_pval(winner_pval)} &gt; 0,05), "
                            f"sedangkan {loser} ditolak (p-value = {fmt_pval(loser_pval)} ≤ 0,05)."
                        )
                    elif winner_status == "diterima" and loser_status == "diterima":
                        alasan = (
                            f"Kedua model diterima oleh Uji Kupiec. Metode <strong>{winner}</strong> "
                            f"dipilih sebagai yang lebih baik karena proporsi pelanggarannya "
                            f"({winner_phat:.4f}) lebih mendekati α = {backtest_alpha:.4f} "
                            f"dibandingkan {loser} ({loser_phat:.4f})."
                        )
                    else:
                        alasan = (
                            f"Kedua model ditolak oleh Uji Kupiec. Metode <strong>{winner}</strong> "
                            f"menunjukkan kinerja yang relatif lebih baik karena proporsi pelanggarannya "
                            f"({winner_phat:.4f}) lebih mendekati α = {backtest_alpha:.4f} "
                            f"dibandingkan {loser} ({loser_phat:.4f})."
                        )

                    # Konteks normalitas data — dari JB, default None
                    if is_normal_data is False:
                        konteks_normalitas = (
                            f"Perlu diperhatikan bahwa data return tidak berdistribusi normal "
                            f"berdasarkan uji <strong>Jarque–Bera</strong>. Dalam kondisi ini, "
                            f"metode Cornish-Fisher secara teoritis lebih sesuai karena "
                            f"memperhitungkan skewness dan excess kurtosis data aktual."
                        )
                    elif is_normal_data is True:
                        konteks_normalitas = (
                            f"Data return berdistribusi normal berdasarkan uji <strong>Jarque–Bera</strong>, "
                            f"sehingga kedua metode memiliki landasan asumsi yang valid."
                        )
                    else:
                        konteks_normalitas = (
                            f"Status normalitas data belum diketahui (uji normalitas di Tab 3 "
                            f"belum dijalankan). Interpretasi normalitas tidak dapat ditampilkan."
                        )

                    st.markdown(
                        f'<div class="{result_box}">'
                        f'<strong>Kesimpulan Perbandingan:</strong> '
                        f'Metode <strong>{winner}</strong> lebih unggul dalam backtesting ini.<br><br>'
                        f'{alasan}<br><br>'
                        f'Ringkasan:<br>'
                        f'• Variance-Covariance: {violations_vc} pelanggaran, '
                        f'p̂ = {phat_vc:.4f}, p-value Kupiec = {fmt_pval(pval_vc)} '
                        f'({"✅ Diterima" if vc_accepted else "❌ Ditolak"})<br>'
                        f'• Cornish-Fisher: {violations_cf} pelanggaran, '
                        f'p̂ = {phat_cf:.4f}, p-value Kupiec = {fmt_pval(pval_cf)} '
                        f'({"✅ Diterima" if cf_accepted else "❌ Ditolak"})<br>'
                        f'• Expected violations: {expected_violations:.2f} (α = {backtest_alpha:.4f})<br><br>'
                        f'{konteks_normalitas}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                elif backtest_method != 'Keduanya':
                    # Interpretasi untuk satu metode saja
                    if backtest_method == 'Variance-Covariance' and pval_vc is not None:
                        accepted  = pval_vc > 0.05
                        box_s     = "box-success" if accepted else "box-warning"
                        status_s  = "diterima" if accepted else "ditolak"
                        st.markdown(
                            f'<div class="{box_s}">'
                            f'Model Variance-Covariance <strong>{status_s}</strong> '
                            f'berdasarkan Uji Kupiec (p-value = {fmt_pval(pval_vc)}). '
                            f'Jumlah pelanggaran aktual: {violations_vc}, '
                            f'expected: {expected_violations:.2f} (α = {backtest_alpha:.4f}).'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    elif backtest_method == 'Cornish-Fisher' and pval_cf is not None:
                        accepted  = pval_cf > 0.05
                        box_s     = "box-success" if accepted else "box-warning"
                        status_s  = "diterima" if accepted else "ditolak"
                        st.markdown(
                            f'<div class="{box_s}">'
                            f'Model Cornish-Fisher <strong>{status_s}</strong> '
                            f'berdasarkan Uji Kupiec (p-value = {fmt_pval(pval_cf)}). '
                            f'Jumlah pelanggaran aktual: {violations_cf}, '
                            f'expected: {expected_violations:.2f} (α = {backtest_alpha:.4f}).'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                # Catatan metodologi
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("#### Catatan Metodologi")
                st.code(
                    f"1. Split data\n"
                    f"   Training (TE) = {TE} hari  →  estimasi parameter μ, σ, S, EK\n"
                    f"   Testing  (TT) = {TT} hari  →  backtesting out-of-sample\n\n"
                    f"2. VaR dihitung SEKALI dari data training sebagai return threshold (negatif)\n"
                    f"   lalu diuji terhadap setiap jendela data testing\n\n"
                    f"3. Definisi pelanggaran\n"
                    f"   I_t = 1  jika  r_t < VaR_t  (return lebih buruk dari threshold)\n"
                    f"   I_t = 0  jika  r_t ≥ VaR_t\n\n"
                    f"4. Uji Kupiec (LR test)\n"
                    f"   H₀: proporsi pelanggaran = α  (model akurat)\n"
                    f"   Tolak H₀ jika p-value < 0,05\n\n"
                    f"5. Expected violations = n × α = {n_observations} × {backtest_alpha:.4f} = {expected_violations:.2f}",
                    language=None
                )

                # ---- Download ----
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                output_bt = io.BytesIO()
                with pd.ExcelWriter(output_bt, engine='openpyxl') as writer:
                    df_compare.to_excel(writer, index=False, sheet_name='Perbandingan VaR ES')
                    if backtest_data_vc:
                        pd.DataFrame(backtest_data_vc).to_excel(
                            writer, index=False, sheet_name='Backtest VC')
                    if backtest_data_cf:
                        pd.DataFrame(backtest_data_cf).to_excel(
                            writer, index=False, sheet_name='Backtest CF')
                    summary_rows = [
                        {'Parameter': 'Confidence Level',     'Nilai': f"{backtest_conf}%"},
                        {'Parameter': 'Alpha',                'Nilai': f"{backtest_alpha:.4f}"},
                        {'Parameter': 'Holding Period',       'Nilai': backtest_hp_lbl},
                        {'Parameter': 'Total Data',           'Nilai': n_total},
                        {'Parameter': 'Training Window (TE)', 'Nilai': TE},
                        {'Parameter': 'Testing Window (TT)',  'Nilai': TT},
                        {'Parameter': 'Jendela Uji Efektif',  'Nilai': n_observations},
                        {'Parameter': 'Expected Violations',  'Nilai': f"{expected_violations:.2f}"},
                    ]
                    if backtest_method in ['Variance-Covariance', 'Keduanya']:
                        summary_rows += [
                            {'Parameter': 'VC Threshold Return', 'Nilai': f"{var_vc_bt:.8f}"},
                            {'Parameter': 'VC Violations',       'Nilai': violations_vc},
                            {'Parameter': 'VC LR Kupiec',        'Nilai': f"{lr_vc:.4f}"      if lr_vc  is not None else 'N/A'},
                            {'Parameter': 'VC p-value',          'Nilai': fmt_pval(pval_vc)   if pval_vc is not None else 'N/A'},
                            {'Parameter': 'VC Proporsi Aktual',  'Nilai': f"{phat_vc:.4f}"    if phat_vc is not None else 'N/A'},
                        ]
                    if backtest_method in ['Cornish-Fisher', 'Keduanya']:
                        summary_rows += [
                            {'Parameter': 'CF Threshold Return', 'Nilai': f"{var_cf_bt:.8f}"},
                            {'Parameter': 'CF Violations',       'Nilai': violations_cf},
                            {'Parameter': 'CF LR Kupiec',        'Nilai': f"{lr_cf:.4f}"      if lr_cf  is not None else 'N/A'},
                            {'Parameter': 'CF p-value',          'Nilai': fmt_pval(pval_cf)   if pval_cf is not None else 'N/A'},
                            {'Parameter': 'CF Proporsi Aktual',  'Nilai': f"{phat_cf:.4f}"    if phat_cf is not None else 'N/A'},
                        ]
                    pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name='Summary')

                st.download_button(
                    label="💾 Download Hasil Perbandingan & Backtesting",
                    data=output_bt.getvalue(),
                    file_name=f"backtesting_{backtest_conf}pct_{backtest_hp}days.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

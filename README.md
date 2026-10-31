# INVESTA — Dashboard Manajemen Risiko Investasi

Aplikasi Streamlit untuk analisis risiko saham menggunakan **Value at Risk (VaR)** dan **Expected Shortfall (ES)** dengan metode **Variance–Covariance** serta koreksi **Cornish–Fisher**.

## Fitur
- Unduh harga saham Indonesia/internasional melalui Yahoo Finance (yfinance)
- Impor data Excel dan uji normalitas
- Analisis VaR dan ES parametrik (Variance–Covariance & Cornish–Fisher)
- Perbandingan hasil dan backtesting model
- Visualisasi harga, log return, dan ekspor hasil ke Excel

## Menjalankan secara lokal
1. Siapkan Python (disarankan 3.12).
2. Buka terminal pada folder ini, lalu jalankan:

```bash
python -m pip install -r requirements.txt
python -m streamlit run investa.py
```
## Catatan
- Tanggal unduhan harga saham menggunakan layanan yfinance/Yahoo Finance, sehingga memerlukan koneksi internet dan bergantung pada layanan pihak ketiga.

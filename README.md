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

## Deploy ke Streamlit Community Cloud
1. Buat repository GitHub publik, lalu upload `investa.py` dan `requirements.txt` ke direktori utama repository. `README.md` dan `.gitignore` juga boleh diikutkan.
2. Buka https://share.streamlit.io, login, kemudian pilih **Create app**.
3. Pilih repository, branch `main`, dan isi **Main file path**: `investa.py`.
4. Jika diperlukan, pada **Advanced settings** gunakan Python 3.12.
5. Klik **Deploy**. Setelah aplikasi aktif, uji menu **Download Data Saham**, **Upload & Uji Normalitas**, kalkulasi VaR/ES dan ekspor Excel.

## Catatan
- Tanggal unduhan harga saham menggunakan layanan yfinance/Yahoo Finance, sehingga memerlukan koneksi internet dan bergantung pada layanan pihak ketiga.
- Jangan upload data pribadi, kredensial, atau data internal perusahaan ke repository publik.
- Aplikasi ini untuk keperluan akademik dan edukasi, bukan rekomendasi investasi.

# Analisis & Prediksi Keterlambatan Pembayaran Transaksi Menggunakan Machine Learning

Project ini dibuat untuk tugas **Informatics Practitioner Lecture Series** dengan materi
**"Make Sense of Data with Analysis and AI"** (Pemateri: Andria Arisal, S.T., M.Eng. — BRIN).

---

## 1. Latar Belakang

Dalam pengelolaan transaksi/invoice, keterlambatan pembayaran pelanggan berdampak
langsung pada arus kas (*cash flow*) dan risiko piutang. Sebagian besar perusahaan hanya
mengetahui keterlambatan *setelah* kejadian. Padahal, jika pola data historis dianalisis,
potensi keterlambatan bisa diprediksi lebih awal.

## 2. Permasalahan

Bagaimana memanfaatkan data transaksi untuk **memprediksi apakah sebuah invoice berpotensi
terlambat dibayar**, dan bagaimana mengomunikasikan hasil tersebut sehingga bisa dipakai
untuk pengambilan keputusan?

## 3. Tujuan

1. Melakukan analisis kualitas data transaksi (missing value, duplikat, tipe data, statistik).
2. Melakukan preprocessing dan feature engineering tanpa data leakage.
3. Melatih dan membandingkan 3 model: Logistic Regression, Decision Tree, Random Forest.
4. Mengevaluasi model dengan Accuracy, Precision, Recall, F1, dan ROC-AUC.
5. Menyediakan aplikasi web (Streamlit) untuk prediksi transaksi baru + probabilitasnya.

## 4. Kaitan dengan Materi Pemateri

| Materi | Implementasi di Project |
|---|---|
| Data → Information → Insight → Decision | Data transaksi → metrik & grafik → probabilitas keterlambatan → dashboard prediksi |
| CRISP-DM | Alur `train.py`: Data Understanding → Data Preparation → Modeling → Evaluation |
| Data Understanding | `dataset_quality_report()` (baris, kolom, missing, duplikat, tipe data, statistik) |
| Data Preparation | `clean_data()` + `add_features()` + `ColumnTransformer` (imputer, scaler, one-hot) |
| Data Quality | Pemeriksaan missing value, duplikat, dan konsistensi tipe |
| Missing Value | `SimpleImputer` di dalam pipeline (median / most frequent) |
| Duplicate | `drop_duplicates()` pada tahap cleaning |
| Data Leakage | Kolom pasca-kejadian dibuang (`check_leakage_columns()`), daftar leakage dijaga eksplisit |
| Class Imbalance | `class_weight="balanced"`; seleksi model memakai ROC-AUC, bukan accuracy |
| Logistic Regression / Decision Tree / Random Forest | `src/train_model.py` |
| Evaluasi model | `src/evaluate.py`: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve |
| AI sebagai alat bantu analisis | Aplikasi Streamlit yang memberi prediksi + interpretasi singkat |

## 5. Dataset

Dataset **sintetis** yang dibuat oleh `generate_dataset.py` (3.000 baris) karena dataset
transaksi asli tidak disediakan. Distribusi target dibuat tidak seimbang (sekitar 27,6%
terlambat) agar menyerupai kasus nyata.

Kolom dataset:

| Kolom | Tipe | Keterangan |
|---|---|---|
| invoice_id | str | Identitas invoice (tidak dipakai sebagai fitur) |
| invoice_date | date | Tanggal invoice |
| due_date | date | Tanggal jatuh tempo |
| customer_id | str | Identitas pelanggan (tidak dipakai sebagai fitur) |
| customer_type | kategorikal | Retail / Corporate / SME / Government |
| product_category | kategorikal | Kategori produk |
| payment_method | kategorikal | Metode pembayaran |
| invoice_amount | numerik | Nilai invoice |
| discount | numerik | Diskon (%) |
| previous_late_payment | biner | Riwayat keterlambatan sebelumnya |
| customer_transaction_count | numerik | Jumlah transaksi pelanggan |
| customer_avg_payment_delay | numerik | Rata-rata keterlambatan historis (hari) |
| late_payment | target | 0 = tidak terlambat, 1 = terlambat |

**Anti data leakage.** Kolom yang baru diketahui setelah pembayaran (mis. `payment_date`,
`actual_payment_date`, `overdue_days`) sengaja **tidak** dibuat dan tidak digunakan.

## 6. Metodologi (CRISP-DM)

1. **Data Understanding** — load CSV + quality report.
2. **Data Preparation** — buang duplikat, buang kolom leakage, feature engineering:
   `payment_term_days` (due_date − invoice_date), `invoice_month`, `invoice_weekday`.
3. **Modeling** — train-test split **stratified** (80/20, `random_state=42`), lalu latih
   3 model di dalam `Pipeline` (preprocessing + model).
4. **Evaluation** — bandingkan metrik, pilih model terbaik berdasarkan **ROC-AUC**.
5. **Deployment** — simpan model (`models/best_model.pkl`) + aplikasi Streamlit.

## 7. Preprocessing

- Missing value: imputasi median (numerik) & modus (kategorikal).
- Kategorikal: `OneHotEncoder(handle_unknown="ignore")`.
- Numerik: `StandardScaler`.
- Semua transformasi berada di dalam `Pipeline`, sehingga otomatis diterapkan sama
  pada data training dan data prediksi baru.

## 8. Model

| Model | Catatan |
|---|---|
| Logistic Regression | baseline, `max_iter=1000`, `class_weight="balanced"` |
| Decision Tree | `max_depth=8`, `min_samples_leaf=20`, `class_weight="balanced"` |
| Random Forest | `n_estimators=300`, `max_depth=12`, `min_samples_leaf=5`, `class_weight="balanced"` |

## 9. Evaluasi

Metrik dihitung dari hasil prediksi pada data test (600 baris). Hasil aktual
(dari `results/training_summary.json`, akan berubah jika dataset/parameter diubah):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.6983 | 0.4670 | 0.6386 | 0.5394 | **0.7361** |
| Random Forest | 0.7100 | 0.4815 | 0.6265 | 0.5445 | 0.7220 |
| Decision Tree | 0.6483 | 0.4082 | 0.6024 | 0.4866 | 0.6665 |

- **Model terbaik: Logistic Regression** (ROC-AUC = 0.7361).
- Confusion matrix (baris = aktual, kolom = prediksi):
  - Tidak terlambat: 313 benar, 121 salah
  - Terlambat: 60 salah, 106 benar
- Alasan memakai ROC-AUC: data tidak seimbang, sehingga accuracy bisa menyesatkan.

## 10. Cara Instalasi

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## 11. Cara Menjalankan Training

```bash
# 1) Buat dataset sintetis (3.000 baris)
python generate_dataset.py

# 2) Jalankan seluruh pipeline training + evaluasi
python train.py
```

Output: `models/best_model.pkl`, `results/model_comparison.csv`,
`results/confusion_matrix.png`, `results/roc_curve.png`, `results/training_summary.json`.

## 12. Cara Menjalankan Aplikasi

```bash
streamlit run app.py
```

Halaman: **Dashboard**, **Dataset**, **Model Performance**, **Prediction**.

## 13. Contoh Penggunaan Prediksi

```python
from src.predict import LatePaymentPredictor

predictor = LatePaymentPredictor("models/best_model.pkl")
hasil = predictor.predict({
    "customer_type": "SME",
    "product_category": "Electronics",
    "payment_method": "Installment",
    "invoice_amount": 50_000_000,
    "discount": 5,
    "previous_late_payment": 1,
    "customer_transaction_count": 15,
    "customer_avg_payment_delay": 20,
})
print(hasil["label"], f"{hasil['probability']*100:.2f}%")
```

## 14. Struktur Project

```
late-payment-prediction/
├── app.py
├── train.py
├── generate_dataset.py
├── requirements.txt
├── README.md
├── .gitignore
├── dataset/transactions.csv
├── models/best_model.pkl
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── model_comparison.csv
│   └── training_summary.json
└── src/
    ├── __init__.py
    ├── preprocessing.py
    ├── train_model.py
    ├── evaluate.py
    └── predict.py
```

## 15. Kesimpulan

Project ini menunjukkan alur analitik prediktif dari materi pemateri: dari data mentah →
pemeriksaan kualitas → preprocessing → pemodelan → evaluasi → aplikasi yang dapat dipakai.
Hasil menunjukkan bahwa keterlambatan pembayaran **dapat** diprediksi dari fitur riwayat
dan karakteristik transaksi, dengan Logistic Regression sebagai model terbaik pada data ini.

## 16. Pengembangan Selanjutnya

- Menggunakan dataset nyata (bukan sintetis).
- Menambahkan hyperparameter tuning (GridSearch/RandomizedSearch).
- Menambahkan SHAP untuk explainability.
- Mengintegrasikan database dan penjadwalan batch scoring.
- Menambahkan kalibrasi probabilitas.

## 17. Upload ke GitHub

```bash
git init
git add .
git commit -m "Initial implementation: late payment prediction"
git branch -M main
git remote add origin [GITHUB_REPOSITORY]
git push -u origin main
```

> `dataset/*.csv` dan `models/*.pkl` di-ignore karena bisa di-generate ulang.
> Jalankan `python generate_dataset.py` dan `python train.py` setelah clone.

## Identitas Mahasiswa

- Nama  : Reffki Andrea Pratama
- NPM   : G1A023039
- Prodi : Informatika
- Universitas : Universitas Bengkulu


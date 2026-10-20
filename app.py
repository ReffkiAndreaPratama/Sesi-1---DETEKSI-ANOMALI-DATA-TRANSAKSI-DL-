"""Aplikasi Streamlit: Analisis & Prediksi Keterlambatan Pembayaran.

Jalankan:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import INPUT_FIELDS, LatePaymentPredictor

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "transactions.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

CUSTOMER_TYPES = ["Retail", "Corporate", "SME", "Government"]
PRODUCT_CATEGORIES = ["Electronics", "Fashion", "Groceries", "Furniture", "Services"]
PAYMENT_METHODS = ["Bank Transfer", "Credit Card", "E-Wallet", "Cash on Delivery", "Installment"]


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df if not df.empty else None


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_predictor(model_path: str) -> LatePaymentPredictor | None:
    try:
        return LatePaymentPredictor(model_path)
    except FileNotFoundError:
        return None


def page_dashboard(df: pd.DataFrame | None, summary: dict) -> None:
    st.header("Dashboard")
    if df is None:
        st.error("Dataset tidak ditemukan. Jalankan: python generate_dataset.py")
        return

    total = len(df)
    late = int(df["late_payment"].sum())
    pct = late / total * 100 if total else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transaksi", f"{total:,}")
    c2.metric("Transaksi Terlambat", f"{late:,}")
    c3.metric("Persentase Keterlambatan", f"{pct:.2f}%")

    if summary:
        c4, c5 = st.columns(2)
        c4.metric("Model Terbaik", summary.get("best_model", "-"))
        c5.metric(
            "ROC-AUC Model Terbaik",
            f"{summary.get('best_model_metrics', {}).get('roc_auc', 0):.4f}",
        )
    else:
        st.info("Model belum dilatih. Jalankan: python train.py")

    st.subheader("Distribusi Target")
    st.bar_chart(pd.DataFrame({"jumlah": [total - late, late]}, index=["Tidak Terlambat", "Terlambat"]))


def page_dataset(df: pd.DataFrame | None) -> None:
    st.header("Dataset")
    if df is None:
        st.error("Dataset tidak tersedia atau kosong.")
        return

    st.subheader("Tabel Data")
    n = st.slider("Jumlah baris ditampilkan", 5, min(200, len(df)), min(20, len(df)))
    st.dataframe(df.head(n), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Missing Value")
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("Tidak ada missing value.")
        else:
            st.dataframe(missing.rename("jumlah_missing"))
    with c2:
        st.subheader("Distribusi Target")
        st.dataframe(df["late_payment"].value_counts().sort_index().rename("jumlah"))

    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe(include="all").transpose(), width="stretch")


def page_model_performance() -> None:
    st.header("Model Performance")
    comparison_path = RESULTS_DIR / "model_comparison.csv"
    if not comparison_path.exists():
        st.warning("Belum ada hasil training. Jalankan: python train.py")
        return

    comparison = pd.read_csv(comparison_path)
    st.subheader("Perbandingan Model")
    st.dataframe(comparison, width="stretch")

    summary = load_json(str(RESULTS_DIR / "training_summary.json"))
    if summary:
        st.success(f"Model terbaik: **{summary.get('best_model', '-')}** "
                   f"(ROC-AUC = {summary.get('best_model_metrics', {}).get('roc_auc', 0):.4f})")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm_path = RESULTS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), width="stretch")
        else:
            st.info("Confusion matrix belum tersedia.")
    with col2:
        st.subheader("ROC Curve")
        roc_path = RESULTS_DIR / "roc_curve.png"
        if roc_path.exists():
            st.image(str(roc_path), width="stretch")
        else:
            st.info("ROC curve belum tersedia.")


def page_prediction(predictor: LatePaymentPredictor | None) -> None:
    st.header("Prediction")
    st.caption("Masukkan data transaksi untuk memprediksi potensi keterlambatan pembayaran.")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            customer_type = st.selectbox("Customer Type", CUSTOMER_TYPES)
            product_category = st.selectbox("Product Category", PRODUCT_CATEGORIES)
            payment_method = st.selectbox("Payment Method", PAYMENT_METHODS)
            invoice_amount = st.number_input("Invoice Amount", min_value=0.0, value=5_000_000.0, step=100_000.0)
        with c2:
            discount = st.slider("Discount (%)", min_value=0.0, max_value=30.0, value=0.0, step=0.5)
            previous_late_payment = st.selectbox("Previous Late Payment", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
            customer_transaction_count = st.number_input("Customer Transaction Count", min_value=1, value=10, step=1)
            customer_avg_payment_delay = st.number_input("Customer Average Payment Delay (hari)", min_value=0.0, value=5.0, step=0.5)

        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    if predictor is None:
        st.error("Model belum tersedia. Jalankan dulu: python train.py")
        return

    record = {
        "customer_type": customer_type,
        "product_category": product_category,
        "payment_method": payment_method,
        "invoice_amount": float(invoice_amount),
        "discount": float(discount),
        "previous_late_payment": int(previous_late_payment),
        "customer_transaction_count": int(customer_transaction_count),
        "customer_avg_payment_delay": float(customer_avg_payment_delay),
    }

    result = predictor.predict(record)
    prob_pct = result["probability"] * 100

    if result["prediction"] == 1:
        st.error(f"Prediction: **{result['label']}**")
    else:
        st.success(f"Prediction: **{result['label']}**")

    st.metric("Probability of Late Payment", f"{prob_pct:.2f}%")
    st.progress(min(max(result["probability"], 0.0), 1.0))
    st.write(result["interpretation"])


def main() -> None:
    st.set_page_config(page_title="Prediksi Keterlambatan Pembayaran", page_icon="â°", layout="wide")
    st.title("Analisis & Prediksi Keterlambatan Pembayaran")
    st.caption("Studi kasus Machine Learning - CRISP-DM (Logistic Regression, Decision Tree, Random Forest)")

    df = load_dataset(str(DATASET_PATH))
    summary = load_json(str(RESULTS_DIR / "training_summary.json"))
    predictor = load_predictor(str(MODELS_DIR / "best_model.pkl"))

    page = st.sidebar.radio("Navigasi", ["Dashboard", "Dataset", "Model Performance", "Prediction"])
    st.sidebar.markdown("---")
    st.sidebar.caption("Jalankan `python train.py` untuk melatih model.")

    if page == "Dashboard":
        page_dashboard(df, summary)
    elif page == "Dataset":
        page_dataset(df)
    elif page == "Model Performance":
        page_model_performance()
    else:
        page_prediction(predictor)


if __name__ == "__main__":
    main()


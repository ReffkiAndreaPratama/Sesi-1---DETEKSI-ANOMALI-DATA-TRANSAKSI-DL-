"""Script utama training pipeline.

Alur (mengikuti CRISP-DM pada tahap modeling & evaluation):
1. Load dataset
2. Data quality checking
3. Cleaning
4. Feature engineering
5. Train-test split (stratified)
6. Training 3 model (Logistic Regression, Decision Tree, Random Forest)
7. Evaluasi & perbandingan
8. Simpan model terbaik + artefak hasil

Jalankan:
    python train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.metrics import confusion_matrix

from src.evaluate import save_comparison_csv, save_confusion_matrix, save_roc_curve
from src.preprocessing import (
    TARGET_COLUMN,
    add_features,
    check_leakage_columns,
    clean_data,
    dataset_quality_report,
    get_feature_target,
    load_dataset,
)
from src.train_model import compute_metrics, select_best_model, split_data, train_all_models

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "transactions.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RANDOM_STATE = 12345


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TRAINING PIPELINE - PREDIKSI KETERLAMBATAN PEMBAYARAN")
    print("=" * 60)

    # 1. Load + 2. Data quality checking
    df = load_dataset(DATASET_PATH)
    report = dataset_quality_report(df)
    print("\n[1] DATA QUALITY")
    print(f"    Baris/kolom      : {report['n_rows']} / {report['n_columns']}")
    print(f"    Total missing    : {report['total_missing']}")
    print(f"    Duplikat         : {report['duplicated_rows']}")
    print(f"    Distribusi target: {report.get('target_distribution')}")

    leakage = check_leakage_columns(df)
    if leakage:
        print(f"    [WARN] Kolom leakage terdeteksi dan dibuang: {leakage}")

    # 3. Cleaning
    df = clean_data(df)

    # 4. Feature engineering
    df = add_features(df)

    X, y = get_feature_target(df)
    print(f"\n[2] FITUR ({len(X.columns)}): {list(X.columns)}")
    print(f"    Rasio kelas positif: {y.mean():.3f}")

    # 5. Split
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=RANDOM_STATE)

    # 6. Training
    print("\n[3] TRAINING MODEL ...")
    trained_models, results = train_all_models(X_train, y_train, X_test, y_test)

    # 7. Evaluasi
    print("\n[4] HASIL EVALUASI (test set)")
    comparison = save_comparison_csv(results, RESULTS_DIR / "model_comparison.csv")
    print(comparison.to_string(index=False))

    best_name = select_best_model(results, metric="roc_auc")
    best_model = trained_models[best_name]
    print(f"\n    Model terbaik: {best_name} (ROC-AUC = {results[best_name]['roc_auc']:.4f})")

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    best_metrics = compute_metrics(y_test, y_pred, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    # 8. Simpan artefak
    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"\n[5] Model disimpan: {model_path}")

    save_confusion_matrix(
        y_test,
        y_pred,
        RESULTS_DIR / "confusion_matrix.png",
        title=f"Confusion Matrix - {best_name}",
    )
    save_roc_curve(trained_models, X_test, y_test, RESULTS_DIR / "roc_curve.png")

    summary = {
        "best_model": best_name,
        "selection_metric": "roc_auc",
        "best_model_metrics": best_metrics,
        "confusion_matrix": cm.tolist(),
        "all_models": results,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_balance": float(y.mean()),
        "features": list(X.columns),
        "target_distribution": report.get("target_distribution"),
    }
    with open(RESULTS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Ringkasan training: {RESULTS_DIR / 'training_summary.json'}")
    print("\nSelesai. Jalankan aplikasi dengan: streamlit run app.py")


if __name__ == "__main__":
    main()

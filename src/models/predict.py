from __future__ import annotations

import argparse
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "issue_category_model.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Bir GitHub issue'sunu dört ana kategoriden birine tahmin eder.")
    parser.add_argument("--title", required=True, help="Issue başlığı")
    parser.add_argument("--body", default="", help="Issue açıklaması (isteğe bağlı)")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bulunamadı. Önce eğitim komutunu çalıştırın: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    prediction = model.predict([f"{args.title.strip()}\n{args.body.strip()}"])[0]
    print(f"Tahmin edilen kategori: {prediction}")


if __name__ == "__main__":
    main()

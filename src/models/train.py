from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "github_issues_processed.csv"
DEFAULT_TEST_IDS = PROJECT_ROOT / "data" / "processed" / "model_test_ids.csv"
CATEGORIES = ["Documentation", "Bug", "Question", "Feature Request"]


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2, max_df=0.98, max_features=50_000, sublinear_tf=True)),
        ("classifier", LinearSVC(class_weight="balanced", random_state=42)),
    ])


def log_with_mlflow(run_name: str, metrics: dict, model_path: Path, metrics_path: Path, data_path: Path) -> None:
    try:
        import mlflow
    except ImportError:
        print("MLflow kurulu değil; yerel metrik dosyası kaydedildi, deney kaydı atlandı.")
        return

    tracking_db = PROJECT_ROOT / "mlflow" / "tracking.db"
    tracking_db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    mlflow.set_experiment("GitHub Issue Triage AI")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("classifier", "LinearSVC")
        mlflow.log_param("vectorizer", "TF-IDF word uni/bigrams")
        mlflow.log_param("training_rows", metrics["training_rows"])
        mlflow.log_param("test_rows", metrics["test_rows"])
        mlflow.log_param("data_file", str(data_path))
        mlflow.log_metric("accuracy", metrics["holdout_accuracy"])
        mlflow.log_metric("macro_f1", metrics["holdout_macro_f1"])
        for category in CATEGORIES:
            mlflow.log_metric(f"f1_{category.lower().replace(' ', '_')}", metrics["classification_report"][category]["f1-score"])
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dört kategorili issue sınıflandırıcısını eğitir ve ölçer.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-output", type=Path, default=PROJECT_ROOT / "models" / "issue_category_model.joblib")
    parser.add_argument("--metrics-output", type=Path, default=PROJECT_ROOT / "data" / "processed" / "model_evaluation.json")
    parser.add_argument("--test-ids", type=Path, default=DEFAULT_TEST_IDS)
    parser.add_argument("--run-name", default="expanded-data")
    args = parser.parse_args()

    df = pd.read_csv(args.data, keep_default_na=False, dtype={"issue_id": "string"})
    required = {"issue_id", "title", "body", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Veri dosyasında gerekli sütunlar eksik: {', '.join(sorted(missing))}")
    data = df.loc[df["category"].isin(CATEGORIES)].copy()
    data["text"] = data["title"].astype(str).str.strip() + "\n" + data["body"].astype(str).str.strip()
    data = data.loc[data["text"].str.strip().ne("")].reset_index(drop=True)

    if args.test_ids.exists():
        test_ids = set(pd.read_csv(args.test_ids, dtype={"issue_id": "string"})["issue_id"].astype(str))
        test_mask = data["issue_id"].astype(str).isin(test_ids)
        test_data = data.loc[test_mask]
        train_data = data.loc[~test_mask]
        missing_test_ids = test_ids - set(test_data["issue_id"].astype(str))
        if missing_test_ids:
            raise ValueError(f"Sabit test kümesinde veri dosyasında bulunmayan {len(missing_test_ids)} issue_id var.")
        if test_data.empty or train_data.empty:
            raise ValueError("Sabit test kümesi eğitim/test ayrımı oluşturamadı.")
    else:
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data["category"])

    eval_model = build_pipeline()
    eval_model.fit(train_data["text"], train_data["category"])
    predictions = eval_model.predict(test_data["text"])
    report = classification_report(test_data["category"], predictions, labels=CATEGORIES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(test_data["category"], predictions, labels=CATEGORIES)

    # Evaluate without training on the fixed test IDs, then refit the saved model on all eligible rows.
    final_model = build_pipeline()
    final_model.fit(data["text"], data["category"])
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, args.model_output)

    metrics = {
        "data_file": str(args.data.resolve()),
        "training_rows_for_evaluation": int(len(train_data)),
        "training_rows": int(len(data)),
        "excluded_rows": int(len(df) - len(data)),
        "category_counts": {category: int((data["category"] == category).sum()) for category in CATEGORIES},
        "test_rows": int(len(test_data)),
        "holdout_accuracy": float(accuracy_score(test_data["category"], predictions)),
        "holdout_macro_f1": float(f1_score(test_data["category"], predictions, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix_labels": CATEGORIES,
        "confusion_matrix": matrix.tolist(),
        "model_file": str(args.model_output.resolve()),
        "test_ids_file": str(args.test_ids.resolve()) if args.test_ids.exists() else None,
        "feature_fields": ["title", "body"],
    }
    args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Deney: {args.run_name}")
    print(f"Eğitimde kullanılan kayıt: {len(data)} (test ölçümü için ayrılan dışındaki eğitim: {len(train_data)})")
    print(f"Test kümesi: {len(test_data)}")
    print(f"Test doğruluğu: {metrics['holdout_accuracy']:.3f}")
    print(f"Test macro F1: {metrics['holdout_macro_f1']:.3f}")
    for category in CATEGORIES:
        row = report[category]
        print(f"  {category}: precision={row['precision']:.3f}, recall={row['recall']:.3f}, F1={row['f1-score']:.3f}")
    print(f"Model: {args.model_output.resolve()}")
    print(f"Ölçüm: {args.metrics_output.resolve()}")
    log_with_mlflow(args.run_name, metrics, args.model_output, args.metrics_output, args.data)


if __name__ == "__main__":
    main()

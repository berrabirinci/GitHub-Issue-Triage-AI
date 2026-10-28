from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_TEST = PROJECT_ROOT / "data" / "test" / "raw" / "github_issues_external_test.csv"
PROCESSED_TEST = PROJECT_ROOT / "data" / "test" / "processed" / "github_issues_external_test.csv"
REVIEW_TEST = PROJECT_ROOT / "data" / "test" / "processed" / "github_issues_external_test_review.csv"
EVALUATION = PROJECT_ROOT / "data" / "test" / "processed" / "model_external_test_evaluation.json"
MODEL_PATH = PROJECT_ROOT / "models" / "issue_category_model.joblib"
CATEGORIES = ["Documentation", "Bug", "Question", "Feature Request"]

# Existing exact mappings plus labels that state the category directly.
LABEL_MAPPING = {
    "bug": "Bug",
    "documentation": "Documentation",
    "type: docs": "Documentation",
    "docs": "Documentation",
    "new feature": "Feature Request",
    "enhancement": "Feature Request",
    "feature request": "Feature Request",
    "type: enhancement": "Feature Request",
    "question": "Question",
    "type: question": "Question",
    "question/not a bug": "Question",
}


def prepare_test_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RAW_TEST, dtype={"issue_id": "string"}, keep_default_na=False)
    if raw["issue_id"].duplicated().any():
        raise ValueError("Dış test CSV'sinde yinelenen issue_id bulundu.")

    train_ids: set[str] = set()
    for path in (
        PROJECT_ROOT / "data" / "raw" / "github_issues_all.csv",
        PROJECT_ROOT / "data" / "raw" / "github_issues_additional.csv",
        PROJECT_ROOT / "data" / "raw" / "github_issues_category_expansion.csv",
    ):
        if path.exists():
            source = pd.read_csv(path, dtype={"issue_id": "string"}, keep_default_na=False)
            train_ids.update(source["issue_id"].astype(str))
    overlap = set(raw["issue_id"].astype(str)) & train_ids
    if overlap:
        raise ValueError(f"Eğitim verisiyle ortak {len(overlap)} issue_id bulundu; değerlendirme durduruldu.")

    records: list[dict[str, object]] = []
    for row in raw.to_dict("records"):
        labels = [part.strip() for part in str(row["labels"]).split(";") if part.strip()]
        categories = sorted({LABEL_MAPPING[label.casefold()] for label in labels if label.casefold() in LABEL_MAPPING})
        reasons: list[str] = []
        if len(categories) > 1:
            reasons.append("ambiguous_multiple_categories")
        elif not categories:
            reasons.append("unmapped_or_unlabeled")
        if not str(row["title"]).strip():
            reasons.append("missing_title_low_quality")
        records.append({
            **row,
            "labels_normalized": ";".join(labels),
            "mapped_categories": ";".join(categories),
            "category": categories[0] if len(categories) == 1 else "",
            "review_required": bool(reasons),
            "review_reason": ";".join(reasons),
        })

    prepared = pd.DataFrame(records)
    scored = prepared.loc[~prepared["review_required"] & prepared["category"].ne("")].copy()
    review = prepared.loc[prepared["review_required"]].copy()
    if scored.empty:
        raise ValueError("Güvenilir kategori eşlemesi olan dış test issue'su bulunamadı.")
    scored.to_csv(PROCESSED_TEST, index=False, encoding="utf-8-sig")
    review.to_csv(REVIEW_TEST, index=False, encoding="utf-8-sig")
    return raw, scored, review


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Eğitilmiş model bulunamadı: {MODEL_PATH}")
    raw, test, review = prepare_test_data()
    test["text"] = test["title"].astype(str).str.strip() + "\n" + test["body"].astype(str).str.strip()
    model = joblib.load(MODEL_PATH)
    predictions = model.predict(test["text"])
    report = classification_report(test["category"], predictions, labels=CATEGORIES, output_dict=True, zero_division=0)
    metrics = {
        "model_file": str(MODEL_PATH.resolve()),
        "raw_test_file": str(RAW_TEST.resolve()),
        "scored_test_file": str(PROCESSED_TEST.resolve()),
        "review_file": str(REVIEW_TEST.resolve()),
        "source_repositories": sorted(raw["repository"].unique().tolist()),
        "raw_issue_count": int(len(raw)),
        "scored_issue_count": int(len(test)),
        "review_issue_count": int(len(review)),
        "training_id_overlap": 0,
        "label_mappings_used": LABEL_MAPPING,
        "category_counts": {category: int((test["category"] == category).sum()) for category in CATEGORIES},
        "accuracy": float(accuracy_score(test["category"], predictions)),
        "macro_f1": float(f1_score(test["category"], predictions, labels=CATEGORIES, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix_labels": CATEGORIES,
        "confusion_matrix": confusion_matrix(test["category"], predictions, labels=CATEGORIES).tolist(),
        "interpretation_note": "Dış repo testidir. Yalnızca güvenilir, tek kategoriye eşleşen kayıtlar puanlandı; sonuç bu örneklemin kategori dağılımına göre yorumlanmalıdır.",
    }
    EVALUATION.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Ham dış issue: {len(raw)}")
    print(f"Puanlanan issue: {len(test)}")
    print(f"İnceleme kuyruğu: {len(review)}")
    print(f"Sınıf dağılımı: {metrics['category_counts']}")
    print(f"Doğruluk: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    for category in CATEGORIES:
        row = report[category]
        print(f"{category}: n={metrics['category_counts'][category]}, precision={row['precision']:.3f}, recall={row['recall']:.3f}, F1={row['f1-score']:.3f}")
    print(f"Ölçüm dosyası: {EVALUATION.resolve()}")


if __name__ == "__main__":
    main()




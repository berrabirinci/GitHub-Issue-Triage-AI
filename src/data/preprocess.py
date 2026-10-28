from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Exact GitHub-label matches only. Label comparison is case-insensitive and
# whitespace-trimmed; repository-specific/module labels are intentionally ignored.
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
}

REQUIRED_COLUMNS = {"repository", "issue_id", "title", "body", "labels", "issue_url", "created_at"}


def parse_labels(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def prepare(input_paths: list[Path], processed_path: Path, review_path: Path) -> dict[str, object]:
    frames = [pd.read_csv(path, keep_default_na=False, dtype={"issue_id": "string"}) for path in input_paths if path.exists()]
    if not frames:
        raise FileNotFoundError("İşlenecek CSV dosyası bulunamadı.")
    df = pd.concat(frames, ignore_index=True)
    if "issue_id" in df.columns:
        df = df.drop_duplicates(subset=["issue_id"], keep="first").reset_index(drop=True)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Gerekli sütunlar eksik: {', '.join(sorted(missing))}")

    original_columns = list(df.columns)
    df["labels_normalized"] = df["labels"].apply(parse_labels)
    df["mapped_categories"] = df["labels_normalized"].apply(
        lambda labels: sorted({LABEL_MAPPING[label.casefold()] for label in labels if label.casefold() in LABEL_MAPPING})
    )
    df["category"] = df["mapped_categories"].apply(lambda categories: categories[0] if len(categories) == 1 else "")

    reasons: list[str] = []
    for row in df.itertuples(index=False):
        categories = row.mapped_categories
        row_reasons = []
        if len(categories) > 1:
            row_reasons.append("ambiguous_multiple_categories")
        elif len(categories) == 0:
            row_reasons.append("unmapped_or_unlabeled")
        title = str(getattr(row, "title", "")).strip()
        body = str(getattr(row, "body", "")).strip()
        if not title:
            row_reasons.append("missing_title_low_quality")
        if not title and not body:
            row_reasons.append("missing_title_and_body_low_quality")
        reasons.append(";".join(row_reasons))

    df["review_required"] = [bool(reason) for reason in reasons]
    df["review_reason"] = reasons
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)

    export = df.copy()
    export["labels_normalized"] = export["labels_normalized"].apply(lambda values: ";".join(values))
    export["mapped_categories"] = export["mapped_categories"].apply(lambda values: ";".join(values))
    export.to_csv(processed_path, index=False, encoding="utf-8")
    review = export.loc[export["review_required"]].copy()
    review.to_csv(review_path, index=False, encoding="utf-8")

    mapped = df["category"].ne("")
    reasons_series = pd.Series(reasons)
    return {
        "total": len(df),
        "mapped": int(mapped.sum()),
        "review": int(df["review_required"].sum()),
        "ambiguous": int(reasons_series.str.contains("ambiguous_multiple_categories").sum()),
        "unmapped": int(reasons_series.str.contains("unmapped_or_unlabeled").sum()),
        "missing_title": int(reasons_series.str.contains("missing_title_low_quality").sum()),
        "category_counts": df.loc[mapped, "category"].value_counts().to_dict(),
        "processed_path": str(processed_path.resolve()),
        "review_path": str(review_path.resolve()),
        "source_columns": original_columns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ham GitHub issue CSV'lerini birleştirip güvenli biçimde işle.")
    parser.add_argument("--input", type=Path, default=None, help="İsteğe bağlı tek CSV. Verilmezse base ve additional dosyaları birleştirir.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/github_issues_processed.csv"))
    parser.add_argument("--review-output", type=Path, default=Path("data/processed/github_issues_review_full.csv"))
    args = parser.parse_args()
    if args.input:
        input_paths = [args.input]
    else:
        input_paths = [Path("data/raw/github_issues_all.csv"), Path("data/raw/github_issues_additional.csv"), Path("data/raw/github_issues_category_expansion.csv")]
    summary = prepare(input_paths, args.output, args.review_output)
    print("VERİ HAZIRLAMA ÖZETİ")
    print(f"Toplam kayıt: {summary['total']}")
    print(f"Tek ve açık kategoriye eşlenen: {summary['mapped']}")
    print(f"İnceleme kuyruğu: {summary['review']}")
    print(f"Birden fazla kategori (belirsiz): {summary['ambiguous']}")
    print(f"Eşleşmeyen/etiketsiz: {summary['unmapped']}")
    print(f"Başlığı eksik (düşük kalite): {summary['missing_title']}")
    print("Kategori sayıları:")
    for category, count in sorted(summary["category_counts"].items()):
        print(f"  {category}: {count}")
    print(f"İşlenmiş veri: {summary['processed_path']}")
    print(f"İnceleme dosyası: {summary['review_path']}")


if __name__ == "__main__":
    main()




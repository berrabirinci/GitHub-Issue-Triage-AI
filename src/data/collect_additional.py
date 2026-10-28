from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

REPOSITORIES = [
    ("fastapi", "fastapi"),
    ("pandas-dev", "pandas"),
    ("scikit-learn", "scikit-learn"),
]
TARGET_PER_REPOSITORY = 1000
PER_PAGE = 100
SEARCH_URL = "https://api.github.com/search/issues"
COLUMNS = ["repository", "issue_id", "title", "body", "labels", "issue_url", "created_at"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BASE_DATASET = RAW_DIR / "github_issues_all.csv"  # Read only.
ADDITIONAL_DATASET = RAW_DIR / "github_issues_additional.csv"
UNLABELED_DATASET = PROCESSED_DIR / "github_issues_unlabeled.csv"


def read_csv_if_present(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype={"issue_id": "string"}, keep_default_na=False)
    for column in COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[COLUMNS]


def request_search(session: requests.Session, query: str, page: int) -> dict:
    while True:
        response = session.get(
            SEARCH_URL,
            params={"q": query, "per_page": PER_PAGE, "page": page, "sort": "created", "order": "desc"},
            timeout=30,
        )
        if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
            reset_at = int(response.headers.get("X-RateLimit-Reset", "0"))
            wait_seconds = max(1, reset_at - int(time.time()) + 1)
            print(f"GitHub arama sınırı doldu; {wait_seconds} saniye bekleniyor.", flush=True)
            time.sleep(wait_seconds)
            continue
        response.raise_for_status()
        return response.json()


def fetch_issues() -> pd.DataFrame:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    collected: list[dict[str, str]] = []
    with requests.Session() as session:
        session.headers.update(headers)
        for owner, repo in REPOSITORIES:
            query = f"repo:{owner}/{repo} is:issue"
            repo_rows: list[dict[str, str]] = []
            print(f"Alınıyor: {owner}/{repo}", flush=True)
            total_available = None
            page = 1
            while len(repo_rows) < TARGET_PER_REPOSITORY:
                result = request_search(session, query, page)
                if total_available is None:
                    total_available = result.get("total_count", 0)
                    print(f"  Bulunan toplam issue: {total_available}", flush=True)
                items = result.get("items", [])
                if not items:
                    break
                for issue in items:
                    repository = issue.get("repository_url", "").rstrip("/").split("/repos/")[-1]
                    repo_rows.append({
                        "repository": repository or f"{owner}/{repo}",
                        "issue_id": str(issue["id"]),
                        "title": issue.get("title") or "",
                        "body": issue.get("body") or "",
                        "labels": ";".join(label.get("name", "") for label in issue.get("labels", [])),
                        "issue_url": issue.get("html_url", ""),
                        "created_at": issue.get("created_at", ""),
                    })
                    if len(repo_rows) >= TARGET_PER_REPOSITORY:
                        break
                page += 1
                if page > 10:  # Search API returns at most 1,000 results per query.
                    break
            print(f"  Çekilen issue: {len(repo_rows)}", flush=True)
            collected.extend(repo_rows)
    return pd.DataFrame(collected, columns=COLUMNS)


def main() -> None:
    base = read_csv_if_present(BASE_DATASET)
    additional = read_csv_if_present(ADDITIONAL_DATASET)
    incoming = fetch_issues()

    known_ids = set(base["issue_id"].astype(str)) | set(additional["issue_id"].astype(str))
    new_rows = incoming.loc[~incoming["issue_id"].astype(str).isin(known_ids)].copy()
    merged_additional = pd.concat([additional, new_rows], ignore_index=True)
    merged_additional = merged_additional.drop_duplicates(subset=["issue_id"], keep="first")

    # Keep the original raw snapshot unchanged; append new IDs to a separate file.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    merged_additional.to_csv(ADDITIONAL_DATASET, index=False, encoding="utf-8-sig")

    combined = pd.concat([base, merged_additional], ignore_index=True)
    combined = combined.drop_duplicates(subset=["issue_id"], keep="first")
    unlabeled = combined.loc[combined["labels"].fillna("").str.strip().eq("")].copy()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    unlabeled.to_csv(UNLABELED_DATASET, index=False, encoding="utf-8-sig")

    new_unlabeled = int(new_rows["labels"].fillna("").str.strip().eq("").sum())
    print("\nTOPLAMA ÖZETİ")
    print(f"API'den çekilen toplam: {len(incoming)}")
    print(f"Daha önce bulunmayan yeni issue: {len(new_rows)}")
    print(f"Yeni labelsız issue: {new_unlabeled}")
    print(f"Ek veri dosyasındaki toplam yeni kayıt: {len(merged_additional)}")
    print(f"Birleşik benzersiz issue: {len(combined)}")
    print(f"Birleşik labelsız issue: {len(unlabeled)}")
    print(f"Ana ham dosya (değiştirilmedi): {BASE_DATASET}")
    print(f"Yeni kayıtlar: {ADDITIONAL_DATASET}")
    print(f"Labelsız kayıtlar: {UNLABELED_DATASET}")


if __name__ == "__main__":
    main()


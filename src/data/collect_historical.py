from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

REPOSITORIES = [("fastapi", "fastapi"), ("pandas-dev", "pandas"), ("scikit-learn", "scikit-learn")]
TARGET_OLDER_PER_REPOSITORY = 500
PER_PAGE = 100
SEARCH_URL = "https://api.github.com/search/issues"
COLUMNS = ["repository", "issue_id", "title", "body", "labels", "issue_url", "created_at"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BASE_DATASET = RAW_DIR / "github_issues_all.csv"  # Read only.
ADDITIONAL_DATASET = RAW_DIR / "github_issues_additional.csv"
UNLABELED_DATASET = PROCESSED_DIR / "github_issues_unlabeled.csv"


def read_csv(path: Path) -> pd.DataFrame:
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


def main() -> None:
    base = read_csv(BASE_DATASET)
    additional = read_csv(ADDITIONAL_DATASET)
    existing = pd.concat([base, additional], ignore_index=True).drop_duplicates(subset=["issue_id"], keep="first")
    existing["created_at"] = pd.to_datetime(existing["created_at"], errors="coerce", utc=True)

    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    older_rows: list[dict[str, str]] = []
    with requests.Session() as session:
        session.headers.update(headers)
        for owner, repo in REPOSITORIES:
            repo_name = f"{owner}/{repo}"
            repo_existing = existing.loc[existing["repository"].eq(repo_name), "created_at"].dropna()
            if repo_existing.empty:
                raise ValueError(f"{repo_name} için başlangıç verisi bulunamadı.")
            cutoff = repo_existing.min().strftime("%Y-%m-%d")
            query = f"repo:{repo_name} is:issue created:<{cutoff}"
            page = 1
            repo_rows: list[dict[str, str]] = []
            print(f"{repo_name}: {cutoff} tarihinden daha eski kayıtlar alınıyor", flush=True)

            while len(repo_rows) < TARGET_OLDER_PER_REPOSITORY and page <= 10:
                result = request_search(session, query, page)
                if page == 1:
                    print(f"  Bu tarihten eski bulunan issue: {result.get('total_count', 0)}", flush=True)
                items = result.get("items", [])
                if not items:
                    break
                for issue in items:
                    repo_rows.append({
                        "repository": repo_name,
                        "issue_id": str(issue["id"]),
                        "title": issue.get("title") or "",
                        "body": issue.get("body") or "",
                        "labels": ";".join(label.get("name", "") for label in issue.get("labels", [])),
                        "issue_url": issue.get("html_url", ""),
                        "created_at": issue.get("created_at", ""),
                    })
                    if len(repo_rows) >= TARGET_OLDER_PER_REPOSITORY:
                        break
                page += 1
            print(f"  Çekilen eski issue: {len(repo_rows)}", flush=True)
            older_rows.extend(repo_rows)

    incoming = pd.DataFrame(older_rows, columns=COLUMNS)
    known_ids = set(base["issue_id"].astype(str)) | set(additional["issue_id"].astype(str))
    new_rows = incoming.loc[~incoming["issue_id"].astype(str).isin(known_ids)]
    merged_additional = pd.concat([additional, new_rows], ignore_index=True).drop_duplicates(subset=["issue_id"], keep="first")
    merged_additional.to_csv(ADDITIONAL_DATASET, index=False, encoding="utf-8-sig")

    combined = pd.concat([base, merged_additional], ignore_index=True).drop_duplicates(subset=["issue_id"], keep="first")
    unlabeled = combined.loc[combined["labels"].fillna("").str.strip().eq("")].copy()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    unlabeled.to_csv(UNLABELED_DATASET, index=False, encoding="utf-8-sig")

    print("\nGEÇMİŞ VERİ TOPLAMA ÖZETİ")
    print(f"Daha eski çekilen: {len(incoming)}")
    print(f"Yeni ve benzersiz eklenen: {len(new_rows)}")
    print(f"Yeni labelsız eklenen: {int(new_rows['labels'].fillna('').str.strip().eq('').sum())}")
    print(f"Birleşik issue sayısı: {len(combined)}")
    print(f"Birleşik labelsız issue sayısı: {len(unlabeled)}")
    print(f"Yeni ham kayıtlar: {ADDITIONAL_DATASET}")
    print(f"Labelsız kayıtlar: {UNLABELED_DATASET}")


if __name__ == "__main__":
    main()

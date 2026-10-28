from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT = RAW_DIR / "github_issues_category_expansion.csv"
COLUMNS = ["repository", "issue_id", "title", "body", "labels", "issue_url", "created_at"]
TARGET_PER_CATEGORY = 300
PER_PAGE = 100
MAX_PAGES_PER_QUERY = 5
SEARCH_URL = "https://api.github.com/search/issues"
CATEGORY_QUERIES = {
    "Documentation": [
        ("pallets/flask", "docs"),
        ("encode/httpx", "docs"),
        ("pydantic/pydantic", "documentation"),
    ],
    "Feature Request": [
        ("encode/httpx", "enhancement"),
        ("pydantic/pydantic", "feature request"),
        ("pytest-dev/pytest", "type: enhancement"),
    ],
    "Question": [
        ("pallets/flask", "question"),
        ("encode/httpx", "question"),
        ("pydantic/pydantic", "question"),
    ],
}
LABEL_MAPPING = {
    "bug": "Bug",
    "documentation": "Documentation",
    "docs": "Documentation",
    "type: docs": "Documentation",
    "new feature": "Feature Request",
    "enhancement": "Feature Request",
    "feature request": "Feature Request",
    "type: enhancement": "Feature Request",
    "question": "Question",
    "type: question": "Question",
}


def mapped_category(labels: list[str]) -> str | None:
    categories = {LABEL_MAPPING[label.strip().casefold()] for label in labels if label.strip().casefold() in LABEL_MAPPING}
    return next(iter(categories)) if len(categories) == 1 else None


def read_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, dtype={"issue_id": "string"}, keep_default_na=False)
    return set(frame["issue_id"].astype(str)) if "issue_id" in frame.columns else set()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    current = pd.read_csv(OUTPUT, dtype={"issue_id": "string"}, keep_default_na=False) if OUTPUT.exists() else pd.DataFrame(columns=COLUMNS)
    for column in COLUMNS:
        if column not in current.columns:
            current[column] = ""
    current = current[COLUMNS].drop_duplicates(subset=["issue_id"], keep="first")

    known_ids: set[str] = set(current["issue_id"].astype(str))
    source_files = list(RAW_DIR.glob("*.csv")) + list((PROJECT_ROOT / "data" / "test" / "raw").glob("*.csv"))
    for path in source_files:
        known_ids.update(read_ids(path))

    counts = {category: 0 for category in CATEGORY_QUERIES}
    for row in current.to_dict("records"):
        labels = [part.strip() for part in str(row["labels"]).split(";") if part.strip()]
        category = mapped_category(labels)
        if category in counts:
            counts[category] += 1

    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "github-issue-triage-data-collection"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    added: list[dict[str, str]] = []
    session = requests.Session()
    session.headers.update(headers)
    for category, queries in CATEGORY_QUERIES.items():
        if counts[category] >= TARGET_PER_CATEGORY:
            continue
        for repository, label in queries:
            if counts[category] >= TARGET_PER_CATEGORY:
                break
            query = f'repo:{repository} is:issue label:"{label}"'
            print(f"Aranıyor: {repository} — {label}", flush=True)
            for page in range(1, MAX_PAGES_PER_QUERY + 1):
                if counts[category] >= TARGET_PER_CATEGORY:
                    break
                response = session.get(SEARCH_URL, params={"q": query, "per_page": PER_PAGE, "page": page, "sort": "created", "order": "desc"}, timeout=30)
                if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
                    wait_seconds = max(1, int(response.headers.get("X-RateLimit-Reset", "0")) - int(time.time()) + 1)
                    print(f"GitHub arama sınırı nedeniyle {wait_seconds} sn bekleniyor.", flush=True)
                    time.sleep(wait_seconds)
                    response = session.get(SEARCH_URL, params={"q": query, "per_page": PER_PAGE, "page": page, "sort": "created", "order": "desc"}, timeout=30)
                response.raise_for_status()
                payload = response.json()
                items = payload.get("items", [])
                if page == 1:
                    print(f"  Uyan sonuç: {payload.get('total_count', 0)}", flush=True)
                if not items:
                    break
                for issue in items:
                    issue_id = str(issue.get("id", ""))
                    labels = [str(item.get("name", "")) for item in issue.get("labels", [])]
                    if not issue_id or issue_id in known_ids or mapped_category(labels) != category:
                        continue
                    added.append({
                        "repository": repository,
                        "issue_id": issue_id,
                        "title": issue.get("title") or "",
                        "body": issue.get("body") or "",
                        "labels": ";".join(labels),
                        "issue_url": issue.get("html_url") or "",
                        "created_at": issue.get("created_at") or "",
                    })
                    known_ids.add(issue_id)
                    counts[category] += 1
                    if counts[category] >= TARGET_PER_CATEGORY:
                        break
                if len(items) < PER_PAGE:
                    break

    if added:
        combined = pd.concat([current, pd.DataFrame(added, columns=COLUMNS)], ignore_index=True)
        combined = combined.drop_duplicates(subset=["issue_id"], keep="first")[COLUMNS]
        combined.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    elif not OUTPUT.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print("\nTOPLAMA ÖZETİ")
    print(f"Bu çalışmada eklenen benzersiz issue: {len(added)}")
    for category in CATEGORY_QUERIES:
        print(f"{category}: {counts[category]} / {TARGET_PER_CATEGORY}")
    print(f"Kaynak: {OUTPUT}")
    session.close()


if __name__ == "__main__":
    main()




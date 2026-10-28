import requests
import pandas as pd
from pathlib import Path

# 1. Repositories to collect data from
REPOSITORIES = [
    ("fastapi", "fastapi"),
    ("pandas-dev", "pandas"),
    ("scikit-learn", "scikit-learn")
]

# 2. Collection settings
TARGET_ISSUES = 300
MAX_PAGES = 30
PER_PAGE = 100

# 3. GitHub API settings
BASE_URL = "https://api.github.com/repos"

all_issues = []

# 4. Collect issues from each repository
for owner, repo in REPOSITORIES:

    print(f"\nCollecting data from {owner}/{repo}...")

    url = f"{BASE_URL}/{owner}/{repo}/issues"

    params = {
        "state": "all",
        "per_page": PER_PAGE,
        "sort": "created",
        "direction": "desc"
    }

    repo_issues = []
    page = 1

    # 5. Pagination
    while len(repo_issues) < TARGET_ISSUES and page <= MAX_PAGES:

        print(f"Fetching page {page}...")

        params["page"] = page

        try:
            response = requests.get(
                url,
                params=params,
                timeout=30
            )
        except requests.RequestException as error:
            print("Connection error:", error)
            break

        print("Status code:", response.status_code)

        if response.status_code != 200:
            print("API Error:", response.text)
            break

        issues = response.json()

        if not issues:
            print("No more records found.")
            break

        # 6. Process each issue
        for issue in issues:

            # Skip Pull Requests
            if "pull_request" in issue:
                continue

            labels = [
                label["name"]
                for label in issue.get("labels", [])
            ]

            repo_issues.append({
                "repository": f"{owner}/{repo}",
                "issue_id": issue["id"],
                "title": issue["title"],
                "body": issue["body"] or "",
                "labels": ";".join(labels),
                "issue_url": issue["html_url"],
                "created_at": issue["created_at"]
            })

            if len(repo_issues) >= TARGET_ISSUES:
                break

        page += 1

    print(f"Issues collected from {owner}/{repo}: {len(repo_issues)}")

    all_issues.extend(repo_issues)


# 7. Convert all collected issues into a DataFrame
df = pd.DataFrame(
    all_issues,
    columns=[
        "repository",
        "issue_id",
        "title",
        "body",
        "labels",
        "issue_url",
        "created_at"
    ]
)

# 8. Save the combined dataset
output_path = Path("data/raw/github_issues_all.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False, encoding="utf-8-sig")

# 9. Print collection summary
print("\n========== COLLECTION SUMMARY ==========")

print("Total issues collected:", len(df))

print("\nIssues per repository:")
print(df["repository"].value_counts())

print("\nLabeled issues:", (df["labels"].str.len() > 0).sum())
print("Unlabeled issues:", (df["labels"].str.len() == 0).sum())

print("\nSaved to:", output_path)
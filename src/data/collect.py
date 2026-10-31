"""Collect labeled issues from public GitHub repositories as JSON Lines."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import urllib.error
import urllib.parse
import urllib.request

DEFAULT_REPOSITORIES = [
    "fastapi/fastapi",
    "mlflow/mlflow",
    "scikit-learn/scikit-learn",
    "pandas-dev/pandas",
    "encode/httpx",
]
GITHUB_API = "https://api.github.com"
PAGE_SIZE = 100  # GitHub's maximum page size for this endpoint.


def make_session() -> dict[str, str]:
    """Create request headers, using a token if one is available in the environment."""
    session = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "github-issue-triage-course-project",
    }
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if token:
        session["Authorization"] = f"Bearer {token}"
    return session


def get_issues(
    session: dict[str, str],
    repository: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Fetch up to `limit` newest issues, excluding pull requests."""
    url = f"{GITHUB_API}/repos/{repository}/issues"
    params: dict[str, Any] = {
         "state": "all",
        "sort": "created",
        "direction": "desc",
        "per_page": PAGE_SIZE,
        "page": 1,
    }
    collected: list[dict[str, Any]] = []

    while len(collected) < limit:
        request_url = f"{url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(request_url, headers=session)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                page_items = json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code == 403 and error.headers.get("X-RateLimit-Remaining") == "0":
                reset_at = error.headers.get("X-RateLimit-Reset", "unknown")
                raise RuntimeError(
                    "GitHub API rate limit reached. Set GITHUB_TOKEN and try again. "
                    f"Rate limit reset timestamp: {reset_at}."
                ) from error
            raise
        if not isinstance(page_items, list):
            raise RuntimeError(f"Unexpected GitHub API response for {repository}.")

        for item in page_items:
            # The issues endpoint also returns pull requests; they are not issue examples.
            if "pull_request" in item:
                continue
            labels = [label["name"] for label in item.get("labels", []) if label.get("name")]
            collected.append(
                {
                    "repository": repository,
                    "issue_number": item["number"],
                    "title": item.get("title") or "",
                    "body": item.get("body") or "",
                    "state": item.get("state"),
                    "labels": labels,
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "closed_at": item.get("closed_at"),
                    "comments": item.get("comments", 0),
                    "html_url": item.get("html_url"),
                }
            )
            if len(collected) >= limit:
                break

        if len(page_items) < PAGE_SIZE:
            break
        params["page"] += 1

    return collected


def collect(repositories: list[str], limit_per_repo: int, output: Path) -> int:
    """Fetch issues from each repository and replace the output JSONL file."""
    if limit_per_repo < 1:
        raise ValueError("--limit-per-repo must be at least 1.")

    session = make_session()
    all_issues: list[dict[str, Any]] = []
    for repository in repositories:
        print(f"Fetching up to {limit_per_repo} issues from {repository}...")
        issues = get_issues(session, repository, limit_per_repo)
        all_issues.extend(issues)
        print(f"  Collected {len(issues)} issues.")
        # Be considerate of the unauthenticated GitHub API rate limit.
        time.sleep(0.25)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        for issue in all_issues:
            file.write(json.dumps(issue, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_issues)} issues to {output}")
    return len(all_issues)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect recent GitHub issues and their original labels as JSONL."
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        default=DEFAULT_REPOSITORIES,
        help="Repositories in owner/name format. Defaults to five example projects.",
    )
    parser.add_argument(
        "--limit-per-repo",
        type=int,
        default=100,
        help="Maximum number of issues to collect from each repository (default: 100).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/github_issues.jsonl"),
        help="Output JSONL path, relative to the project directory by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        collect(args.repos, args.limit_per_repo, args.output)
    except urllib.error.HTTPError as error:
        status = error.code
        print(
            f"GitHub API request failed (HTTP {status}): {error}. "
            "Check the repository names and your internet connection.",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    except (urllib.error.URLError, RuntimeError, ValueError) as error:
        print(f"Collection failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

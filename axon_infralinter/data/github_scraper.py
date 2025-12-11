from __future__ import annotations

"""
GitHub repository discovery and cloning for Terraform IaC files.

This script:
- Searches GitHub for repositories containing Terraform (`.tf`) files.
- Clones a configurable number of repositories locally.
- Produces a simple manifest JSONL with repo metadata and clone status.

Environment:
- GITHUB_TOKEN: GitHub personal access token (recommended to avoid rate limits).
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

from github import Github
from tqdm import tqdm

from axon_infralinter.config import (
    GITHUB_MAX_REPOS,
    GITHUB_SEARCH_QUERY,
    PROJECT_ROOT,
    REPOS_ROOT,
    ensure_directories,
)


@dataclass
class RepoRecord:
    full_name: str
    clone_url: str
    default_branch: str | None
    stars: int
    ssh_url: str
    html_url: str
    cloned_path: str | None = None
    clone_ok: bool | None = None
    error: str | None = None


def get_github_client() -> Github:
    from axon_infralinter.config import GITHUB_TOKEN
    if GITHUB_TOKEN:
        return Github(GITHUB_TOKEN)
    print("Warning: No GITHUB_TOKEN set. Rate limits will be very restrictive.")
    return Github()


def search_repositories(client: Github, max_repos: int) -> List[RepoRecord]:
    """Search GitHub for Terraform repositories."""
    print(f"Searching GitHub for: {GITHUB_SEARCH_QUERY}")
    results: List[RepoRecord] = []
    
    repos = client.search_repositories(
        query=GITHUB_SEARCH_QUERY,
        sort="stars",
        order="desc"
    )
    
    for repo in repos:
        if len(results) >= max_repos:
            break
        
        record = RepoRecord(
            full_name=repo.full_name,
            clone_url=repo.clone_url,
            default_branch=repo.default_branch,
            stars=repo.stargazers_count,
            ssh_url=repo.ssh_url,
            html_url=repo.html_url,
        )
        results.append(record)
    return results


def clone_repo(record: RepoRecord, dest_root: Path) -> RepoRecord:
    import subprocess
    import shlex

    repo_dir = dest_root / record.full_name.replace("/", "__")
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    if repo_dir.exists():
        record.cloned_path = str(repo_dir)
        record.clone_ok = True
        return record

    cmd = f"git clone --depth 1 {shlex.quote(record.clone_url)} {shlex.quote(str(repo_dir))}"
    try:
        subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        record.cloned_path = str(repo_dir)
        record.clone_ok = True
    except subprocess.CalledProcessError as exc:
        record.clone_ok = False
        record.error = exc.stderr.decode("utf-8", errors="ignore")
    return record


def write_manifest(records: Iterable[RepoRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            json_line = json.dumps(asdict(rec), ensure_ascii=False)
            f.write(json_line + "\n")


def main() -> None:
    ensure_directories()
    client = get_github_client()

    print(f"Searching for up to {GITHUB_MAX_REPOS} repositories...")
    repos = search_repositories(client, GITHUB_MAX_REPOS)

    manifest_path = PROJECT_ROOT / "data" / "github_repos_manifest.jsonl"
    print(f"Cloning {len(repos)} repositories into {REPOS_ROOT} ...")

    cloned_records: List[RepoRecord] = []
    for rec in tqdm(repos, desc="Cloning repos"):
        cloned_records.append(clone_repo(rec, REPOS_ROOT))

    write_manifest(cloned_records, manifest_path)
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()


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
    """
    Create a PyGithub client instance, using authentication token if available.
    
    Returns:
        Authenticated Github client if GITHUB_TOKEN is set, otherwise unauthenticated client.
    
    Student understanding:
        GitHub's unauthenticated API has strict rate limits (60 requests/hour), which would
        severely limit our ability to scrape hundreds of repositories. Using a personal access
        token increases this to 5000 requests/hour, making large-scale scraping feasible.
        This was a critical learning - always check API rate limits before building scrapers!
    """
    from axon_infralinter.config import GITHUB_TOKEN
    if GITHUB_TOKEN:
        return Github(GITHUB_TOKEN)
    print("Warning: No GITHUB_TOKEN set. Rate limits will be very restrictive.")
    return Github()


def search_repositories(client: Github, max_repos: int) -> List[RepoRecord]:
    """
    Search GitHub for repositories matching the configured query.
    
    Args:
        client: Authenticated PyGithub client
        max_repos: Maximum number of repositories to return
    
    Returns:
        List of RepoRecord objects with repository metadata (no clone status yet)
    
    Student understanding:
        GitHub code search is imperfect - it may return forks, repositories with only
        tiny Terraform snippets, or repositories that no longer exist. We handle these
        gracefully in the cloning phase by recording clone success/failure in the manifest.
        Sorting by stars helps prioritize popular, well-maintained repositories that are
        more likely to contain substantial Terraform code.
    """
    print(f"Searching GitHub for: {GITHUB_SEARCH_QUERY}")
    results: List[RepoRecord] = []
    
    # Sort by stars descending to prioritize popular, well-maintained repos
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
    """
    Clone a GitHub repository using shallow clone.
    
    Args:
        record: RepoRecord with repository metadata
        dest_root: Root directory where repositories should be cloned
    
    Returns:
        Updated RepoRecord with clone status (cloned_path, clone_ok, error)
    
    Student understanding:
        Shallow clones (--depth 1) only fetch the latest commit, saving significant disk
        space and time. For our use case (extracting .tf files), we don't need git history.
        This optimization was crucial when cloning hundreds of repositories - full clones
        would have taken hours and consumed gigabytes of disk space unnecessarily.
        
        We use shlex.quote() to safely escape URLs and paths, preventing command injection
        vulnerabilities. This is a security best practice when constructing shell commands
        from user-controlled input (even though in this case the input is from GitHub API).
    """
    import subprocess
    import shlex

    # Create safe directory name by replacing / with __
    repo_dir = dest_root / record.full_name.replace("/", "__")
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already cloned (allows re-running script safely)
    if repo_dir.exists():
        record.cloned_path = str(repo_dir)
        record.clone_ok = True
        return record

    # Shallow clone: --depth 1 fetches only the latest commit
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
        # Record error for debugging - some repos may be private, deleted, or have other issues
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


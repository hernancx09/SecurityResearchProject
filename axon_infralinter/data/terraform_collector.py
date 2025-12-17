from __future__ import annotations

"""
Extract Terraform files from cloned repositories.

Reads the repository manifest and extracts all .tf files, copying them
to a flat directory structure with unique names.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional

from dataclasses import dataclass
from tqdm import tqdm

from axon_infralinter.config import (
    TERRAFORM_FILES_ROOT,
    REPOS_ROOT,
)


@dataclass
class TerraformFileRecord:
    source_repo: str
    source_path: str
    target_path: str
    file_size: int


def iter_terraform_files(repo_dir: Path) -> Iterable[Path]:
    """Yield all .tf files in a repository directory."""
    if not repo_dir.exists():
        return
    
    for p in repo_dir.rglob("*.tf"):
        if p.is_file():
            yield p


def collect_files(
    manifest: List[dict],
    max_total_files: Optional[int] = None,
    max_per_repo: Optional[int] = None,
) -> List[TerraformFileRecord]:
    """
    Collect Terraform files from repos with optional limits.
    
    Args:
        manifest: Repository manifest
        max_total_files: Maximum total files to collect (None = no limit)
        max_per_repo: Maximum files per repository (None = no limit)
    """
    TERRAFORM_FILES_ROOT.mkdir(parents=True, exist_ok=True)
    records: List[TerraformFileRecord] = []
    repo_file_counts: dict[str, int] = defaultdict(int)

    # Filter to only repos that were successfully cloned
    valid_repos = [
        rec for rec in manifest
        if rec.get("clone_ok") and rec.get("cloned_path")
    ]

    with tqdm(total=len(valid_repos), desc="Processing repos", unit="repo") as pbar:
        for rec in valid_repos:
            # Stop if we've reached the total limit
            if max_total_files is not None and len(records) >= max_total_files:
                pbar.set_postfix({
                    "status": "LIMIT REACHED",
                    "total": len(records)
                })
                break

            cloned_path = rec.get("cloned_path")
            if not cloned_path:
                pbar.update(1)
                continue

            repo_dir = Path(cloned_path)
            if not repo_dir.exists():
                pbar.update(1)
                continue

            repo_name = rec["full_name"]
            files_collected = 0
            files_skipped = 0

            for tf_path in iter_terraform_files(repo_dir):
                # Stop if we've reached the total limit
                if max_total_files is not None and len(records) >= max_total_files:
                    break
                
                # Stop if we've reached per-repo limit
                if max_per_repo is not None and repo_file_counts[repo_name] >= max_per_repo:
                    files_skipped += 1
                    continue

                # Create unique filename: repo__path__to__file.tf
                relative_path = tf_path.relative_to(repo_dir)
                safe_name = str(relative_path).replace("/", "__").replace("\\", "__")
                target_name = f"{repo_name.replace('/', '__')}__{safe_name}"
                target_path = TERRAFORM_FILES_ROOT / target_name

                # Copy file
                try:
                    import shutil
                    shutil.copy2(tf_path, target_path)
                    
                    record = TerraformFileRecord(
                        source_repo=repo_name,
                        source_path=str(tf_path),
                        target_path=str(target_path),
                        file_size=target_path.stat().st_size,
                    )
                    records.append(record)
                    repo_file_counts[repo_name] += 1
                    files_collected += 1
                except Exception as e:
                    files_skipped += 1
                    continue

            pbar.set_postfix({
                "collected": files_collected,
                "skipped": files_skipped,
                "total": len(records)
            })
            pbar.update(1)

    return records


def main() -> None:
    manifest_path = REPOS_ROOT.parent / "github_repos_manifest.jsonl"
    
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        print("Run github_scraper.py first to clone repositories.")
        return

    print(f"Loading manifest from {manifest_path}")
    manifest: List[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            manifest.append(json.loads(line))

    print(f"Found {len(manifest)} repository records")
    
    print(f"Collecting Terraform files into {TERRAFORM_FILES_ROOT}...")
    records = collect_files(manifest)
    
    print(f"Collected {len(records)} Terraform files")
    
    # Write manifest
    output_manifest = TERRAFORM_FILES_ROOT.parent / "terraform_files_manifest.jsonl"
    with output_manifest.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({
                "source_repo": rec.source_repo,
                "source_path": rec.source_path,
                "target_path": rec.target_path,
                "file_size": rec.file_size,
            }, ensure_ascii=False) + "\n")
    
    print(f"Wrote file manifest to {output_manifest}")


if __name__ == "__main__":
    main()


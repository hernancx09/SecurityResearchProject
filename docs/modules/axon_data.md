# Axon InfraLinter – Data Modules

This document describes the data collection and preprocessing modules that build the Terraform corpus from GitHub repositories.

## `axon_infralinter/data/github_scraper.py`

**Role**: Discover and clone GitHub repositories that contain Terraform code.

### Overview

This module searches GitHub for repositories matching a configurable query (default: "terraform in:name,description,readme"), clones them locally, and produces a manifest file recording clone status.

### Key Functions

#### `get_github_client() -> Github`

Creates a PyGithub client instance. Uses `GITHUB_TOKEN` from environment if available to avoid rate limits.

**Student understanding**: GitHub's unauthenticated API has strict rate limits (60 requests/hour). Using a personal access token increases this to 5000 requests/hour, which is essential for scraping hundreds of repositories.

#### `search_repositories(client: Github, max_repos: int) -> List[RepoRecord]`

Searches GitHub using the query from `config.GITHUB_SEARCH_QUERY`, sorted by stars (descending).

**Returns**: List of `RepoRecord` objects with metadata (full_name, clone_url, stars, etc.) but no clone status yet.

**Student understanding**: GitHub code search is imperfect - it may return forks, repositories with only tiny Terraform snippets, or repositories that no longer exist. We handle these gracefully in the cloning phase.

#### `clone_repo(record: RepoRecord, dest_root: Path) -> RepoRecord`

Clones a repository using `git clone --depth 1` (shallow clone for efficiency).

**Updates**: The `RepoRecord` with `cloned_path`, `clone_ok`, and `error` fields.

**Student understanding**: Shallow clones (`--depth 1`) only fetch the latest commit, saving significant disk space and time. For our use case (extracting `.tf` files), we don't need git history.

#### `main() -> None`

Orchestrates the full pipeline:
1. Search GitHub for repositories
2. Clone each repository
3. Write manifest to `data/github_repos_manifest.jsonl`

### Output

**`data/github_repos_manifest.jsonl`**: One JSON object per repository with fields:
- `full_name`: "owner/repo"
- `clone_url`: HTTPS URL
- `cloned_path`: Local path (or null if failed)
- `clone_ok`: Boolean success flag
- `error`: Error message if clone failed

---

## `axon_infralinter/data/terraform_collector.py`

**Role**: Extract `.tf` files from cloned repositories into a flat corpus directory.

### Overview

This module reads the repository manifest, walks each successfully cloned repository, finds all `.tf` files, and copies them into a single flat directory with unique names that preserve traceability.

### Key Functions

#### `iter_terraform_files(repo_dir: Path) -> Iterable[Path]`

Generator that yields all `.tf` files in a repository directory using `rglob("*.tf")`.

**Student understanding**: Using `rglob` recursively searches all subdirectories, which is important because Terraform projects often organize files into modules, environments, or feature directories.

#### `collect_files(manifest: List[dict], max_total_files: Optional[int] = None, max_per_repo: Optional[int] = None) -> List[TerraformFileRecord]`

Main collection function that processes repositories and extracts Terraform files.

**Parameters**:
- `manifest`: List of repository records from `github_repos_manifest.jsonl`
- `max_total_files`: Optional cap on total files across all repos
- `max_per_repo`: Optional cap on files per repository

**Returns**: List of `TerraformFileRecord` objects describing each copied file.

**Key logic**:
1. Filters to repos with `clone_ok == True` and valid `cloned_path`
2. For each repo, iterates through `.tf` files
3. Creates unique filename: `{repo_name}__{relative_path}` (with `/` replaced by `__`)
4. Copies file to `TERRAFORM_FILES_ROOT`
5. Records metadata

**Student understanding**: 
- **Flattening the corpus** decouples downstream scanning from git repository structure. This simplifies scanner invocation (no need to handle nested directories or git submodules) and makes it easier to share the dataset as a static corpus.
- **Encoding repo and path in filename** preserves traceability. When we see a misconfiguration, we can trace it back to the original repository and understand which projects contribute which types of issues. This is crucial for research reproducibility and for understanding dataset composition.

#### `main() -> None`

Entry point that:
1. Loads `github_repos_manifest.jsonl`
2. Calls `collect_files()` to extract Terraform files
3. Writes `terraform_files_manifest.jsonl` with metadata

### Output

**`data/terraform_files/`**: Flat directory containing all extracted `.tf` files with unique names like:
```
aws-ia__terraform-aws-eks-blueprints__patterns__aws-neuron-efa__main.tf
```

**`data/terraform_files_manifest.jsonl`**: One JSON object per file with fields:
- `source_repo`: Original repository name
- `source_path`: Full path in cloned repo
- `target_path`: Path in corpus directory
- `file_size`: Size in bytes

### Design Decisions

**Why flatten?**
- Simplifies scanner invocation (scanners can process a single directory)
- Removes dependency on git structure (can delete `data/repos/` after extraction)
- Makes dataset portable (can zip `terraform_files/` and share)

**Why preserve repo/path in filename?**
- Enables traceability for debugging and analysis
- Allows filtering by repository if needed
- Maintains provenance for research reproducibility

**Why optional limits?**
- `max_total_files`: Control corpus size for experiments or resource constraints
- `max_per_repo`: Prevent a single large repository from dominating the dataset


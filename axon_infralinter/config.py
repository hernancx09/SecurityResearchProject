"""
Central configuration for Axon InfraLinter experiments.

Adjust paths and model identifiers here rather than scattering constants.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

# Raw GitHub repositories and extracted Terraform files
REPOS_ROOT = DATA_ROOT / "repos"
TERRAFORM_FILES_ROOT = DATA_ROOT / "terraform_files"

# Scanner outputs and labels
SCANS_ROOT = DATA_ROOT / "scans"
DATASET_ROOT = DATA_ROOT / "datasets"

# GitHub configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_SEARCH_QUERY = "terraform in:name,description,readme"
GITHUB_MAX_REPOS = int(os.getenv("AXON_MAX_REPOS", "500"))

# Model configuration
BASE_MODEL_NAME = os.getenv(
    "AXON_BASE_MODEL",
    "codellama/CodeLlama-7b-Instruct-hf",
)
_lora_dir_env = os.getenv("AXON_LORA_DIR")
if _lora_dir_env:
    LORA_OUTPUT_DIR = Path(_lora_dir_env)
else:
    LORA_OUTPUT_DIR = PROJECT_ROOT / "models" / "axon_lora_codellama"

# Dataset sizes (targets, not hard guarantees)
TARGET_NUM_FILES = 1200
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def ensure_directories() -> None:
    """
    Create all required directory roots if they do not already exist.
    """
    for path in [
        DATA_ROOT,
        REPOS_ROOT,
        TERRAFORM_FILES_ROOT,
        SCANS_ROOT,
        DATASET_ROOT,
        LORA_OUTPUT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "DATA_ROOT",
    "REPOS_ROOT",
    "TERRAFORM_FILES_ROOT",
    "SCANS_ROOT",
    "DATASET_ROOT",
    "GITHUB_TOKEN",
    "GITHUB_SEARCH_QUERY",
    "GITHUB_MAX_REPOS",
    "BASE_MODEL_NAME",
    "LORA_OUTPUT_DIR",
    "TARGET_NUM_FILES",
    "TRAIN_RATIO",
    "VAL_RATIO",
    "TEST_RATIO",
    "ensure_directories",
]

from __future__ import annotations

"""
Build balanced train/validation/test splits from labeled Terraform files.

Input:
- `file_labels.jsonl` produced by `axon_infralinter.scanning.scanner`.

Output:
- `datasets/train.jsonl`
- `datasets/val.jsonl`
- `datasets/test.jsonl`

Each line:
{
  "input_text": "...prompt + file content...",
  "target_text": "SECURE\\nshort explanation" or "INSECURE\\nshort explanation",
  "secure": bool,
  "corpus_path": "...",
  "tool_findings": {...}
}
"""

import json
import random
from pathlib import Path
from typing import Dict, List

from axon_infralinter.config import (
    DATASET_ROOT,
    PROJECT_ROOT,
    TARGET_NUM_FILES,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    ensure_directories,
)


def load_labels(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_prompt(file_text: str) -> str:
    """
    Construct the instruction prompt that will be given to the LLM.
    
    Args:
        file_text: The Terraform file content
    
    Returns:
        Complete prompt string with instructions and file content
    
    Student understanding:
        Clear instructions help the model understand the task. The prompt explicitly lists
        what to look for (IAM policies, encryption, public exposure, secrets, network configs),
        which guides the model's attention. The structured output format (exactly two lines)
        makes parsing easier and more reliable than free-form text. Role-playing ("You are a
        security auditor") helps the model adopt the right perspective, similar to how
        instruction-tuned models are trained. Using the same prompt format in training and
        inference is crucial - the model learned to respond to this specific format.
    """
    instruction = (
        "You are a security auditor for Terraform Infrastructure-as-Code.\n"
        "Given the following Terraform file, decide if it is SECURE or INSECURE "
        "from a cloud security perspective. Consider issues like overly permissive "
        "IAM policies, missing encryption, public exposure of resources, hard-coded "
        "secrets, and weak network configurations.\n\n"
        "Return exactly two lines:\n"
        "First line: either SECURE or INSECURE.\n"
        "Second line: a brief explanation (one sentence).\n\n"
        "Terraform file:\n"
        "----------------\n"
    )
    return instruction + file_text


def build_target(secure: bool) -> str:
    if secure:
        return "SECURE\nNo high or critical security misconfigurations were detected in this file."
    return "INSECURE\nAt least one high or critical security misconfiguration was detected in this file."


def main() -> None:
    """
    Build balanced train/validation/test splits from labeled Terraform files.
    
    Pipeline:
        1. Load labels from file_labels.jsonl
        2. Filter to files that still exist on disk
        3. Split into secure vs. insecure subsets
        4. Sample up to TARGET_NUM_FILES with equal class sizes
        5. Shuffle and split into train/val/test by configured ratios
    
    Student understanding:
        Class balancing is critical. Real-world security datasets are naturally imbalanced
        (insecure examples are relatively rare). Without balancing, simple baselines like
        "always predict secure" achieve deceptively high accuracy, masking true performance
        differences. By enforcing equal numbers of secure and insecure examples, we ensure
        that models must actually learn to distinguish between classes, not just predict the
        majority. Fixed random seed (42) is essential for reproducibility - same dataset
        splits across runs enables fair comparison between model variants.
    """
    ensure_directories()
    labels_path = PROJECT_ROOT / "data" / "file_labels.jsonl"
    print(f"Loading labels from {labels_path}")
    labels = load_labels(labels_path)
    print(f"Loaded {len(labels)} labeled files")

    # Filter to those that still exist on disk (files may have been deleted during cleanup)
    labels = [r for r in labels if Path(r["corpus_path"]).exists()]
    print(f"After filtering for existing files: {len(labels)} files")

    # Split by class - this is the first step toward balancing
    secure_records = [r for r in labels if r.get("secure", False)]
    insecure_records = [r for r in labels if not r.get("secure", False)]

    print(f"Secure files: {len(secure_records)}, Insecure files: {len(insecure_records)}")

    # Balance classes: sample equal numbers from each class
    # This prevents class imbalance from dominating metrics
    target_per_class = TARGET_NUM_FILES // 2
    random.seed(42)  # Fixed seed for reproducibility
    random.shuffle(secure_records)
    random.shuffle(insecure_records)

    secure_sample = secure_records[:target_per_class]
    insecure_sample = insecure_records[:target_per_class]
    all_records = secure_sample + insecure_sample
    random.shuffle(all_records)  # Shuffle combined set before splitting

    n_total = len(all_records)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    # Remaining go to test
    n_test = n_total - n_train - n_val

    train_recs = all_records[:n_train]
    val_recs = all_records[n_train : n_train + n_val]
    test_recs = all_records[n_train + n_val :]

    print(f"Dataset sizes -> train: {len(train_recs)}, val: {len(val_recs)}, test: {len(test_recs)}")

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    def write_split(recs: List[Dict], out_path: Path) -> None:
        with out_path.open("w", encoding="utf-8") as f:
            for r in recs:
                file_text = Path(r["corpus_path"]).read_text(encoding="utf-8", errors="ignore")
                example = {
                    "input_text": build_prompt(file_text),
                    "target_text": build_target(r.get("secure", False)),
                    "secure": r.get("secure", False),
                    "corpus_path": r["corpus_path"],
                    "tool_findings": r.get("tool_findings", {}),
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

    write_split(train_recs, DATASET_ROOT / "train.jsonl")
    write_split(val_recs, DATASET_ROOT / "val.jsonl")
    write_split(test_recs, DATASET_ROOT / "test.jsonl")

    print(f"Wrote dataset splits under {DATASET_ROOT}")


if __name__ == "__main__":
    main()


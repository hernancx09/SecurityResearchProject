# Axon InfraLinter – Dataset Module

This document describes the dataset building module that creates balanced train/validation/test splits from labeled Terraform files.

## `axon_infralinter/dataset/build_dataset.py`

**Role**: Build balanced train/val/test splits from labeled Terraform files for model training and evaluation.

### Overview

This module reads the security labels produced by the scanner, filters to files that still exist on disk, balances secure and insecure examples, and creates train/validation/test splits with configurable ratios. Each example includes a prompt, the Terraform file content, and the expected model output.

### Key Functions

#### `load_labels(path: Path) -> List[Dict]`

Loads the `file_labels.jsonl` file into a list of dictionaries.

**Simple JSONL parsing**: Each line is a JSON object, so we parse line-by-line.

#### `build_prompt(file_text: str) -> str`

Constructs the instruction prompt that will be given to the LLM.

**Prompt structure**:
1. Role definition: "You are a security auditor for Terraform Infrastructure-as-Code"
2. Task description: Decide if file is SECURE or INSECURE
3. Considerations: Lists types of issues to look for (IAM policies, encryption, public exposure, secrets, network configs)
4. Output format: Exactly two lines - first line is SECURE/INSECURE, second line is brief explanation
5. File content: The actual Terraform code

**Student understanding**: 
- **Clear instructions** help the model understand the task. The prompt explicitly lists what to look for, which guides the model's attention.
- **Structured output format** (two lines) makes parsing easier and more reliable than free-form text.
- **Role-playing** ("You are a security auditor") helps the model adopt the right perspective, similar to how instruction-tuned models are trained.

#### `build_target(secure: bool) -> str`

Generates the expected model output (target text) for a given label.

**Format**:
- Secure: `"SECURE\nNo high or critical security misconfigurations were detected in this file."`
- Insecure: `"INSECURE\nAt least one high or critical security misconfiguration was detected in this file."`

**Student understanding**: The target text matches the prompt's output format specification. This consistency is important for fine-tuning - the model learns to produce outputs in this exact format.

#### `main() -> None`

Main pipeline that orchestrates dataset building:

1. **Load labels**: Reads `data/file_labels.jsonl`
2. **Filter existing files**: Removes records where `corpus_path` no longer exists (files may have been deleted)
3. **Split by class**: Separates into `secure_records` and `insecure_records`
4. **Balance**: Samples up to `TARGET_NUM_FILES // 2` from each class
5. **Shuffle**: Combines and shuffles all records
6. **Split**: Divides into train/val/test using configured ratios
7. **Write**: Creates JSONL files for each split

**Key parameters** (from `config.py`):
- `TARGET_NUM_FILES`: Total target size (default: 1200)
- `TRAIN_RATIO`: Training set proportion (default: 0.70)
- `VAL_RATIO`: Validation set proportion (default: 0.15)
- `TEST_RATIO`: Test set proportion (default: 0.15)

**Random seed**: Fixed to `42` for reproducibility.

**Student understanding**:
- **Class balancing** is critical. Real-world security datasets are naturally imbalanced (insecure examples are rare). Without balancing:
  - Simple baselines like "always predict secure" achieve deceptively high accuracy
  - Model performance differences are masked by class distribution effects
  - Evaluation metrics become less meaningful
  
  By enforcing equal numbers of secure and insecure examples, we ensure that:
  - Models must actually learn to distinguish between classes
  - Performance differences reflect true learning, not just label distribution
  - Metrics like precision, recall, and F1 are interpretable

- **Fixed random seed** (`random.seed(42)`) is essential for:
  - Reproducibility: Same dataset splits across runs
  - Fair comparison: Different models evaluated on identical test sets
  - Debugging: Can reproduce issues with specific examples

- **Filtering existing files**: Files may be deleted between scanning and dataset building (e.g., during cleanup). We check existence to avoid errors during training.

### Output Format

Each example in `train.jsonl`, `val.jsonl`, or `test.jsonl` is a JSON object with:

```json
{
  "input_text": "You are a security auditor...\n\nTerraform file:\n----------------\nresource \"aws_s3_bucket\" \"example\" {...}",
  "target_text": "INSECURE\nAt least one high or critical security misconfiguration was detected in this file.",
  "secure": false,
  "corpus_path": "/path/to/file.tf",
  "tool_findings": {
    "checkov": [...],
    "tfsec": [...]
  }
}
```

**Fields**:
- `input_text`: Full prompt + Terraform file content
- `target_text`: Expected model output (SECURE/INSECURE + explanation)
- `secure`: Boolean label (for easy filtering/analysis)
- `corpus_path`: Path to original file (for traceability)
- `tool_findings`: Original scanner findings (for reference, not used in training)

### Design Decisions

**Why balance classes?**
- Prevents class imbalance from dominating metrics
- Makes evaluation more meaningful
- Ensures model learns to distinguish classes, not just predict majority

**Why preserve tool_findings?**
- Useful for analysis and debugging
- Allows understanding why a file was labeled secure/insecure
- Enables future work on explanation quality

**Why filter existing files?**
- Handles cases where files are deleted between stages
- Prevents runtime errors during training
- Makes pipeline more robust

**Why fixed seed?**
- Reproducibility is essential for research
- Enables fair comparison across model variants
- Makes debugging easier

### Usage in Training

The dataset files are consumed by `axon_infralinter/models/train_lora.py`:

```python
dataset = load_dataset(
    "json",
    data_files={
        "train": str(DATASET_ROOT / "train.jsonl"),
        "validation": str(DATASET_ROOT / "val.jsonl"),
    },
)
```

The training script concatenates `input_text` and `target_text` and uses causal language modeling (next-token prediction) to fine-tune the model.


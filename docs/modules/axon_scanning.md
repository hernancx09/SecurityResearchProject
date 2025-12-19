# Axon InfraLinter – Scanning Module

This document describes the security scanning module that runs rule-based scanners (Checkov and tfsec) on Terraform files and produces unified labels.

## `axon_infralinter/scanning/scanner.py`

**Role**: Run Checkov and tfsec on each Terraform file and aggregate their findings into a unified label format.

### Overview

This module processes all `.tf` files in the corpus, runs both Checkov and tfsec on each file, parses their JSON outputs, normalizes findings into a common structure, and produces a label file indicating whether each file is secure or insecure.

### Key Functions

#### `find_checkov_command() -> List[str]`

Determines how to invoke Checkov, trying multiple methods:
1. Direct `checkov` command (if in PATH)
2. Python module: `python -m checkov.cli`

**Student understanding**: Checkov can be installed as a standalone binary or as a Python package. This function handles both cases to make the pipeline more robust across different installation methods.

#### `find_tfsec_command() -> List[str]`

Determines how to invoke tfsec, checking:
1. `tfsec` in PATH
2. Common installation locations (`~/.local/bin/tfsec`, `/usr/local/bin/tfsec`)

**Student understanding**: tfsec is typically installed as a standalone binary. We check common locations because it may not always be in PATH (especially on Linux systems where users install to `~/.local/bin`).

#### `run_checkov_on_file(tf_path: Path) -> Dict`

Runs Checkov on a single Terraform file and returns parsed JSON output.

**Command**: `checkov -f {file} --output json --quiet`

**Error handling**: Returns `{}` on timeout, file not found, or invalid JSON.

**Exit codes**: Checkov returns non-zero when findings are detected, but we accept exit codes 0 and 1 as success (only codes ≥2 indicate actual errors).

**Output**: Raw JSON is written to `SCANS_ROOT/checkov/{filename}.json` for debugging.

#### `run_tfsec_on_file(tf_path: Path) -> Dict`

Runs tfsec on a Terraform file's parent directory and filters results to the target file.

**Important note**: tfsec requires a directory, not a file. We run it on `tf_path.parent` and filter results.

**Command**: `tfsec {directory} --format json --no-color`

**Exit code handling**: 
- Exit code `0`: Clean (no issues)
- Exit code `1`: Issues found (this is **success**, not failure!)
- Exit code `2+`: Actual errors

**Student understanding**: This was a critical bug I had to fix. tfsec's exit code semantics are non-standard - exit code 1 means "scan succeeded and found issues", not "scan failed". Many tools use non-zero exit codes to indicate errors, but here 1 is actually a successful run. Misinterpreting this would cause us to silently drop valid security findings, severely biasing our dataset toward secure examples.

**Filtering**: tfsec scans the entire directory, so we filter results to only include findings for the specific file we're interested in.

#### `parse_checkov_findings(data: Dict, tf_path: Path) -> List[Finding]`

Parses Checkov's JSON output structure into a list of `Finding` objects.

**Checkov JSON structure**:
```json
{
  "results": {
    "failed_checks": [
      {
        "check_id": "CKV_AWS_26",
        "severity": "HIGH",
        "check_name": "Ensure all data stored in the SNS topic is encrypted",
        "file_path": "...",
        "file_line_range": [199, 199]
      }
    ]
  }
}
```

**Normalization**: Converts Checkov's structure into our common `Finding` dataclass with fields:
- `tool`: "checkov"
- `rule_id`: Checkov rule ID (e.g., "CKV_AWS_26")
- `severity`: Normalized to uppercase ("HIGH", "CRITICAL", etc.)
- `message`: Human-readable description
- `file_path`: Path to file
- `start_line`: Line number (if available)

#### `parse_tfsec_findings(data: Dict, tf_path: Path) -> List[Finding]`

Parses tfsec's JSON output structure into a list of `Finding` objects.

**tfsec JSON structure**:
```json
{
  "results": [
    {
      "rule_id": "aws-sns-enable-topic-encryption",
      "severity": "HIGH",
      "description": "SNS topic should be encrypted",
      "location": {
        "filename": "...",
        "start_line": 199
      }
    }
  ]
}
```

**Filtering**: Only includes findings where `location.filename` matches our target file (since tfsec scans the entire directory).

**Normalization**: Converts to the same `Finding` structure as Checkov for consistency.

#### `is_secure(checkov_findings: List[Finding], tfsec_findings: List[Finding]) -> bool`

Determines whether a file should be labeled as secure.

**Rule**: Returns `False` (insecure) if **any** finding has severity `HIGH` or `CRITICAL` from **either** tool.

**Student understanding**: 
- We use a security-centric definition: a file is insecure if it has **any** high-severity issue, not just if it has more issues than some threshold.
- We combine findings from both tools using OR logic - if either tool finds a high-severity issue, the file is marked insecure. This is conservative but appropriate for security applications.

#### `main() -> None`

Orchestrates the full scanning pipeline:
1. Finds all `.tf` files in `TERRAFORM_FILES_ROOT`
2. For each file:
   - Runs Checkov and tfsec
   - Parses findings
   - Determines secure/insecure label
   - Writes label to `file_labels.jsonl`

### Output

**`data/file_labels.jsonl`**: One JSON object per Terraform file with:
- `corpus_path`: Path to file in corpus
- `secure`: Boolean label
- `tool_findings`: Dictionary with `checkov` and `tfsec` keys, each containing lists of Finding objects

**Raw scan outputs**: Written to `SCANS_ROOT/checkov/` and `SCANS_ROOT/tfsec/` for debugging and analysis.

### Design Decisions

**Why normalize findings?**
- Different scanners use different output formats and field names
- Normalization makes it easier to:
  - Compare coverage between tools
  - Add new scanners later
  - Reason about the dataset uniformly

**Why store raw outputs?**
- Enables debugging when labels seem incorrect
- Allows re-parsing if we change the normalization logic
- Provides full context for analysis

**Why combine tools with OR logic?**
- Security is about finding **any** vulnerability, not averaging across tools
- Different tools have different strengths (Checkov has broader coverage, tfsec is AWS-focused)
- Combining with OR maximizes recall (fewer false negatives)

### Common Pitfalls

1. **tfsec exit codes**: Remember that exit code 1 is success with findings, not failure
2. **Directory vs. file**: tfsec requires a directory, so we must filter results
3. **Timeout handling**: Some files may be very large or cause scanners to hang - we use 30-second timeouts
4. **JSON parsing**: Both scanners may produce malformed JSON in edge cases - we catch `JSONDecodeError` and return empty results


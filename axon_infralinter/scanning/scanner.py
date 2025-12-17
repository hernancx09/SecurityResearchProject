from __future__ import annotations

"""
Run Checkov and tfsec over collected Terraform files and generate labels.

Outputs:
- Per-file scan results (raw JSON) under `SCANS_ROOT`.
- A unified JSONL file `file_labels.jsonl` with one record per Terraform file:
  {
    "corpus_path": "...",
    "secure": bool,
    "tool_findings": {
        "checkov": [...],
        "tfsec": [...]
    }
  }

Label rule:
- A file is INSECURE if any HIGH or CRITICAL finding exists from either tool.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from axon_infralinter.config import SCANS_ROOT, TERRAFORM_FILES_ROOT, ensure_directories


@dataclass
class Finding:
    tool: str
    rule_id: str
    severity: str
    message: str
    file_path: str
    start_line: int | None = None


def find_checkov_command() -> List[str]:
    """Find the best way to run checkov - try multiple methods."""
    # Try 1: Direct command (if in PATH)
    checkov_path = shutil.which("checkov")
    if checkov_path:
        return [checkov_path]
    
    # Try 2: Python module
    return [sys.executable, "-m", "checkov.cli"]


def find_tfsec_command() -> List[str]:
    """Find tfsec command."""
    tfsec_path = shutil.which("tfsec")
    if tfsec_path:
        return [tfsec_path]
    
    # Try common installation locations
    common_paths = [
        Path.home() / ".local" / "bin" / "tfsec",
        Path("/usr/local/bin/tfsec"),
    ]
    for path in common_paths:
        if path.exists():
            return [str(path)]
    
    return ["tfsec"]  # Fallback, will fail if not found


def run_checkov_on_file(tf_path: Path) -> Dict:
    """Run Checkov on a single Terraform file, returning parsed JSON (or {} on failure)."""
    output_path = SCANS_ROOT / "checkov" / (tf_path.name + ".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = find_checkov_command() + [
        "-f", str(tf_path),
        "--output", "json",
        "--quiet",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Checkov may return non-zero on findings, so we check for actual errors
        if result.returncode not in [0, 1]:
            return {}
        
        # Parse JSON output
        if result.stdout:
            try:
                data = json.loads(result.stdout)
                # Write raw output
                output_path.write_text(result.stdout, encoding="utf-8")
                return data
            except json.JSONDecodeError:
                return {}
        
        return {}
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return {}


def parse_checkov_findings(data: Dict, tf_path: Path) -> List[Finding]:
    """Parse Checkov JSON output into Finding objects."""
    findings: List[Finding] = []
    
    if not isinstance(data, dict):
        return findings
    
    # Checkov output structure: results_dict -> results -> failed_checks
    results_dict = data.get("results", {})
    if not isinstance(results_dict, dict):
        return findings
    
    failed_checks = results_dict.get("failed_checks", [])
    if not isinstance(failed_checks, list):
        return findings
    
    for check in failed_checks:
        if not isinstance(check, dict):
            continue
        
        # Extract relevant fields
        rule_id = check.get("check_id", "")
        severity = check.get("severity", "UNKNOWN")
        message = check.get("check_name", check.get("check_message", ""))
        file_path = check.get("file_path", str(tf_path))
        start_line = check.get("file_line_range", [None])[0] if isinstance(check.get("file_line_range"), list) else None
        
        findings.append(Finding(
            tool="checkov",
            rule_id=rule_id,
            severity=severity.upper(),
            message=message,
            file_path=file_path,
            start_line=start_line,
        ))
    
    return findings


def run_tfsec_on_file(tf_path: Path) -> Dict:
    """
    Run tfsec on a single Terraform file, returning parsed JSON (or {} on failure).
    
    Note: tfsec requires a directory, not a file. We'll run it on the file's parent directory
    and filter results to only this file. Also, tfsec returns exit code 1 when it finds issues.
    """
    output_path = SCANS_ROOT / "tfsec" / (tf_path.name + ".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # tfsec requires a directory, not a file
    # Since all our files are in TERRAFORM_FILES_ROOT, we can run on that directory
    # and filter results to only this file
    cmd = find_tfsec_command() + [
        str(tf_path.parent),
        "--format", "json",
        "--no-color",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # tfsec returns 1 when it finds issues, 0 when clean
        # Exit code 2+ indicates actual errors
        if result.returncode >= 2:
            return {}
        
        # Parse JSON output
        if result.stdout:
            try:
                data = json.loads(result.stdout)
                # Write raw output
                output_path.write_text(result.stdout, encoding="utf-8")
                return data
            except json.JSONDecodeError:
                return {}
        
        return {}
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return {}


def parse_tfsec_findings(data: Dict, tf_path: Path) -> List[Finding]:
    """Parse tfsec JSON output into Finding objects."""
    findings: List[Finding] = []
    
    if not isinstance(data, dict):
        return findings
    
    results = data.get("results", [])
    if not isinstance(results, list):
        return findings
    
    for result in results:
        if not isinstance(result, dict):
            continue
        
        # Filter to only this file
        result_file = result.get("location", {}).get("filename", "")
        if result_file != str(tf_path) and not str(tf_path).endswith(result_file):
            continue
        
        rule_id = result.get("rule_id", "")
        severity = result.get("severity", "UNKNOWN")
        message = result.get("description", "")
        file_path = result.get("location", {}).get("filename", str(tf_path))
        start_line = result.get("location", {}).get("start_line")
        
        findings.append(Finding(
            tool="tfsec",
            rule_id=rule_id,
            severity=severity.upper(),
            message=message,
            file_path=file_path,
            start_line=start_line,
        ))
    
    return findings


def is_secure(checkov_findings: List[Finding], tfsec_findings: List[Finding]) -> bool:
    """Determine if file is secure (no HIGH or CRITICAL findings)."""
    all_findings = checkov_findings + tfsec_findings
    for finding in all_findings:
        if finding.severity in ["HIGH", "CRITICAL"]:
            return False
    return True


def main() -> None:
    ensure_directories()
    
    print(f"Scanning Terraform files in {TERRAFORM_FILES_ROOT}")
    
    tf_files = list(TERRAFORM_FILES_ROOT.glob("*.tf"))
    print(f"Found {len(tf_files)} Terraform files")
    
    labels_path = TERRAFORM_FILES_ROOT.parent / "file_labels.jsonl"
    
    with labels_path.open("w", encoding="utf-8") as f:
        for tf_path in tqdm(tf_files, desc="Scanning files"):
            # Run scanners
            checkov_data = run_checkov_on_file(tf_path)
            tfsec_data = run_tfsec_on_file(tf_path)
            
            # Parse findings
            checkov_findings = parse_checkov_findings(checkov_data, tf_path)
            tfsec_findings = parse_tfsec_findings(tfsec_data, tf_path)
            
            # Determine security status
            secure = is_secure(checkov_findings, tfsec_findings)
            
            # Write label
            label = {
                "corpus_path": str(tf_path),
                "secure": secure,
                "tool_findings": {
                    "checkov": [asdict(f) for f in checkov_findings],
                    "tfsec": [asdict(f) for f in tfsec_findings],
                }
            }
            f.write(json.dumps(label, ensure_ascii=False) + "\n")
    
    print(f"Wrote labels to {labels_path}")


if __name__ == "__main__":
    main()


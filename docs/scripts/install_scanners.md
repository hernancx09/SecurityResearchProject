# `install_scanners.sh`

**Purpose**: Install and verify the two rule-based security scanners used in this project (Checkov and tfsec).

## Overview

This bash script automates the installation of Checkov and tfsec, which are required dependencies for the scanning phase of the Axon InfraLinter pipeline. It checks for existing installations, installs missing tools, and verifies that both scanners work correctly.

## Usage

```bash
bash install_scanners.sh
```

**Prerequisites**: 
- Python virtual environment activated (script will warn if not)
- `pip` available for Checkov installation
- `wget` and `curl` for tfsec download (typically pre-installed on Linux)

## What It Does

### 1. Virtual Environment Check

Checks if a Python virtual environment is active by examining `$VIRTUAL_ENV`. Warns if not found and prompts to continue anyway.

**Student understanding**: Virtual environments isolate project dependencies. Running this script outside a venv could install packages globally, which may conflict with other projects.

### 2. Checkov Installation

- **Method**: `pip install checkov`
- **Verification**: Runs `python -m checkov --version` to confirm installation
- **Fallback**: Suggests `pip install --upgrade checkov` if verification fails

**Student understanding**: Checkov is a Python package, so `pip` is the standard installation method. The `--version` check ensures the installation worked and the command is accessible.

### 3. tfsec Installation

- **Checks existing installation**: Looks for `tfsec` in PATH
- **Downloads latest release**: Uses GitHub API to find latest tfsec version, then downloads the Linux AMD64 binary
- **Installation locations** (in order of preference):
  1. `~/.local/bin/tfsec` (user-local, no sudo needed)
  2. `/usr/local/bin/tfsec` (system-wide, requires sudo)
- **Sets permissions**: Makes binary executable with `chmod +x`
- **Verification**: Runs `tfsec --version` to confirm installation

**Student understanding**: 
- **tfsec is a binary**: Unlike Checkov, tfsec is a compiled Go binary, not a Python package. We download the pre-built binary rather than compiling from source.
- **User-local vs. system-wide**: Installing to `~/.local/bin` doesn't require sudo and is safer (doesn't affect system packages). However, you need to ensure `~/.local/bin` is in your PATH.
- **GitHub API**: The script uses `curl` to query GitHub's releases API to find the latest version automatically, so it stays up-to-date without manual updates.

### 4. Verification

After installation, the script verifies both tools:
- Checkov: `python -m checkov --version`
- tfsec: `tfsec --version`

If verification fails, it prints troubleshooting suggestions.

## Output Example

```
==========================================
Installing Security Scanners
==========================================

1. Installing Checkov...
✓ Checkov installed successfully
checkov 2.3.123

2. Installing tfsec...
Downloading tfsec...
Latest version: v1.28.1
✓ tfsec installed to ~/.local/bin/tfsec
  Make sure ~/.local/bin is in your PATH

Verifying tfsec installation...
✓ tfsec installed successfully
tfsec version v1.28.1

==========================================
Installation complete!
==========================================
```

## Troubleshooting

### Checkov not found

- Ensure virtual environment is activated
- Try: `pip install --upgrade checkov`
- Check: `python -m checkov --version`

### tfsec not found in PATH

- If installed to `~/.local/bin`, add to PATH:
  ```bash
  export PATH="$HOME/.local/bin:$PATH"
  ```
- Or add to `~/.bashrc` or `~/.zshrc` for persistence

### Permission denied (tfsec)

- Ensure binary is executable: `chmod +x ~/.local/bin/tfsec`
- If installing to `/usr/local/bin`, may need sudo

## Student Understanding

**Why automate installation?**
- **Reproducibility**: Ensures everyone uses the same tool versions
- **Reduces setup friction**: One command instead of manual installation steps
- **Error prevention**: Catches common issues (missing venv, PATH problems) early

**Why verify installations?**
- **Catches errors immediately**: Better to fail during setup than during scanning
- **Confirms PATH configuration**: Verifies tools are accessible
- **Provides clear feedback**: Users know exactly what went wrong if verification fails

**Why check for existing installations?**
- **Idempotency**: Running the script multiple times is safe
- **Faster**: Skips unnecessary downloads/installs
- **User-friendly**: Doesn't overwrite existing installations

## Integration with Pipeline

This script is typically run once before starting the Axon InfraLinter pipeline:

```bash
# Setup
bash install_scanners.sh

# Then run pipeline
python -m axon_infralinter.data.github_scraper
python -m axon_infralinter.data.terraform_collector
python -m axon_infralinter.scanning.scanner  # Requires Checkov and tfsec
```

The scanning module (`axon_infralinter/scanning/scanner.py`) will automatically detect Checkov and tfsec using the same logic (checking PATH and common locations), so this script ensures they're installed and accessible.


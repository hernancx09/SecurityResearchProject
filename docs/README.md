# Axon InfraLinter Documentation

This directory contains comprehensive documentation for the Axon InfraLinter research project.

## Documentation Structure

### Core Documentation

- **[Overview](overview.md)** - System overview, pipeline diagram, and file map
- **[Research Notes](research_notes.md)** - Motivation, research questions, and key findings
- **[Artifacts](artifacts.md)** - Documentation of data files, models, and figures

### Module Documentation

- **[Data Modules](modules/axon_data.md)** - GitHub scraping and Terraform file collection
- **[Scanning Module](modules/axon_scanning.md)** - Security scanning with Checkov and tfsec
- **[Dataset Module](modules/axon_dataset.md)** - Balanced dataset building
- **[Models Module](modules/axon_models.md)** - Training, evaluation, and visualization
- **[CLI Module](modules/axon_cli.md)** - Command-line interface

### Script Documentation

- **[check_checkpoints.py](scripts/check_checkpoints.md)** - Checkpoint analysis utility
- **[install_scanners.sh](scripts/install_scanners.md)** - Scanner installation script

## Quick Start

1. Read the [Overview](overview.md) to understand the system architecture
2. Check [Research Notes](research_notes.md) for motivation and key learnings
3. Review module docs for specific components you're working with
4. Consult [Artifacts](artifacts.md) to understand data formats and model checkpoints

## Documentation Philosophy

This documentation is designed to:

- **Demonstrate student understanding**: Comments and docs explain not just *what* the code does, but *why* design decisions were made
- **Support reproducibility**: Clear explanations of data formats, model configurations, and experimental setup
- **Enable extension**: Detailed API documentation makes it easy to modify or extend the codebase
- **Show learning**: Explicit notes about pitfalls encountered, bugs fixed, and lessons learned

## Code Comments

The codebase itself contains extensive inline comments that:

- Explain security concepts (e.g., why tfsec exit codes matter)
- Highlight design tradeoffs (e.g., LoRA vs. full fine-tuning)
- Document pitfalls (e.g., class imbalance in datasets)
- Connect code decisions to research goals

These comments are integrated with the module documentation for a complete understanding.


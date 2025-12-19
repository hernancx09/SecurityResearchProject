# Axon InfraLinter – CLI Module

This document describes the command-line interface for running the trained model on new Terraform files.

## `axon_infralinter/cli/axon_infralinter_cli.py`

**Role**: Simple command-line interface to analyze Terraform files for security misconfigurations using the fine-tuned LLM.

### Overview

This module provides a minimal CLI that loads the fine-tuned model, accepts a path to a Terraform file, runs inference, and prints a human-readable security assessment.

### Key Functions

#### `load_model()`

Loads the fine-tuned model and tokenizer.

**Process**:
1. Loads base model (CodeLlama) from `BASE_MODEL_NAME`
2. Loads LoRA adapters from `LORA_OUTPUT_DIR` if available
3. Configures tokenizer (sets pad_token if missing)
4. Returns model and tokenizer

**Student understanding**: 
- **LoRA adapters**: The fine-tuned weights are stored as small adapter matrices. To use the model, we load the base model and then apply the adapters using `PeftModel.from_pretrained()`.
- **Device mapping**: Uses `device_map="auto"` to automatically handle GPU/CPU placement, which is convenient for different hardware configurations.

#### `build_prompt(file_text: str) -> str`

Constructs the instruction prompt (same format as used in training).

**Format**: Matches the prompt used in `build_dataset.py` to ensure consistency between training and inference.

**Student understanding**: Using the same prompt format as training is important - the model learned to respond to this specific format, so changing it could degrade performance.

#### `main(path: Path) -> None`

Main CLI entry point (using Click for argument parsing).

**Process**:
1. Loads model and tokenizer
2. Reads Terraform file from provided path
3. Builds prompt
4. Tokenizes input
5. Generates response (greedy decoding, max 50 tokens)
6. Extracts generated text (removes prompt)
7. Prints formatted output

**Generation parameters**:
- `max_new_tokens`: 50 (enough for SECURE/INSECURE + explanation)
- `do_sample`: False (deterministic)
- `temperature`: 0.1 (low temperature for consistent outputs)

**Output format**:
```
======================================================================
AXON INFRALINTER - SECURITY ANALYSIS
======================================================================

INSECURE
At least one high or critical security misconfiguration was detected in this file.
```

### Usage

```bash
python -m axon_infralinter.cli.axon_infralinter_cli path/to/file.tf
```

### Design Decisions

**Why minimal CLI?**

- **Focus on model behavior**: The CLI is intentionally simple to keep attention on what the model does, not on UI polish
- **Research prototype**: This is a research tool, not a production security scanner. Complex features (batch processing, output formats, etc.) can be added later if needed
- **Easy to extend**: The simple structure makes it easy to add features (e.g., JSON output, batch mode) without major refactoring

**Why human-readable output?**

- **Demonstration**: Clear text output is easy to show in demos or presentations
- **Copy-paste friendly**: Results can be easily copied into reports or documentation
- **Interpretability**: Natural language explanations are more accessible than raw model outputs

**Why deterministic generation?**

- **Reproducibility**: Same file always produces same output (important for research)
- **Consistency**: Users expect consistent results when analyzing the same file multiple times
- **Debugging**: Deterministic outputs make it easier to debug model behavior

### Limitations

- **Single file only**: No batch processing (can be added if needed)
- **Text output only**: No JSON or other formats (can be added)
- **No confidence scores**: Just outputs SECURE/INSECURE, not probability scores
- **Parsing fragility**: Relies on model following the expected output format

### Future Enhancements

- **Batch mode**: Process multiple files at once
- **JSON output**: Machine-readable format for integration
- **Confidence scores**: Probability that file is secure/insecure
- **Explanation extraction**: Parse and highlight specific issues mentioned in explanation
- **Comparison mode**: Compare LLM output with rule-based scanner results


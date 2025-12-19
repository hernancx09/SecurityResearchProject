# `check_checkpoints.py`

**Purpose**: Inspect training checkpoints to identify the best one based on validation loss.

## Overview

This utility script analyzes Hugging Face Trainer checkpoints to help select the best model for evaluation. It reads `trainer_state.json` files from checkpoint directories and identifies which checkpoint has the lowest validation loss.

## Usage

```bash
python check_checkpoints.py
```

By default, it analyzes checkpoints in `models/axon_lora_codellama/`.

## What It Does

1. **Finds checkpoints**: Searches for `checkpoint-*/trainer_state.json` files
2. **Extracts metrics**: Reads epoch, global step, and validation loss from each checkpoint
3. **Prints summary**: Displays information for each checkpoint:
   - Checkpoint name (e.g., `checkpoint-172`)
   - Epoch number
   - Global step
   - Validation loss (if available)
   - Best metric (if tracked)
   - Full path
4. **Identifies best**: Highlights the checkpoint with the lowest validation loss
5. **Shows resume command**: Prints how to resume training from the best checkpoint

## Output Example

```
======================================================================
CHECKPOINT ANALYSIS
======================================================================

Checkpoint: checkpoint-172
  Epoch: 1.50
  Global Step: 172
  Validation Loss: 0.234567
  Path: models/axon_lora_codellama/checkpoint-172

Checkpoint: checkpoint-516
  Epoch: 3.00
  Global Step: 516
  Validation Loss: 0.198765
  Path: models/axon_lora_codellama/checkpoint-516

======================================================================
[BEST] CHECKPOINT: checkpoint-516
  Validation Loss: 0.198765
  Path: models/axon_lora_codellama/checkpoint-516

To resume training from this checkpoint:
  Set resume_from_checkpoint="models/axon_lora_codellama/checkpoint-516" in TrainConfig
```

## Student Understanding

**Why checkpoints matter**: During training, the model's performance on the validation set fluctuates. The checkpoint with the lowest validation loss is typically the best model, not necessarily the final checkpoint. This script helps identify that checkpoint.

**Hugging Face Trainer state**: The `trainer_state.json` file contains rich metadata:
- Training history (losses at each step)
- Evaluation metrics (validation loss, best metric)
- Training progress (epoch, global step)
- Checkpoint paths

**Resuming training**: The script shows how to resume from a checkpoint, which is useful if:
- Training was interrupted
- You want to continue training beyond the original epoch limit
- You want to fine-tune further from a specific point

**Best checkpoint selection**: Choosing the checkpoint with lowest validation loss (rather than always using the final checkpoint) is a common practice in machine learning. It helps avoid overfitting - the model may perform better on validation data at an earlier epoch than at the end of training.


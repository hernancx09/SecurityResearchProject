# Checkpoint Guide for LoRA Training

## Current Checkpoint Analysis

Based on your training checkpoints:

| Checkpoint | Epoch | Validation Loss | Status |
|------------|-------|-----------------|--------|
| **checkpoint-172** | 1.0 | **7.763681** | ✅ **BEST** |
| checkpoint-344 | 2.0 | 7.765751 | Slightly worse |

**Best Checkpoint:** `checkpoint-172` (lowest validation loss)

## How to Check Checkpoints

Run the analysis script:
```bash
python check_checkpoints.py
```

This will show:
- All available checkpoints
- Their validation losses
- Which checkpoint is best
- How to resume from it

## How to Resume Training

### Method 1: Modify TrainConfig (Recommended)

Edit `axon_infralinter/models/train_lora.py`:

```python
@dataclass
class TrainConfig:
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    logging_steps: int = 50
    resume_from_checkpoint: Optional[str] = "models/axon_lora_codellama/checkpoint-172"  # Add this line
```

Then run:
```bash
python -m axon_infralinter.models.train_lora
```

### Method 2: Resume from Latest Checkpoint

If training crashed and you want to resume from the most recent checkpoint:

1. Find the latest checkpoint:
   ```bash
   ls -t models/axon_lora_codellama/checkpoint-* | head -1
   ```

2. Use that path in `TrainConfig.resume_from_checkpoint`

### Method 3: Resume from Specific Epoch

To resume from a specific epoch checkpoint:

```python
resume_from_checkpoint: Optional[str] = "models/axon_lora_codellama/checkpoint-344"  # Epoch 2
```

## What Happens When You Resume

- Training continues from the checkpoint's global step
- Optimizer and scheduler state are restored
- Random number generator state is restored (for reproducibility)
- Training continues for the remaining epochs

**Example:** If you resume from `checkpoint-172` (epoch 1, step 172) with `num_train_epochs=3`:
- Training will continue from step 172
- Will complete epochs 2 and 3
- Total steps: 516 (172 steps per epoch × 3 epochs)

## Checkpoint Contents

Each checkpoint contains:
- `adapter_model.safetensors` - LoRA weights (for inference)
- `adapter_config.json` - LoRA configuration
- `trainer_state.json` - Training progress, metrics, best checkpoint
- `optimizer.pt` - Optimizer state (needed for resuming)
- `scheduler.pt` - Learning rate scheduler state (needed for resuming)
- `rng_state.pth` - Random state (for reproducibility)

## Tips

1. **Best checkpoint is automatically loaded**: The final model in `models/axon_lora_codellama/` is the best checkpoint (lowest validation loss)

2. **Check training progress**: Look at `trainer_state.json` to see:
   - Current epoch
   - Training loss progression
   - Validation loss at each epoch
   - Which checkpoint was best

3. **If training crashes**: Resume from the latest checkpoint to avoid losing progress

4. **Compare checkpoints**: Use `check_checkpoints.py` to compare validation losses across epochs

5. **Manual checkpoint selection**: You can manually copy a checkpoint to the main directory:
   ```bash
   cp -r models/axon_lora_codellama/checkpoint-172/* models/axon_lora_codellama/
   ```

## Quick Reference

```bash
# Check all checkpoints
python check_checkpoints.py

# Resume from best checkpoint (checkpoint-172)
# Edit train_lora.py: resume_from_checkpoint = "models/axon_lora_codellama/checkpoint-172"
python -m axon_infralinter.models.train_lora

# Resume from latest checkpoint
# Edit train_lora.py: resume_from_checkpoint = "models/axon_lora_codellama/checkpoint-344"
python -m axon_infralinter.models.train_lora
```




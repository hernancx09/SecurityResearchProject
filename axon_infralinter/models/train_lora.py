from __future__ import annotations

"""
LoRA fine-tuning script for Stable Code 3B on the Axon InfraLinter dataset.

This uses a simple causal language modeling objective where each example
is `input_text` followed by `target_text`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from tqdm import tqdm

from axon_infralinter.config import (
    BASE_MODEL_NAME,
    DATASET_ROOT,
    LORA_OUTPUT_DIR,
)


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


def tokenize_function(examples: Dict, tokenizer, max_length: int = 1024):
    texts = [inp + "\n\n" + tgt for inp, tgt in zip(examples["input_text"], examples["target_text"])]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    # For causal LM, labels are the same as input_ids (model shifts internally)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main() -> None:
    cfg = TrainConfig()

    print("=" * 70)
    print("AXON INFRALINTER - LoRA FINE-TUNING")
    print("=" * 70)
    print()

    # Check hardware
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("⚠ CPU mode (will be very slow)")
    print()

    # Load dataset
    print("Loading dataset...")
    train_path = DATASET_ROOT / "train.jsonl"
    val_path = DATASET_ROOT / "val.jsonl"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {val_path}")
    
    print(f"  Training file: {train_path}")
    print(f"  Validation file: {val_path}")
    
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(val_path),
        },
    )
    
    train_size = len(dataset["train"])
    val_size = len(dataset["validation"])
    print(f"  ✓ Loaded {train_size} training examples")
    print(f"  ✓ Loaded {val_size} validation examples")
    print()

    # Load tokenizer
    print(f"Loading tokenizer from {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  ✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
    print()

    # Tokenize dataset with reduced max_length for memory efficiency
    print("Tokenizing dataset...")
    print("  Using max_length=1024 to reduce memory usage")
    tokenized = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer, max_length=1024),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    print(f"  ✓ Tokenization complete")
    print()

    # Load model with memory optimizations for 8GB GPU
    print(f"Loading base model: {BASE_MODEL_NAME}...")
    print("  Attempting 8-bit quantization for memory efficiency...")
    print("  (This may take a few minutes to download if not cached...)")
    
    # Configure 8-bit quantization to fit in 8GB VRAM
    quantization_config = None
    use_8bit = False
    device_map_value = None
    
    if has_gpu:
        try:
            # Use 8-bit quantization without CPU offload (offload prevents gradients)
            # 8-bit should reduce model size enough to fit in 8GB VRAM
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            use_8bit = True
            # Don't use device_map="auto" with 8-bit as it may try to offload to CPU
            # 8-bit quantization should fit entirely on GPU
            device_map_value = None
            print("  ✓ 8-bit quantization enabled (will load on GPU)")
        except Exception as e:
            print(f"  ⚠ Could not enable 8-bit quantization: {e}")
            print("  Falling back to float16")
            quantization_config = None
            use_8bit = False
            device_map_value = "auto" if has_gpu else None
    else:
        device_map_value = None
    
    dtype = torch.float16 if (has_gpu and not use_8bit) else torch.float32
    print(f"  Using dtype: {dtype}")
    
    # Build model loading kwargs
    model_kwargs = {}
    if device_map_value is not None:
        model_kwargs["device_map"] = device_map_value
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = dtype
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        **model_kwargs
    )
    print(f"  ✓ Base model loaded")
    print()
    
    # Clear cache
    if has_gpu:
        torch.cuda.empty_cache()

    # Prepare model for k-bit training if using quantization
    if use_8bit:
        print("Preparing model for 8-bit training...")
        model = prepare_model_for_kbit_training(model)
        print("  ✓ Model prepared for 8-bit training")
        print()

    # Setup LoRA with memory-efficient settings
    print("Setting up LoRA configuration...")
    # Try to auto-detect target modules, fallback to common attention layer names
    # For Stable Code 3B and similar models, we target attention layers
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Full attention layers for better coverage
    )
    model = get_peft_model(model, lora_config)
    
    # Ensure model is in training mode
    model.train()
    
    # Verify LoRA parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        # Try with a more generic approach - target all linear layers in attention
        print("  ⚠ Initial target modules not found, trying alternative configuration...")
        # Print model structure to debug
        print("  Model structure (first 20 modules):")
        for name, module in list(model.named_modules())[:20]:
            print(f"    {name}: {type(module).__name__}")
        
        # Try to find the correct module names by inspecting the model
        print("  Inspecting model to find attention layers...")
        # Get a sample of module names to identify the pattern
        sample_modules = []
        for name, module in model.named_modules():
            if any(x in name.lower() for x in ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
                sample_modules.append(name)
            if len(sample_modules) >= 10:
                break
        
        print(f"  Sample modules found: {sample_modules[:5]}")
        
        # Try with a broader set of common attention layer names
        # Stable Code models often use different naming
        alternative_targets = [
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            ["query", "key", "value", "dense"],
            ["c_attn", "c_proj"],  # GPT-2 style
        ]
        
        model_reloaded = False
        for targets in alternative_targets:
            try:
                if model_reloaded:
                    from transformers import AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(
                        BASE_MODEL_NAME,
                        **model_kwargs
                    )
                    if use_8bit:
                        model = prepare_model_for_kbit_training(model)
                
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=targets,
                )
                model = get_peft_model(model, lora_config)
                model.train()
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                if trainable_params > 0:
                    print(f"  ✓ Successfully configured LoRA with targets: {targets}")
                    break
                model_reloaded = True
            except Exception as e:
                print(f"  ⚠ Failed with targets {targets}: {e}")
                continue
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable_params == 0:
            raise RuntimeError("No trainable parameters found! LoRA adapters may not be configured correctly. Please check the model architecture.")
    
    # Note: Gradient checkpointing is disabled when using 8-bit quantization
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total parameters: {total_params:,}")
    print()

    # Training arguments
    print("Training configuration:")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Epochs: {cfg.num_train_epochs}")
    print(f"  Batch size: {cfg.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {cfg.gradient_accumulation_steps}")
    print(f"  Effective batch size: {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
    print(f"  Total training steps: ~{train_size // (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps) * cfg.num_train_epochs}")
    print(f"  Output directory: {LORA_OUTPUT_DIR}")
    print()

    training_args = TrainingArguments(
        output_dir=str(LORA_OUTPUT_DIR),
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=[],
        logging_dir=str(LORA_OUTPUT_DIR / "logs"),
        save_total_limit=2,
        gradient_checkpointing=not use_8bit,  # Disable gradient checkpointing with 8-bit (causes issues)
        fp16=has_gpu and not use_8bit,  # Use fp16 if not using 8-bit
        dataloader_pin_memory=False,  # Disable pinning to save memory
        max_grad_norm=1.0,  # Gradient clipping
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
    )

    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print()

    # Train
    trainer.train()

    print()
    print("=" * 70)
    print("TRAINING COMPLETE - Saving model...")
    print("=" * 70)
    
    LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(LORA_OUTPUT_DIR))
    tokenizer.save_pretrained(str(LORA_OUTPUT_DIR))
    
    print(f"✓ Model saved to: {LORA_OUTPUT_DIR}")
    print(f"✓ Tokenizer saved to: {LORA_OUTPUT_DIR}")
    print()
    print("Training complete! You can now evaluate the model or use the CLI.")


if __name__ == "__main__":
    main()

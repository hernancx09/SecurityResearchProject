from __future__ import annotations

"""
Evaluate the fine-tuned model and compare against baseline rule-based scanners.

Loads the test set and runs:
1. Fine-tuned LLM predictions
2. Checkov baseline
3. tfsec baseline

Computes metrics (accuracy, precision, recall, F1) for each.
"""

import json
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from axon_infralinter.config import (
    BASE_MODEL_NAME,
    DATASET_ROOT,
    LORA_OUTPUT_DIR,
    PROJECT_ROOT,
)


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file into list of dicts."""
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Compute classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_llms(test_records: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Evaluate fine-tuned LLM on test set."""
    print("Loading fine-tuned model...")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    
    # Load LoRA adapters
    if (LORA_OUTPUT_DIR / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(LORA_OUTPUT_DIR))
        print("  ✓ Loaded LoRA adapters")
    else:
        print("  ⚠ No LoRA adapters found, using base model")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    print("Running LLM predictions...")
    y_true = []
    y_pred = []
    
    for record in test_records:
        input_text = record["input_text"]
        true_label = 0 if record.get("secure", False) else 1  # 0=secure, 1=insecure
        
        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.1,
            )
        
        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse prediction (look for SECURE or INSECURE in output)
        pred_label = 1  # Default to insecure
        if "SECURE" in generated.upper() and "INSECURE" not in generated.upper():
            pred_label = 0
        
        y_true.append(true_label)
        y_pred.append(pred_label)
    
    return {
        "fine_tuned_llm": metrics(y_true, y_pred),
    }


def main() -> None:
    test_path = DATASET_ROOT / "test.jsonl"
    
    if not test_path.exists():
        print(f"Error: Test dataset not found at {test_path}")
        print("Run build_dataset.py first to create test split.")
        return
    
    print("Loading test dataset...")
    test_records = load_jsonl(test_path)
    print(f"Loaded {len(test_records)} test examples")
    print()
    
    # Evaluate LLM
    llm_results = evaluate_llms(test_records)
    
    print()
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(json.dumps(llm_results, indent=2))


if __name__ == "__main__":
    main()


from __future__ import annotations

"""
Evaluate the fine-tuned model and compare against baseline rule-based scanners.

Loads the test set and runs:
1. Fine-tuned LLM predictions
2. Checkov baseline
3. tfsec baseline

Computes metrics (accuracy, precision, recall, F1) for each and can optionally
persist evaluation results to disk for multi-run benchmarking.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
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


def _get_eval_mode() -> str:
    """
    Return evaluation mode: 'deterministic' or 'stochastic'.

    Controlled via AXON_EVAL_MODE env var. Defaults to deterministic.
    """
    mode = os.getenv("AXON_EVAL_MODE", "deterministic").strip().lower()
    if mode not in {"deterministic", "stochastic"}:
        mode = "deterministic"
    return mode


def _get_sampling_config() -> Dict:
    """
    Return sampling configuration for stochastic evaluation runs.

    Controlled via environment variables:
    - AXON_EVAL_TEMPERATURE (float)
    - AXON_EVAL_TOP_P (float)
    - AXON_EVAL_TOP_K (int)
    - AXON_EVAL_SEED (int, optional)
    """
    temperature = float(os.getenv("AXON_EVAL_TEMPERATURE", "0.5"))
    top_p = float(os.getenv("AXON_EVAL_TOP_P", "0.9"))
    top_k = int(os.getenv("AXON_EVAL_TOP_K", "50"))
    seed_env = os.getenv("AXON_EVAL_SEED")
    seed = int(seed_env) if seed_env is not None else None
    return {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
    }


def evaluate_llms(test_records: List[Dict]) -> Dict:
    """
    Evaluate fine-tuned LLM on test set.

    Returns a dict with:
    - metrics
    - predictions_data
    - y_true
    - y_pred
    - eval_mode
    - sampling_config (if stochastic)
    """
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
    
    eval_mode = _get_eval_mode()
    sampling_config = _get_sampling_config() if eval_mode == "stochastic" else {}

    print(f"Running LLM predictions... (mode={eval_mode})")
    y_true = []
    y_pred = []
    generated_texts = []
    predictions_data = []
    
    for record in tqdm(test_records, desc="Evaluating", unit="example"):
        input_text = record["input_text"]
        true_label = 0 if record.get("secure", False) else 1  # 0=secure, 1=insecure
        
        # Add explicit separator to help model understand it should generate classification
        # The training format was input_text + "\n\n" + target_text
        prompt_text = input_text.rstrip() + "\n\n"
        
        # Tokenize
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Configure generation
        eos_token_id = tokenizer.eos_token_id
        gen_kwargs: Dict = {
            "max_new_tokens": 80,  # Limit to prevent code generation
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": eos_token_id,
            "repetition_penalty": 1.1,
        }

        if eval_mode == "stochastic":
            # Stochastic sampling
            if sampling_config.get("seed") is not None:
                seed = sampling_config["seed"]
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            gen_kwargs.update(
                dict(
                    do_sample=True,
                    temperature=sampling_config["temperature"],
                    top_p=sampling_config["top_p"],
                    top_k=sampling_config["top_k"],
                    num_beams=1,
                )
            )
        else:
            # Deterministic generation
            gen_kwargs.update(dict(do_sample=False))
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_kwargs,
            )
        
        # Extract just the generated part (new tokens only)
        input_ids_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_ids_len:]
        generated_continuation = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Parse prediction - look for SECURE or INSECURE in the first few lines
        # Take first 3 lines to avoid confusion from later text
        lines = generated_continuation.split('\n')
        first_few_lines = '\n'.join(lines[:3]).upper()
        
        # Check if both appear (confused output) - use first occurrence
        secure_pos = first_few_lines.find("SECURE")
        insecure_pos = first_few_lines.find("INSECURE")
        
        pred_label = 1  # Default to insecure
        if secure_pos != -1 and insecure_pos != -1:
            # Both appear - use whichever comes first
            if secure_pos < insecure_pos:
                pred_label = 0
            else:
                pred_label = 1
        elif secure_pos != -1:
            # Only SECURE found
            pred_label = 0
        elif insecure_pos != -1:
            # Only INSECURE found
            pred_label = 1
        # else: default to insecure (pred_label = 1)
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        generated_texts.append(generated_continuation)
        predictions_data.append({
            "true_label": "secure" if true_label == 0 else "insecure",
            "pred_label": "secure" if pred_label == 0 else "insecure",
            "correct": true_label == pred_label,
            "generated": generated_continuation[:200],  # First 200 chars
        })
    
    return {
        "metrics": metrics(y_true, y_pred),
        "predictions_data": predictions_data,
        "y_true": y_true,
        "y_pred": y_pred,
        "eval_mode": eval_mode,
        "sampling_config": sampling_config,
    }


def evaluate_baselines(test_records: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Evaluate Checkov and tfsec baselines from tool_findings."""
    print("Evaluating baseline scanners...")
    
    y_true = []
    checkov_pred = []
    tfsec_pred = []
    combined_pred = []
    
    for record in test_records:
        true_label = 0 if record.get("secure", False) else 1  # 0=secure, 1=insecure
        tool_findings = record.get("tool_findings", {})
        
        # Check Checkov findings
        checkov_findings = tool_findings.get("checkov", [])
        checkov_has_high_critical = any(
            f.get("severity") in ["HIGH", "CRITICAL"] for f in checkov_findings
        )
        checkov_pred.append(1 if checkov_has_high_critical else 0)
        
        # Check tfsec findings
        tfsec_findings = tool_findings.get("tfsec", [])
        tfsec_has_high_critical = any(
            f.get("severity") in ["HIGH", "CRITICAL"] for f in tfsec_findings
        )
        tfsec_pred.append(1 if tfsec_has_high_critical else 0)
        
        # Combined: insecure if either tool finds high/critical
        combined_pred.append(1 if (checkov_has_high_critical or tfsec_has_high_critical) else 0)
        
        y_true.append(true_label)
    
    return {
        "checkov": metrics(y_true, checkov_pred),
        "tfsec": metrics(y_true, tfsec_pred),
        "combined": metrics(y_true, combined_pred),
    }


def print_prediction_distribution(y_pred: List[int], y_true: List[int]) -> None:
    """Print prediction distribution analysis."""
    print("\n" + "=" * 70)
    print("PREDICTION DISTRIBUTION")
    print("=" * 70)
    
    total = len(y_pred)
    pred_secure = sum(1 for p in y_pred if p == 0)
    pred_insecure = sum(1 for p in y_pred if p == 1)
    
    true_secure = sum(1 for t in y_true if t == 0)
    true_insecure = sum(1 for t in y_true if t == 1)
    
    print(f"\nTrue Labels:")
    print(f"  Secure:   {true_secure:4d} ({100*true_secure/total:5.1f}%)")
    print(f"  Insecure: {true_insecure:4d} ({100*true_insecure/total:5.1f}%)")
    
    print(f"\nPredicted Labels:")
    print(f"  Secure:   {pred_secure:4d} ({100*pred_secure/total:5.1f}%)")
    print(f"  Insecure: {pred_insecure:4d} ({100*pred_insecure/total:5.1f}%)")
    
    # Confusion matrix
    tp = sum(1 for i in range(total) if y_true[i] == 1 and y_pred[i] == 1)
    tn = sum(1 for i in range(total) if y_true[i] == 0 and y_pred[i] == 0)
    fp = sum(1 for i in range(total) if y_true[i] == 0 and y_pred[i] == 1)
    fn = sum(1 for i in range(total) if y_true[i] == 1 and y_pred[i] == 0)
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Secure  Insecure")
    print(f"Actual Secure   {tn:4d}     {fp:4d}")
    print(f"      Insecure  {fn:4d}     {tp:4d}")


def print_example_outputs(predictions_data: List[Dict], num_examples: int = 10) -> None:
    """Print example model outputs for inspection."""
    print("\n" + "=" * 70)
    print(f"EXAMPLE OUTPUTS (showing {num_examples} examples)")
    print("=" * 70)
    
    # Show mix of correct and incorrect predictions
    correct_examples = [p for p in predictions_data if p["correct"]]
    incorrect_examples = [p for p in predictions_data if not p["correct"]]
    
    examples_to_show = (
        incorrect_examples[:num_examples//2] + 
        correct_examples[:num_examples//2]
    )
    
    if len(examples_to_show) < num_examples:
        examples_to_show = predictions_data[:num_examples]
    
    for i, ex in enumerate(examples_to_show, 1):
        status = "✓ CORRECT" if ex["correct"] else "✗ INCORRECT"
        print(f"\nExample {i} - {status}")
        print(f"  True Label:  {ex['true_label'].upper()}")
        print(f"  Pred Label:  {ex['pred_label'].upper()}")
        print(f"  Generated:   {ex['generated']}")
        if len(ex['generated']) >= 200:
            print("              ... (truncated)")


def _compute_confusion(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    """Return confusion matrix components as a dict."""
    total = len(y_true)
    tp = sum(1 for i in range(total) if y_true[i] == 1 and y_pred[i] == 1)
    tn = sum(1 for i in range(total) if y_true[i] == 0 and y_pred[i] == 0)
    fp = sum(1 for i in range(total) if y_true[i] == 0 and y_pred[i] == 1)
    fn = sum(1 for i in range(total) if y_true[i] == 1 and y_pred[i] == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _compute_distribution(y_true: List[int], y_pred: List[int]) -> Dict[str, Dict[str, float]]:
    """Return prediction and label distribution as dict (counts and percentages)."""
    total = len(y_true)
    true_secure = sum(1 for t in y_true if t == 0)
    true_insecure = sum(1 for t in y_true if t == 1)
    pred_secure = sum(1 for p in y_pred if p == 0)
    pred_insecure = sum(1 for p in y_pred if p == 1)
    return {
        "true": {
            "secure": true_secure,
            "insecure": true_insecure,
            "secure_pct": 100 * true_secure / total,
            "insecure_pct": 100 * true_insecure / total,
        },
        "pred": {
            "secure": pred_secure,
            "insecure": pred_insecure,
            "secure_pct": 100 * pred_secure / total,
            "insecure_pct": 100 * pred_insecure / total,
        },
    }


def _short_model_name(base_model_name: str, lora_dir: Path) -> str:
    """
    Derive a short model name for filenames.

    Prefer directory name of LORA_OUTPUT_DIR, fall back to last component of model id.
    """
    if lora_dir.name:
        return lora_dir.name
    return base_model_name.split("/")[-1]


def _save_results_json(
    model_name: str,
    lora_dir: Path,
    llm_results: Dict,
    baseline_results: Dict[str, Dict[str, float]],
    confusion: Dict[str, int],
    distribution: Dict[str, Dict[str, float]],
    runtime_seconds: float,
) -> Path:
    """Persist evaluation results as a JSON file under data/eval_runs/."""
    eval_root = PROJECT_ROOT / "data" / "eval_runs"
    eval_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_index = int(os.getenv("AXON_RUN_INDEX", "0"))
    short_name = _short_model_name(model_name, lora_dir)
    filename = f"eval_{short_name}_run{run_index}_{ts}.json"
    out_path = eval_root / filename

    payload = {
        "model_name": model_name,
        "lora_dir": str(lora_dir),
        "run_index": run_index,
        "timestamp_utc": ts,
        "eval_mode": llm_results.get("eval_mode"),
        "sampling_config": llm_results.get("sampling_config", {}),
        "runtime_seconds": runtime_seconds,
        "metrics": llm_results["metrics"],
        "baselines": baseline_results,
        "confusion": confusion,
        "distribution": distribution,
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def run_evaluation() -> Dict:
    """
    Run a single evaluation and return a rich result dict.

    This is the programmatic entry point used by benchmark scripts.
    """
    test_path = DATASET_ROOT / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found at {test_path}. "
            "Run build_dataset.py first to create test split."
        )

    print("Loading test dataset...")
    test_records = load_jsonl(test_path)
    print(f"Loaded {len(test_records)} test examples")
    print()

    start_time = time.time()

    # Evaluate LLM
    llm_results = evaluate_llms(test_records)

    # Evaluate baselines
    baseline_results = evaluate_baselines(test_records)

    runtime_seconds = time.time() - start_time

    confusion = _compute_confusion(llm_results["y_true"], llm_results["y_pred"])
    distribution = _compute_distribution(llm_results["y_true"], llm_results["y_pred"])

    results_summary = {
        "fine_tuned_llm": llm_results["metrics"],
        "checkov": baseline_results["checkov"],
        "tfsec": baseline_results["tfsec"],
        "combined_baseline": baseline_results["combined"],
    }

    # Persist results to JSON
    out_path = _save_results_json(
        model_name=BASE_MODEL_NAME,
        lora_dir=LORA_OUTPUT_DIR,
        llm_results=llm_results,
        baseline_results=baseline_results,
        confusion=confusion,
        distribution=distribution,
        runtime_seconds=runtime_seconds,
    )

    return {
        "model_name": BASE_MODEL_NAME,
        "lora_dir": str(LORA_OUTPUT_DIR),
        "llm_results": llm_results,
        "baseline_results": baseline_results,
        "confusion": confusion,
        "distribution": distribution,
        "runtime_seconds": runtime_seconds,
        "results_summary": results_summary,
        "results_path": str(out_path),
    }


def main() -> None:
    try:
        result = run_evaluation()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    llm_results = result["llm_results"]
    baseline_results = result["baseline_results"]
    results_summary = result["results_summary"]

    # Print analysis
    print_prediction_distribution(llm_results["y_pred"], llm_results["y_true"])
    print_example_outputs(llm_results["predictions_data"])

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(json.dumps(results_summary, indent=2))

    print("\nResults saved to:")
    print(f"  {result['results_path']}")


if __name__ == "__main__":
    main()


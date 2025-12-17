from __future__ import annotations

"""
Run multi-run evaluation benchmarks for different fine-tuned models.

This script runs N evaluation runs per model, using stochastic sampling,
and aggregates metrics (mean/std) and runtimes for research reporting.

Usage examples (from project root, in your venv):

    # Run 5 stochastic evaluations for both models
    python -m axon_infralinter.models.benchmark_multi_run --model both --runs 5

    # Only CodeLlama
    python -m axon_infralinter.models.benchmark_multi_run --model codellama --runs 5

    # Only Stable Code
    python -m axon_infralinter.models.benchmark_multi_run --model stable_code --runs 5
"""

import argparse
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal

from axon_infralinter.config import PROJECT_ROOT
from axon_infralinter.models import evaluate as eval_module


ModelKey = Literal["stable_code", "codellama"]


@dataclass
class ModelConfig:
    key: ModelKey
    base_model: str
    lora_dir: Path


def get_model_configs() -> Dict[ModelKey, ModelConfig]:
    root = PROJECT_ROOT
    return {
        "stable_code": ModelConfig(
            key="stable_code",
            base_model="stabilityai/stable-code-3b",
            lora_dir=root / "models" / "axon_lora",
        ),
        "codellama": ModelConfig(
            key="codellama",
            base_model="codellama/CodeLlama-7b-Instruct-hf",
            lora_dir=root / "models" / "axon_lora_codellama",
        ),
    }


def set_env_for_model(mc: ModelConfig, run_index: int) -> None:
    """
    Configure environment variables for a given model and run.

    - AXON_BASE_MODEL: base HF model id
    - AXON_LORA_DIR: LoRA adapter directory
    - AXON_EVAL_MODE: 'stochastic'
    - AXON_EVAL_TEMPERATURE / TOP_P / TOP_K
    - AXON_EVAL_SEED: per-run seed
    - AXON_RUN_INDEX: numeric run index
    """
    os.environ["AXON_BASE_MODEL"] = mc.base_model
    os.environ["AXON_LORA_DIR"] = str(mc.lora_dir)

    # Stochastic evaluation config
    os.environ["AXON_EVAL_MODE"] = "stochastic"
    os.environ.setdefault("AXON_EVAL_TEMPERATURE", "0.5")
    os.environ.setdefault("AXON_EVAL_TOP_P", "0.9")
    os.environ.setdefault("AXON_EVAL_TOP_K", "50")
    # Simple per-run seed
    os.environ["AXON_EVAL_SEED"] = str(42 + run_index)

    os.environ["AXON_RUN_INDEX"] = str(run_index)


def aggregate_metric(runs: List[Dict], metric_key: str) -> Dict[str, float]:
    values = [r["results_summary"]["fine_tuned_llm"][metric_key] for r in runs]
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def aggregate_runtime(runs: List[Dict]) -> Dict[str, float]:
    values = [r["runtime_seconds"] for r in runs]
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def run_benchmark_for_model(mc: ModelConfig, runs: int) -> Dict:
    print("=" * 70)
    print(f"BENCHMARKING MODEL: {mc.key} ({mc.base_model})")
    print("=" * 70)
    print(f"LoRA dir: {mc.lora_dir}")
    print(f"Runs: {runs}")
    print()

    per_run_results: List[Dict] = []

    for i in range(runs):
        print("-" * 70)
        print(f"Run {i + 1}/{runs}")
        set_env_for_model(mc, run_index=i)
        result = eval_module.run_evaluation()
        per_run_results.append(result)
        print()

    # Aggregate metrics
    metrics_summary = {
        "accuracy": aggregate_metric(per_run_results, "accuracy"),
        "precision": aggregate_metric(per_run_results, "precision"),
        "recall": aggregate_metric(per_run_results, "recall"),
        "f1": aggregate_metric(per_run_results, "f1"),
    }
    runtime_summary = aggregate_runtime(per_run_results)

    summary = {
        "model_key": mc.key,
        "base_model": mc.base_model,
        "lora_dir": str(mc.lora_dir),
        "num_runs": runs,
        "metrics_summary": metrics_summary,
        "runtime_summary": runtime_summary,
    }

    # Save summary JSON
    eval_root = PROJECT_ROOT / "data" / "eval_runs"
    eval_root.mkdir(parents=True, exist_ok=True)
    summary_path = eval_root / f"summary_{mc.key}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Pretty print summary
    print("=" * 70)
    print(f"SUMMARY FOR MODEL: {mc.key}")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {summary_path}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-run evaluation benchmarks for fine-tuned models."
    )
    parser.add_argument(
        "--model",
        choices=["stable_code", "codellama", "both"],
        default="both",
        help="Which model(s) to benchmark.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of runs per model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = get_model_configs()

    models_to_run: List[ModelConfig]
    if args.model == "both":
        models_to_run = [configs["stable_code"], configs["codellama"]]
    else:
        models_to_run = [configs[args.model]]

    all_summaries = {}
    for mc in models_to_run:
        summary = run_benchmark_for_model(mc, runs=args.runs)
        all_summaries[mc.key] = summary

    # Save overall summary
    eval_root = PROJECT_ROOT / "data" / "eval_runs"
    overall_path = eval_root / "summary_all_models.json"
    overall_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print("\nOverall summary saved to:", overall_path)


if __name__ == "__main__":
    main()



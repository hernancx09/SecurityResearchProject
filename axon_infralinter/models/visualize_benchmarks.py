from __future__ import annotations

"""
Visualization utilities for multi-run benchmark results.

Loads per-run and summary JSON files from data/eval_runs/ and produces plots:
- Bar charts with error bars (mean ± std) for metrics per model
- Runtime comparison bar chart
- Optional boxplots for F1 across runs

Figures are saved under figures/.
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from axon_infralinter.config import PROJECT_ROOT


EVAL_ROOT = PROJECT_ROOT / "data" / "eval_runs"
FIG_ROOT = PROJECT_ROOT / "figures"


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_files() -> pd.DataFrame:
    """Load all per-run eval_*.json files into a DataFrame."""
    if not EVAL_ROOT.exists():
        raise FileNotFoundError(f"{EVAL_ROOT} does not exist. Run benchmarks first.")

    rows: List[Dict] = []
    for path in EVAL_ROOT.glob("eval_*.json"):
        data = _load_json(path)
        model_name = data.get("model_name", "")
        lora_dir = data.get("lora_dir", "")
        key = Path(lora_dir).name or model_name.split("/")[-1]
        m = data["metrics"]
        rows.append(
            {
                "model_key": key,
                "model_name": model_name,
                "run_index": data.get("run_index", 0),
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "runtime_seconds": data.get("runtime_seconds", None),
            }
        )

    if not rows:
        raise RuntimeError(f"No eval_*.json files found in {EVAL_ROOT}")

    return pd.DataFrame(rows)


def plot_metric_bars(df: pd.DataFrame) -> None:
    """Bar charts with error bars for accuracy, precision, recall, F1."""
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    metrics = ["accuracy", "precision", "recall", "f1"]

    summary = (
        df.groupby("model_key")[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    # Flatten MultiIndex columns
    summary.columns = ["model_key"] + [f"{m}_{s}" for m in metrics for s in ["mean", "std"]]

    melted_rows = []
    for _, row in summary.iterrows():
        for m in metrics:
            melted_rows.append(
                {
                    "model_key": row["model_key"],
                    "metric": m,
                    "mean": row[f"{m}_mean"],
                    "std": row[f"{m}_std"],
                }
            )
    mdf = pd.DataFrame(melted_rows)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=mdf,
        x="metric",
        y="mean",
        hue="model_key",
        ci=None,
    )
    # Add error bars manually
    for i, row in mdf.iterrows():
        plt.errorbar(
            x=i % len(metrics),
            y=row["mean"],
            yerr=row["std"],
            fmt="none",
            ecolor="black",
            capsize=3,
        )

    plt.ylim(0.0, 1.05)
    plt.title("Model Metrics (mean ± std over runs)")
    plt.ylabel("Score")
    plt.tight_layout()
    out_path = FIG_ROOT / "metrics_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)


def plot_runtime_bars(df: pd.DataFrame) -> None:
    """Bar chart comparing mean runtime per model."""
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    summary = (
        df.groupby("model_key")["runtime_seconds"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=summary,
        x="model_key",
        y="mean",
        ci=None,
    )
    # Error bars
    for i, row in summary.iterrows():
        plt.errorbar(
            x=i,
            y=row["mean"],
            yerr=row["std"],
            fmt="none",
            ecolor="black",
            capsize=3,
        )

    plt.title("Runtime per Model (mean ± std over runs)")
    plt.ylabel("Runtime (seconds)")
    plt.xlabel("Model")
    plt.tight_layout()
    out_path = FIG_ROOT / "runtime_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)


def plot_f1_boxplot(df: pd.DataFrame) -> None:
    """Boxplot of F1 scores per model."""
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="model_key", y="f1")
    sns.stripplot(
        data=df,
        x="model_key",
        y="f1",
        color="black",
        alpha=0.6,
    )
    plt.title("F1 Distribution Across Runs")
    plt.ylabel("F1 score")
    plt.xlabel("Model")
    plt.tight_layout()
    out_path = FIG_ROOT / "f1_boxplot.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)


def main() -> None:
    df = load_run_files()
    print("Loaded run results:")
    print(df.head())
    print()

    plot_metric_bars(df)
    plot_runtime_bars(df)
    plot_f1_boxplot(df)


if __name__ == "__main__":
    main()



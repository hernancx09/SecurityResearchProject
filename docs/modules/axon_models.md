# Axon InfraLinter – Models Module

This document describes the model training, evaluation, and visualization modules.

## `axon_infralinter/models/train_lora.py`

**Role**: Fine-tune a base code LLM (CodeLlama) on the Terraform security dataset using LoRA (Low-Rank Adaptation).

### Overview

This module loads the train/validation splits, configures a Hugging Face Trainer with LoRA adapters, and fine-tunes the model. It saves checkpoints periodically and logs training metrics.

### Key Components

#### `TrainConfig` Dataclass

Configuration for training hyperparameters:
- `learning_rate`: 2e-4 (typical for LoRA fine-tuning)
- `num_train_epochs`: 3
- `per_device_train_batch_size`: 1 (reduced for 7B model memory efficiency)
- `gradient_accumulation_steps`: 4 (effective batch size = 1 × 4 = 4)
- `resume_from_checkpoint`: Optional path to resume training

**Student understanding**: 
- **Small batch size** is necessary because 7B parameter models are memory-intensive. Gradient accumulation simulates a larger batch size by accumulating gradients over multiple forward passes before updating weights.
- **LoRA** (Low-Rank Adaptation) trains only a small subset of parameters (typically <1% of the model), making fine-tuning feasible on consumer GPUs. This is a parameter-efficient fine-tuning technique that adds trainable rank-decomposition matrices to attention layers.

#### `tokenize_function(examples: Dict, tokenizer, max_length: int = 1024)`

Tokenizes examples by concatenating `input_text` and `target_text`.

**Format**: `{input_text}\n\n{target_text}`

**Labels**: For causal language modeling, labels are the same as `input_ids` (the model internally shifts to predict next tokens).

**Student understanding**: Causal language models are trained to predict the next token given previous tokens. By concatenating input and target, we teach the model to continue from the prompt with the desired output. The model learns the pattern: "Given this prompt and Terraform code, output SECURE/INSECURE + explanation."

#### Training Process

1. **Load dataset**: Reads `train.jsonl` and `val.jsonl` using Hugging Face `load_dataset`
2. **Load base model**: Loads CodeLlama 7B Instruct from Hugging Face
3. **Configure LoRA**: Sets up LoRA adapters with rank, alpha, and target modules
4. **Prepare for training**: Applies quantization (BitsAndBytesConfig) if needed for memory efficiency
5. **Train**: Runs Hugging Face Trainer with configured hyperparameters
6. **Save checkpoints**: Periodically saves adapter weights and training state

**Output**: 
- `models/axon_lora_codellama/adapter_model.safetensors`: LoRA adapter weights
- `models/axon_lora_codellama/checkpoint-*/`: Training checkpoints with full state

**Student understanding**:
- **Checkpoints** save full Trainer state (optimizer, scheduler, RNG state) enabling exact resumption. This is important for long training runs that may be interrupted.
- **Adapter weights only**: LoRA only saves the small adapter matrices, not the full model. To use the fine-tuned model, you load the base model and then apply the adapters.

---

## `axon_infralinter/models/evaluate.py`

**Role**: Evaluate the fine-tuned LLM and compare against rule-based scanner baselines (Checkov, tfsec).

### Overview

This module loads the test set, runs inference with the fine-tuned model, runs baseline scanners, computes classification metrics (accuracy, precision, recall, F1), and optionally saves results for multi-run benchmarking.

### Key Functions

#### `evaluate_llms(test_records: List[Dict]) -> Dict`

Evaluates the fine-tuned LLM on the test set.

**Process**:
1. Loads base model and LoRA adapters from `LORA_OUTPUT_DIR`
2. For each test example:
   - Tokenizes `input_text`
   - Generates response using `model.generate()`
   - Parses output to extract SECURE/INSECURE prediction
3. Computes metrics comparing predictions to true labels

**Generation parameters**:
- `max_new_tokens`: 50 (enough for SECURE/INSECURE + short explanation)
- `do_sample`: False (deterministic greedy decoding) or True (stochastic sampling)
- `temperature`: Controlled by `AXON_EVAL_MODE` environment variable

**Student understanding**:
- **Deterministic vs. stochastic**: Deterministic mode (greedy decoding) gives consistent results for reproducibility. Stochastic mode (sampling) can reveal model uncertainty but introduces variance.
- **Parsing predictions**: The model generates free-form text, so we parse it to extract the SECURE/INSECURE label. This parsing can fail if the model doesn't follow the format, which is a limitation of the current approach.

#### `evaluate_baselines(test_records: List[Dict]) -> Dict`

Evaluates Checkov and tfsec on the test set.

**Process**:
1. For each test example:
   - Runs Checkov on the Terraform file
   - Runs tfsec on the file's directory
   - Determines secure/insecure label using the same logic as `scanner.py`
2. Computes metrics for each baseline

**Student understanding**: This reuses the same scanning logic as the dataset creation phase, ensuring consistent labeling. The baselines are deterministic (same file always gets same label), which is important for fair comparison.

#### `metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]`

Computes classification metrics using scikit-learn:
- **Accuracy**: Overall correctness
- **Precision**: Of predicted insecure, how many are actually insecure
- **Recall**: Of actually insecure files, how many were detected
- **F1**: Harmonic mean of precision and recall

**Student understanding**: 
- **Precision** measures false positive rate (important for security - we don't want to flag secure files as insecure)
- **Recall** measures false negative rate (critical for security - we don't want to miss real vulnerabilities)
- **F1** balances both, but in security applications, recall is often more important than precision (better to flag a false positive than miss a real issue)

#### Output

Writes evaluation results to `data/eval_runs/eval_axon_lora_codellama_run{N}_{timestamp}.json` with:
- Model identifier
- Run ID and timestamp
- Metrics (accuracy, precision, recall, F1)
- Predictions (for error analysis)
- Runtime statistics

---

## `axon_infralinter/models/benchmark_multi_run.py`

**Role**: Automate multiple evaluation runs to assess variance across random seeds.

### Overview

Runs the evaluation script multiple times (with different seeds or sampling configurations) and aggregates results to understand model stability and variance.

**Student understanding**: Single-run metrics can be misleading due to randomness (in model generation or dataset splits). Multiple runs help distinguish true performance differences from random variation.

---

## `axon_infralinter/models/visualize_benchmarks.py`

**Role**: Generate figures summarizing evaluation metrics.

### Overview

Reads evaluation results from `data/eval_runs/` and creates visualizations:
- **F1 boxplots**: Distribution of F1 scores across runs/models
- **Metrics comparison**: Side-by-side bar charts of accuracy, precision, recall, F1
- **Runtime comparison**: Inference time per file or total evaluation time

**Output**: PNG files in `figures/` directory

**Student understanding**: Visualizations make it easier to:
- Compare models at a glance
- Understand variance across runs
- Identify where each model excels or struggles
- Communicate results in papers/presentations

---

## Design Decisions

### Why LoRA instead of full fine-tuning?

**Memory efficiency**: LoRA trains only ~1% of parameters, making it feasible on consumer GPUs (e.g., 24GB VRAM) that couldn't handle full fine-tuning of a 7B model.

**Speed**: Fewer parameters to update means faster training iterations.

**Modularity**: Can easily swap adapters or combine multiple adapters for different tasks.

**Tradeoff**: May have slightly lower performance than full fine-tuning, but efficiency gains are worth it for research.

### Why causal language modeling?

**Simplicity**: Causal LM is straightforward - just predict next tokens. No need for complex architectures or training objectives.

**Flexibility**: The model can generate explanations naturally, not just binary labels.

**Compatibility**: Works well with instruction-tuned base models like CodeLlama Instruct.

### Why save full Trainer state in checkpoints?

**Reproducibility**: Can resume training exactly where it left off, including optimizer momentum and learning rate schedule.

**Best checkpoint selection**: Can evaluate multiple checkpoints and choose the one with lowest validation loss.

**Debugging**: Full state enables investigating training dynamics (loss curves, learning rate schedules, etc.).


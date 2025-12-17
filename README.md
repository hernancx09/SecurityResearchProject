Axon InfraLinter
=================

Research prototype for **Axon InfraLinter**, an LLM-based tool to detect security misconfigurations in Terraform Infrastructure-as-Code and compare them against rule-based scanners (Checkov, tfsec).

Project layout
--------------

- `requirements.txt` – Python dependencies.
- `axon_infralinter/`
  - `config.py` – central configuration (paths, GitHub, model names).
  - `data/github_scraper.py` – discover and clone GitHub repos with Terraform.
  - `data/terraform_collector.py` – extract `.tf` files and metadata.
  - `scanning/scanner.py` – run Checkov/tfsec and generate unified JSON labels.
  - `dataset/build_dataset.py` – build balanced train/val/test splits (JSONL).
  - `models/train_lora.py` – fine-tune Code Llama with LoRA on the dataset.
  - `models/evaluate.py` – compare baselines and LLM on the held-out test set.
  - `cli/axon_infralinter_cli.py` – simple CLI wrapper for the trained model.

Basic usage
-----------

1. Create and activate a Python virtualenv.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure `axon_infralinter/config.py` (output root, GitHub token, model path).
4. Run the pipeline step by step:

   ```bash
   # 1) Repo discovery + cloning + .tf collection
   python -m axon_infralinter.data.github_scraper
   python -m axon_infralinter.data.terraform_collector

   # 2) Scanning and label generation
   python -m axon_infralinter.scanning.scanner

   # 3) Dataset building
   python -m axon_infralinter.dataset.build_dataset

   # 4) LLM training and evaluation
   python -m axon_infralinter.models.train_lora
   python -m axon_infralinter.models.evaluate

   # 5) CLI inference on new files
   python -m axon_infralinter.cli.axon_infralinter_cli path/to/file.tf
   ```

This is a research prototype, not a production security scanner.


from __future__ import annotations

"""
CLI interface for Axon InfraLinter.

Usage:
    python -m axon_infralinter.cli.axon_infralinter_cli path/to/file.tf
"""

import sys
from pathlib import Path

import click
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from axon_infralinter.config import BASE_MODEL_NAME, LORA_OUTPUT_DIR


def load_model():
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading model from {LORA_OUTPUT_DIR}...")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    
    # Load LoRA adapters if available
    if (LORA_OUTPUT_DIR / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(LORA_OUTPUT_DIR))
        print("  ✓ Loaded fine-tuned LoRA adapters")
    else:
        print("  ⚠ No LoRA adapters found, using base model")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def build_prompt(file_text: str) -> str:
    """Build the prompt for the model."""
    instruction = (
        "You are a security auditor for Terraform Infrastructure-as-Code.\n"
        "Given the following Terraform file, decide if it is SECURE or INSECURE "
        "from a cloud security perspective. Consider issues like overly permissive "
        "IAM policies, missing encryption, public exposure of resources, hard-coded "
        "secrets, and weak network configurations.\n\n"
        "Return exactly two lines:\n"
        "First line: either SECURE or INSECURE.\n"
        "Second line: a brief explanation (one sentence).\n\n"
        "Terraform file:\n"
        "----------------\n"
    )
    return instruction + file_text


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(path: Path) -> None:
    """Analyze a Terraform file for security misconfigurations."""
    try:
        # Load model
        model, tokenizer = load_model()
        model.eval()
        
        # Read file
        file_text = path.read_text(encoding="utf-8", errors="ignore")
        
        # Build prompt
        prompt = build_prompt(file_text)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        print("Analyzing file...")
        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.1,
            )
        
        # Decode and extract result
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the generated part (after the prompt)
        result = generated[len(prompt):].strip()
        
        print()
        print("=" * 70)
        print("AXON INFRALINTER - SECURITY ANALYSIS")
        print("=" * 70)
        print()
        print(result.strip())
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Allow running as a module: python -m axon_infralinter.cli.axon_infralinter_cli
    main(standalone_mode=False)


#!/usr/bin/env python3
"""
Analyze training checkpoints to find the best one and show how to resume.
"""
import json
from pathlib import Path

def analyze_checkpoints(checkpoint_dir: Path):
    """Analyze all checkpoints and find the best one."""
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda x: int(x.parent.name.split("-")[1]) if x.parent.name.split("-")[1].isdigit() else 0
    )
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return None
    
    print("=" * 70)
    print("CHECKPOINT ANALYSIS")
    print("=" * 70)
    print()
    
    best_checkpoint = None
    best_loss = float('inf')
    checkpoint_info = []
    
    for cp_path in checkpoints:
        with open(cp_path) as f:
            state = json.load(f)
        
        checkpoint_name = cp_path.parent.name
        epoch = state.get("epoch", 0)
        global_step = state.get("global_step", 0)
        best_metric = state.get("best_metric")
        best_model_checkpoint = state.get("best_model_checkpoint", "")
        
        # Extract eval_loss from log_history (last eval entry)
        eval_loss = None
        log_history = state.get("log_history", [])
        for entry in reversed(log_history):
            if "eval_loss" in entry:
                eval_loss = entry["eval_loss"]
                break
        
        checkpoint_info.append({
            "name": checkpoint_name,
            "epoch": epoch,
            "eval_loss": eval_loss,
            "global_step": global_step,
            "best_metric": best_metric,
            "path": str(cp_path.parent)
        })
        
        print(f"Checkpoint: {checkpoint_name}")
        print(f"  Epoch: {epoch:.2f}")
        print(f"  Global Step: {global_step}")
        if eval_loss is not None:
            print(f"  Validation Loss: {eval_loss:.6f}")
        if best_metric is not None:
            print(f"  Best Metric: {best_metric:.6f}")
        print(f"  Path: {cp_path.parent}")
        print()
        
        # Track best checkpoint (lowest eval_loss)
        if eval_loss is not None and eval_loss < best_loss:
            best_loss = eval_loss
            best_checkpoint = checkpoint_name
    
    print("=" * 70)
    if best_checkpoint:
        print(f"[BEST] CHECKPOINT: {best_checkpoint}")
        print(f"  Validation Loss: {best_loss:.6f}")
        best_path = checkpoint_dir / best_checkpoint
        print(f"  Path: {best_path}")
        print()
        print("To resume training from this checkpoint:")
        print(f'  Set resume_from_checkpoint="{best_path}" in TrainConfig')
        print()
        print("Or modify train_lora.py:")
        print(f'  resume_from_checkpoint: Optional[str] = "{best_path}"')
    else:
        print("WARNING: Could not determine best checkpoint (no eval_loss found)")
    
    return best_checkpoint, checkpoint_info


if __name__ == "__main__":
    checkpoint_dir = Path("models/axon_lora_codellama")
    analyze_checkpoints(checkpoint_dir)


"""Quick script to train early exit heads for all models."""

import subprocess
import sys
from pathlib import Path

MODELS = [
    "resnet18",
    "resnet50",
    "alexnet",
    "googlenet",
    "mobilenet_v3_large",
    "efficientnet_b2",
]

def train_all_models():
    """Train early exit heads for all models."""
    for model in MODELS:
        print(f"\n{'='*70}")
        print(f"Training early exit heads for {model}")
        print(f"{'='*70}\n")
        
        cmd = [
            sys.executable,
            "train_early_exit_heads.py",
            "--model", model,
            "--batch-size", "32",
            "--epochs", "5",  # Quick training
            "--max-exits", "3",
            "--lr", "0.001",
        ]
        
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        
        if result.returncode != 0:
            print(f"❌ Failed to train {model}")
        else:
            print(f"✓ Successfully trained {model}")

if __name__ == "__main__":
    train_all_models()

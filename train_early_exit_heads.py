"""Train early exit classification heads for DNN_surgery models.

This script trains lightweight classifier heads at intermediate layers of
pretrained models. The heads learn to classify using intermediate features,
enabling confident early exits during inference.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.imagenet_loader import ImageNetMiniLoader
from dnn_surgery import DNNSurgery
from early_exit import EarlyExitHead, _is_residual_block

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EarlyExitTrainer:
    """Trains early exit heads for a given model."""

    def __init__(
        self,
        dnn_surgery: DNNSurgery,
        exit_points: List[int],
        hidden_dim: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.dnn_surgery = dnn_surgery
        self.model = dnn_surgery.model.to(device)
        self.model.eval()  # Freeze backbone
        self.exit_points = exit_points
        self.hidden_dim = hidden_dim
        self.device = device
        self.num_classes = self._infer_num_classes()
        
        # Create exit heads
        self.exit_heads: Dict[int, EarlyExitHead] = {
            idx: EarlyExitHead(self.num_classes, hidden_dim=hidden_dim).to(device)
            for idx in exit_points
        }
        
        logger.info(f"Created {len(self.exit_heads)} exit heads at layers: {exit_points}")

    def _infer_num_classes(self) -> int:
        """Infer number of classes from the model."""
        if hasattr(self.model, "fc") and isinstance(self.model.fc, nn.Linear):
            return self.model.fc.out_features

        classifier = getattr(self.model, "classifier", None)
        if isinstance(classifier, nn.Linear):
            return classifier.out_features
        if isinstance(classifier, nn.Sequential):
            for module in reversed(classifier):
                if isinstance(module, nn.Linear):
                    return module.out_features

        raise ValueError("Unable to infer number of classes from model")

    def _extract_intermediate_features(
        self, input_tensor: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """Extract features at exit points without computing gradients for backbone."""
        features = {}
        
        # Get layers from the model's structure
        layers = list(self.dnn_surgery.splitter.layers)
        
        with torch.no_grad():  # No gradients for backbone
            activation = input_tensor
            for idx, layer in enumerate(layers):
                # Handle flattening if needed
                if self.dnn_surgery.splitter._needs_flattening(layer, activation):
                    activation = torch.flatten(activation, 1)
                activation = layer(activation)
                
                # Save features at exit points
                if idx in self.exit_points:
                    features[idx] = activation.detach().clone()
        
        return features

    def train_heads(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        lr: float = 0.001,
        save_dir: str = "checkpoints/early_exit_heads",
    ) -> Dict[int, str]:
        """Train all exit heads.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of training epochs.
            lr: Learning rate.
            save_dir: Directory to save trained head weights.
            
        Returns:
            Dictionary mapping exit point index to saved checkpoint path.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        for exit_idx, head in self.exit_heads.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training exit head at layer {exit_idx}")
            logger.info(f"{'='*60}")
            
            # Setup optimizer and loss
            optimizer = optim.Adam(head.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            best_val_acc = 0.0
            best_epoch = 0
            
            for epoch in range(epochs):
                # Training phase
                head.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
                for batch_idx, (images, labels) in enumerate(pbar):
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # Extract features at this exit point
                    features = self._extract_intermediate_features(images)
                    exit_features = features[exit_idx]
                    
                    # Forward through exit head
                    optimizer.zero_grad()
                    logits = head(exit_features)
                    loss = criterion(logits, labels)
                    
                    # Backward and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Statistics
                    train_loss += loss.item()
                    _, predicted = logits.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': train_loss / (batch_idx + 1),
                        'acc': 100. * train_correct / train_total
                    })
                
                # Validation phase
                head.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                        images, labels = images.to(self.device), labels.to(self.device)
                        
                        # Extract features at this exit point
                        features = self._extract_intermediate_features(images)
                        exit_features = features[exit_idx]
                        
                        # Forward through exit head
                        logits = head(exit_features)
                        loss = criterion(logits, labels)
                        
                        # Statistics
                        val_loss += loss.item()
                        _, predicted = logits.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()
                
                val_acc = 100. * val_correct / val_total
                logger.info(
                    f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
                    f"Train Acc={100.*train_correct/train_total:.2f}%, "
                    f"Val Loss={val_loss/len(val_loader):.4f}, Val Acc={val_acc:.2f}%"
                )
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
                    checkpoint_path = save_path / f"{self.dnn_surgery.model_name}_exit_{exit_idx}.pt"
                    torch.save(head.state_dict(), checkpoint_path)
                    logger.info(f"✓ Saved best model to {checkpoint_path} (Val Acc: {val_acc:.2f}%)")
            
            logger.info(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
            saved_paths[exit_idx] = str(save_path / f"{self.dnn_surgery.model_name}_exit_{exit_idx}.pt")
        
        return saved_paths


def auto_detect_exit_points(
    dnn_surgery: DNNSurgery,
    max_exits: int = 3,
    split_point: Optional[int] = None,
) -> List[int]:
    """Automatically detect good exit points based on model architecture.
    
    Args:
        dnn_surgery: DNNSurgery instance.
        max_exits: Maximum number of exit points to return.
        split_point: If provided, only return exit points before this split.
        
    Returns:
        List of layer indices suitable for early exits.
    """
    layers = dnn_surgery.splitter.layers
    max_layer = split_point if split_point is not None else len(layers)
    
    # Find residual blocks or major architectural boundaries
    candidates = []
    for idx, layer in enumerate(layers[:max_layer]):
        if _is_residual_block(layer):
            candidates.append(idx)
    
    # If no residual blocks, use evenly spaced layers
    if not candidates:
        step = max(1, max_layer // (max_exits + 1))
        candidates = [i * step for i in range(1, max_exits + 1) if i * step < max_layer]
    
    return candidates[:max_exits]


def main():
    """Train early exit heads for a model."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train early exit heads")
    parser.add_argument("--model", type=str, default="resnet18", help="Model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max-exits", type=int, default=3, help="Maximum number of exit points")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Hidden dimension for exit heads")
    parser.add_argument("--save-dir", type=str, default="checkpoints/early_exit_heads", help="Save directory")
    parser.add_argument("--split-point", type=int, default=None, help="Only train exits before this split point")
    
    args = parser.parse_args()
    
    # Initialize dataset
    logger.info("Loading ImageNet mini dataset...")
    dataset_loader = ImageNetMiniLoader(batch_size=args.batch_size, num_workers=4)
    train_loader, _ = dataset_loader.get_loader(train=True)
    val_loader, _ = dataset_loader.get_loader(train=False)
    
    # Load model
    logger.info(f"Loading pretrained {args.model}...")
    import torchvision.models as models
    model = getattr(models, args.model)(pretrained=True)
    dnn_surgery = DNNSurgery(model, model_name=args.model)
    
    # Auto-detect exit points
    exit_points = auto_detect_exit_points(dnn_surgery, max_exits=args.max_exits, split_point=args.split_point)
    logger.info(f"Selected exit points: {exit_points}")
    
    # Train heads
    trainer = EarlyExitTrainer(
        dnn_surgery=dnn_surgery,
        exit_points=exit_points,
        hidden_dim=args.hidden_dim,
    )
    
    saved_paths = trainer.train_heads(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir,
    )
    
    logger.info("\n" + "="*60)
    logger.info("Training completed! Saved checkpoints:")
    for exit_idx, path in saved_paths.items():
        logger.info(f"  Exit {exit_idx}: {path}")
    logger.info("="*60)
    
    logger.info("\nTo use these heads, configure EarlyExitConfig:")
    logger.info(f"exit_config = EarlyExitConfig(")
    logger.info(f"    enabled=True,")
    logger.info(f"    exit_points={exit_points},")
    logger.info(f"    head_state_dicts={{")
    for exit_idx, path in saved_paths.items():
        logger.info(f"        {exit_idx}: '{path}',")
    logger.info(f"    }}")
    logger.info(f")")


if __name__ == "__main__":
    main()

import os
import logging
from typing import Dict, Tuple, Optional
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ImageNetMiniLoader:
    def __init__(self, batch_size: int = 32, num_workers: int = 4):
        """Initialize ImageNet-Mini dataset loader
        
        Args:
            batch_size: Batch size for data loading
            num_workers: Number of worker processes for data loading
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Set paths
        self.root_dir = Path(__file__).parent.parent / 'data' / 'archive' / 'imagenet-mini'
        if not self.root_dir.exists():
            raise RuntimeError(f"ImageNet-Mini dataset not found at {self.root_dir}")
        
        # ImageNet normalization values
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # Set up transforms and load class mapping
        self._setup_transforms()
        self._setup_class_mapping()
        
        logger.info(f"Initialized ImageNet-Mini loader from {self.root_dir}")
    
    def _setup_transforms(self):
        """Set up data transforms for training and validation"""
        # Training transforms with augmentation
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # Validation transforms - standard ImageNet preprocessing
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        logger.info("Data transforms initialized for training and validation")
    
    def _setup_class_mapping(self):
        """Set up ImageNet class mapping"""
        # Load or create class mapping from directory structure
        train_dir = self.root_dir / 'train'
        
        # Get sorted list of class folders
        class_folders = sorted([d for d in train_dir.iterdir() if d.is_dir()])
        self.classes = [d.name for d in class_folders]
        
        # Create mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.num_classes = len(self.classes)
        
        # Save mapping for reference
        mapping_file = self.root_dir / 'class_mapping.json'
        if not mapping_file.exists():
            mapping = {
                'class_to_idx': self.class_to_idx,
                'idx_to_class': self.idx_to_class,
                'num_classes': self.num_classes
            }
            with open(mapping_file, 'w') as f:
                json.dump(mapping, f, indent=2)
        
        logger.info(f"Found {self.num_classes} classes in ImageNet-Mini dataset")
    
    def get_loader(self, train: bool = False) -> Tuple[DataLoader, Dict]:
        """Get DataLoader for training or validation set
        
        Args:
            train: If True, use training set and augmentation, else use validation set
            
        Returns:
            Tuple of (DataLoader, class mapping dictionary)
        """
        try:
            # Set up dataset path and transforms
            data_path = self.root_dir / ('train' if train else 'val')
            transform = self.train_transform if train else self.val_transform
            
            # Create dataset
            dataset = ImageFolder(
                root=str(data_path),
                transform=transform
            )
            
            # Create dataloader
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=train,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
            logger.info(f"Created {'training' if train else 'validation'} dataloader with {len(dataset)} samples")
            
            # Return loader and class mapping
            return loader, self.idx_to_class
            
        except Exception as e:
            logger.error(f"Error creating dataloader: {str(e)}")
            raise
    
    def verify_data(self, dataloader: DataLoader):
        """Verify loaded dataset and display sample information
        
        Args:
            dataloader: DataLoader to verify
        """
        try:
            # Get a sample batch
            batch = next(iter(dataloader))
            if not isinstance(batch, (tuple, list)):
                logger.error("Unexpected batch format")
                return
            
            images, labels = batch
            
            # Log batch information
            logger.info("\nDataset Verification:")
            logger.info(f"Batch image shape: {images.shape}")
            logger.info(f"Batch label shape: {labels.shape}")
            
            # Verify image properties
            logger.info(f"\nImage Statistics:")
            logger.info(f"Value range: [{images.min():.3f}, {images.max():.3f}]")
            logger.info(f"Mean: {images.mean().item():.3f}")
            logger.info(f"Std: {images.std().item():.3f}")
            
            # Verify labels
            logger.info(f"\nLabel Statistics:")
            logger.info(f"Label range: [{labels.min().item()}, {labels.max().item()}]")
            logger.info(f"Unique labels in batch: {len(torch.unique(labels))}")
            
            # Show some example mappings
            logger.info("\nSample Class Mappings:")
            for i in range(min(5, len(labels))):
                label_idx = labels[i].item()
                class_name = self.idx_to_class[label_idx]
                logger.info(f"Image {i}: Label {label_idx} -> {class_name}")
        
        except Exception as e:
            logger.error(f"Error during dataset verification: {str(e)}")
            raise
    
    @property
    def input_size(self) -> Tuple[int, int, int]:
        """Get input size for the model (C, H, W)"""
        return (3, 224, 224)
    
    @property
    def class_count(self) -> int:
        """Get number of classes in the dataset"""
        return self.num_classes

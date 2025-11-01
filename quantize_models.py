#!/usr/bin/env python3
"""Pre-quantize all supported models and save checkpoints.

This script quantizes full models BEFORE splitting and saves them to disk.
Run this once to create quantized checkpoints that will be loaded at runtime.
"""

import torch
import logging
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    AlexNet_Weights,
    GoogLeNet_Weights,
    EfficientNet_B2_Weights,
    MobileNet_V3_Large_Weights,
)
from dataset.imagenet_loader import ImageNetMiniLoader
from quantization import ModelQuantizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    'resnet18': (models.resnet18, ResNet18_Weights.DEFAULT),
    'resnet50': (models.resnet50, ResNet50_Weights.DEFAULT),
    'alexnet': (models.alexnet, AlexNet_Weights.DEFAULT),
    'googlenet': (models.googlenet, GoogLeNet_Weights.DEFAULT),
    'efficientnet_b2': (models.efficientnet_b2, EfficientNet_B2_Weights.DEFAULT),
    'mobilenet_v3_large': (models.mobilenet_v3_large, MobileNet_V3_Large_Weights.DEFAULT),
}

def main():
    """Quantize all supported models and save checkpoints."""
    logger.info("=" * 80)
    logger.info("Pre-Quantization Script for DNN Surgery")
    logger.info("=" * 80)
    
    logger.info("Loading calibration dataset...")
    dataset_loader = ImageNetMiniLoader(batch_size=4, num_workers=0)
    calibration_dataloader = dataset_loader.get_dataloader()
    logger.info(f"Calibration dataset loaded: {len(dataset_loader)} samples")
    
    quantizer = ModelQuantizer(
        calibration_dataloader=calibration_dataloader,
        num_calibration_batches=10
    )
    
    for model_name, (model_fn, weights) in SUPPORTED_MODELS.items():
        logger.info("\n" + "=" * 80)
        logger.info(f"Processing: {model_name}")
        logger.info("=" * 80)
        
        if quantizer.has_quantized_checkpoint(model_name):
            logger.info(f"✓ Checkpoint already exists: {quantizer.get_checkpoint_path(model_name)}")
            continue
        
        logger.info(f"Loading pretrained {model_name}...")
        model = model_fn(weights=weights).eval()
        
        logger.info(f"Quantizing {model_name}...")
        quantized_model = quantizer.load_or_quantize_model(
            model,
            model_name,
            calibration_dataloader
        )
        
        logger.info(f"✓ Quantized model saved: {quantizer.get_checkpoint_path(model_name)}")
        
        del model
        del quantized_model
        torch.cuda.empty_cache()
    
    logger.info("\n" + "=" * 80)
    logger.info("Pre-Quantization Complete!")
    logger.info("=" * 80)
    logger.info("All models quantized and saved to checkpoints/quantized/")
    logger.info("Future runs will load these pre-quantized models automatically.")

if __name__ == "__main__":
    main()

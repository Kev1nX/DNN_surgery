#!/usr/bin/env python3
"""
DNN Surgery Server Startup Script

This script starts the gRPC server with pre-loaded neural network models.
The server handles the cloud-side computation for distributed DNN inference.

Usage:
    python start_server.py [--port PORT] [--max-workers WORKERS]

Example:
    python start_server.py --port 50051 --max-workers 10
"""

import argparse
import logging
import signal
import sys
import warnings
import torch
import torchvision.models as models
from torchvision.models import (
    AlexNet_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    GoogLeNet_Weights,
    EfficientNet_B2_Weights,
    ConvNeXt_Base_Weights,
)
from server import serve

# Suppress NNPACK warnings
torch.backends.nnpack.enabled = False
warnings.filterwarnings('ignore', message='.*NNPACK.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger(__name__)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info('Received interrupt signal. Shutting down server...')
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Start DNN Surgery gRPC Server')
    parser.add_argument('--port', type=int, default=50051, 
                       help='Port to run the server on (default: 50051)')
    parser.add_argument('--max-workers', type=int, default=10,
                       help='Maximum number of worker threads (default: 10)')
    parser.add_argument('--models', nargs='+', 
                       choices=['resnet18', 'resnet50', 'alexnet', 'googlenet', 'efficientnet_b2', 'convnext_base', 'all'],
                       default=['all'],
                       help='Models to register (default: all)')
    
    args = parser.parse_args()
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        logger.info("Starting DNN Surgery Server...")
        logger.info(f"Device available: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        # Start the server
        server, servicer = serve(port=args.port, max_workers=args.max_workers)
        
        # Register models based on arguments
        models_to_register = []
        
        if 'all' in args.models:
            models_to_register = ['resnet18', 'resnet50', 'alexnet', 'googlenet', 'efficientnet_b2', 'convnext_base']
        else:
            models_to_register = args.models
            
        logger.info(f"Registering models: {models_to_register}")
        
        for model_name in models_to_register:
            try:
                if model_name == 'resnet18':
                    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
                    model.eval()
                    servicer.register_model('resnet18', model)
                    logger.info("ResNet18 (pretrained) registered successfully")
                    
                elif model_name == 'resnet50':
                    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
                    model.eval()
                    servicer.register_model('resnet50', model)
                    logger.info("ResNet50 (pretrained) registered successfully")
                    
                elif model_name == 'alexnet':
                    model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
                    model.eval()
                    servicer.register_model('alexnet', model)
                    logger.info("AlexNet (pretrained) registered successfully")
                    
                elif model_name == 'googlenet':
                    model = models.googlenet(weights=GoogLeNet_Weights.DEFAULT)
                    model.eval()
                    servicer.register_model('googlenet', model)
                    logger.info("GoogLeNet (pretrained) registered successfully")
                    
                elif model_name == 'efficientnet_b2':
                    model = models.efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
                    model.eval()
                    servicer.register_model('efficientnet_b2', model)
                    logger.info("EfficientNet-B2 (pretrained) registered successfully")
                    
                elif model_name == 'convnext_base':
                    model = models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
                    model.eval()
                    servicer.register_model('convnext_base', model)
                    logger.info("ConvNeXt-Base (pretrained) registered successfully")
                    
            except Exception as e:
                logger.error(f"Failed to register {model_name}: {str(e)}")
                continue
        
        logger.info("="*60)
        logger.info(f"DNN Surgery Server started successfully!")
        logger.info(f"Listening on port: {args.port}")
        logger.info(f"Max workers: {args.max_workers}")
        logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        logger.info(f"Available models: {models_to_register}")
        logger.info("="*60)
        logger.info("Server is ready to accept client connections...")
        logger.info("Press Ctrl+C to stop the server")
        
        # Wait for termination
        server.wait_for_termination()
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

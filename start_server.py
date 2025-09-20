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
import torch
from server import serve
from networks.resnet18 import resnet18
from networks.alexnet import alexnet  
from networks.cnn import CNN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dnn_server.log')
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
                       choices=['resnet18', 'alexnet', 'cnn', 'all'],
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
            models_to_register = ['resnet18', 'alexnet', 'cnn']
        else:
            models_to_register = args.models
            
        logger.info(f"Registering models: {models_to_register}")
        
        for model_name in models_to_register:
            try:
                if model_name == 'resnet18':
                    model = resnet18
                    servicer.register_model('resnet18', model)
                    logger.info("ResNet18 registered successfully")
                    
                elif model_name == 'alexnet':
                    model = alexnet
                    servicer.register_model('alexnet', model)
                    logger.info("AlexNet registered successfully")
                    
                elif model_name == 'cnn':
                    model = CNN(num_classes=1000)
                    servicer.register_model('cnn', model)
                    logger.info("CNN registered successfully")
                    
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

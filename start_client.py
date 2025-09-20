#!/usr/bin/env python3

import argparse
import logging
import sys
import time
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import random

from dnn_inference_client import DNNInferenceClient, run_distributed_inference_with_profiling
from dnn_surgery import DNNSurgery
from networks.resnet18 import resnet18
from networks.alexnet import alexnet
from networks.cnn import CNN
from dataset.imagenet_loader import ImageNetMiniLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Global dataset loader instance
_dataset_loader = None
_dataset_iterator = None
_class_mapping = None

def initialize_dataset_loader(batch_size: int = 1) -> None:
    """Initialize the dataset loader for ImageNet mini dataset"""
    global _dataset_loader, _dataset_iterator, _class_mapping
    
    try:
        logger.info("Initializing ImageNet mini dataset loader...")
        _dataset_loader = ImageNetMiniLoader(batch_size=batch_size, num_workers=0)
        
        # Get validation dataset (for inference testing)
        val_loader, class_mapping = _dataset_loader.get_loader(train=False)
        _class_mapping = class_mapping
        _dataset_iterator = iter(val_loader)
        
        logger.info(f"Dataset initialized with {_dataset_loader.class_count} classes")
        
        # Verify the dataset
        _dataset_loader.verify_data(val_loader)
        
    except Exception as e:
        logger.error(f"Failed to initialize dataset: {str(e)}")
        raise RuntimeError(f"ImageNet mini dataset is required but failed to load: {str(e)}")

def get_input_tensor(model_name: str, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Get images from ImageNet mini dataset
    
    Args:
        model_name: Model name (determines input size compatibility)
        batch_size: Number of images to get
        
    Returns:
        Tuple of (input_tensor, labels, class_names)
        
    Raises:
        RuntimeError: If CNN model is used (not compatible with ImageNet) or dataset fails to load
    """
    global _dataset_iterator, _class_mapping
    
    # CNN model is not compatible with ImageNet images (different input size)
    if model_name == 'cnn':
        raise RuntimeError("CNN model requires CIFAR-10 sized inputs (32x32) but ImageNet mini has 224x224 images. "
                          "Please use 'resnet18' or 'alexnet' models for ImageNet mini dataset.")
    
    # Initialize dataset loader if not already done
    if _dataset_loader is None:
        initialize_dataset_loader(batch_size)
    
    # Ensure dataset is properly loaded
    if _dataset_loader is None or _dataset_iterator is None or _class_mapping is None:
        raise RuntimeError("ImageNet mini dataset failed to initialize properly")
    
    try:
        # Get next batch from iterator
        try:
            images, labels = next(_dataset_iterator)
        except StopIteration:
            # Reset iterator if we've exhausted the dataset
            logger.info("Resetting dataset iterator...")
            val_loader, _ = _dataset_loader.get_loader(train=False)
            _dataset_iterator = iter(val_loader)
            images, labels = next(_dataset_iterator)
        
        # Get the requested batch size
        if images.size(0) != batch_size:
            if images.size(0) < batch_size:
                # If we don't have enough images, repeat the batch
                repeat_factor = (batch_size + images.size(0) - 1) // images.size(0)
                images = images.repeat(repeat_factor, 1, 1, 1)[:batch_size]
                labels = labels.repeat(repeat_factor)[:batch_size]
            else:
                # If we have more images than needed, take a subset
                images = images[:batch_size]
                labels = labels[:batch_size]
        
        # Get class names for the labels
        class_names = [_class_mapping[label.item()] for label in labels]
        
        logger.info(f"Loaded images - Shape: {images.shape}, Labels: {labels.tolist()}")
        logger.info(f"Class names: {class_names}")
        
        return images, labels, class_names
        
    except Exception as e:
        logger.error(f"Error loading images: {str(e)}")
        raise RuntimeError(f"Failed to load ImageNet images: {str(e)}")

def create_sample_input(model_name: str, batch_size: int = 1) -> torch.Tensor:
    """Create input tensor using ImageNet mini dataset images
    
    This function only uses ImageNet images - no random tensors
    """
    # Get images - will raise exception if fails
    input_tensor, labels, class_names = get_input_tensor(model_name, batch_size)
    
    # For backward compatibility, just return the tensor
    return input_tensor

def get_model(model_name: str):
    """Get model instance by name"""
    if model_name == 'resnet18':
        return resnet18
    elif model_name == 'alexnet':
        return alexnet
    elif model_name == 'cnn':
        return CNN(num_classes=1000)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def test_connection(server_address: str) -> bool:
    """Test if server is reachable"""
    try:
        logger.info(f"Testing connection to {server_address}...")
        client = DNNInferenceClient(server_address, None)
        
        # Try a simple inference to test connection
        test_input = torch.randn(1, 3, 224, 224)
        result, timings = client.process_tensor(test_input, 'resnet18')
        
        logger.info("✓ Connection test successful!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Connection test failed: {str(e)}")
        return False

def run_single_inference(server_address: str, model_name: str, split_point: int = None, 
                        batch_size: int = 1) -> Tuple[torch.Tensor, Dict]:
    """Run a single inference with NeuroSurgeon optimization
    
    Args:
        server_address: Server address
        model_name: Model name (must be 'resnet18' or 'alexnet' for ImageNet)
        split_point: Optional manual split point (if None, uses NeuroSurgeon optimization)
        batch_size: Batch size
        
    Returns:
        Tuple of (result tensor, timing dictionary)
        
    Raises:
        RuntimeError: If CNN model is used or dataset fails to load
    """
    
    # Create input - only uses ImageNet images
    input_tensor, true_labels, class_names = get_input_tensor(model_name, batch_size)
    
    # Get model and set up DNN surgery
    model = get_model(model_name)
    dnn_surgery = DNNSurgery(model, model_name)
    
    if split_point is None:
        logger.info(f"Running NeuroSurgeon optimization for model={model_name}, batch_size={batch_size}")
    else:
        logger.info(f"Running inference: model={model_name}, split_point={split_point}, batch_size={batch_size}")
    
    # Log information about the input
    logger.info(f"Using ImageNet images with true labels: {true_labels.tolist()}")
    logger.info(f"True classes: {class_names}")
    
    # Run distributed inference
    result, timings = run_distributed_inference_with_profiling(
        model_name, input_tensor, dnn_surgery, split_point, server_address
    )
    
    # Add true label information to timing results for analysis
    timings['true_labels'] = true_labels.tolist()
    timings['class_names'] = class_names
    
    return result, timings

def run_batch_processing(server_address: str, model_name: str, split_point: int = None, 
                        batch_size: int = 1, num_batches: int = 1) -> List[Dict]:
    """Run multiple batches and collect timing statistics"""
    
    # Ensure model supports ImageNet
    if model_name == 'cnn':
        raise RuntimeError("CNN model requires CIFAR-10 sized inputs (32x32) but only ImageNet mini is supported. "
                          "Please use 'resnet18' or 'alexnet' models.")
    
    # Initialize dataset
    initialize_dataset_loader(batch_size)
    
    model = get_model(model_name)
    dnn_surgery = DNNSurgery(model, model_name)
    
    if split_point is not None:
        dnn_surgery.splitter.set_split_point(split_point)
        edge_model = dnn_surgery.splitter.get_edge_model()
        client = DNNInferenceClient(server_address, edge_model)
        logger.info(f"Using manual split point: {split_point}")
    else:
        # Use NeuroSurgeon for first batch, then reuse the optimal split
        logger.info("Using NeuroSurgeon optimization for batch processing")
    
    all_timings = []
    optimal_split_found = None
    
    logger.info(f"Starting batch processing: {num_batches} batches of size {batch_size}")
    
    for batch_idx in range(num_batches):
        # Get input
        input_tensor, true_labels, class_names = get_input_tensor(model_name, batch_size)
        
        start_time = time.time()
        
        if split_point is not None:
            # Use manual split point
            result, timings = client.process_tensor(input_tensor, model_name)
        else:
            # Use NeuroSurgeon (either find optimal or reuse)
            if optimal_split_found is not None:
                # Reuse previously found optimal split
                result, timings = run_distributed_inference_with_profiling(
                    model_name, input_tensor, dnn_surgery, optimal_split_found, server_address
                )
            else:
                # Find optimal split for first batch
                result, timings = run_distributed_inference_with_profiling(
                    model_name, input_tensor, dnn_surgery, None, server_address
                )
                optimal_split_found = timings.get('split_point', 2)
                logger.info(f"Found optimal split point: {optimal_split_found} (will reuse for remaining batches)")
        
        total_time = time.time() - start_time
        
        timings['total_wall_time'] = total_time * 1000  # Convert to ms
        timings['batch_index'] = batch_idx
        
        # Add true label information
        timings['true_labels'] = true_labels.tolist()
        timings['class_names'] = class_names
            
        all_timings.append(timings)
        
        logger.info(f"Batch {batch_idx + 1}/{num_batches} completed in {total_time*1000:.1f}ms")
        
        # Show prediction vs true labels
        if result.dim() == 2:
            probs = torch.softmax(result, dim=1)
            confidence, predicted_class = probs.max(1)
            
            for i in range(len(predicted_class)):
                pred_class = predicted_class[i].item()
                conf = confidence[i].item()
                
                true_class = true_labels[i].item()
                true_name = class_names[i]
                correct = "✓" if pred_class == true_class else "✗"
                logger.info(f"  Image {i}: Predicted={pred_class}, True={true_class} ({true_name}), "
                          f"Confidence={conf:.3f} {correct}")
    
    return all_timings

def test_split_points(server_address: str, model_name: str, split_points: List[int], 
                     num_tests: int = 3) -> Dict[int, Dict]:
    """Test different split points and return performance comparison"""
    
    # Ensure model supports ImageNet
    if model_name == 'cnn':
        raise RuntimeError("CNN model requires CIFAR-10 sized inputs (32x32) but only ImageNet mini is supported. "
                          "Please use 'resnet18' or 'alexnet' models.")
    
    # Initialize dataset
    initialize_dataset_loader(1)  # Use batch size 1 for testing
    
    model = get_model(model_name)
    dnn_surgery = DNNSurgery(model, model_name)
    
    # Profile the model first with a sample input
    input_tensor, _, _ = get_input_tensor(model_name, 1)
    logger.info("Profiling model layers...")
    profiles = dnn_surgery.profile_model(input_tensor)
    
    results = {}
    
    logger.info(f"Testing split points: {split_points}")
    
    for split_point in split_points:
        logger.info(f"\n--- Testing split point {split_point} ---")
        
        timings_list = []
        
        for test_idx in range(num_tests):
            try:
                result, timings = run_single_inference(server_address, model_name, split_point)
                timings_list.append(timings)
                
                total_time = timings.get('edge_time', 0) + timings.get('cloud_time', 0) + timings.get('transfer_time', 0)
                
                # Show prediction results with true labels
                true_labels = timings['true_labels']
                class_names = timings['class_names']
                
                if result.dim() == 2:
                    probs = torch.softmax(result, dim=1)
                    _, predicted = probs.max(1)
                    pred_class = predicted[0].item()
                    true_class = true_labels[0]
                    true_name = class_names[0]
                    correct = "✓" if pred_class == true_class else "✗"
                    
                    logger.info(f"  Test {test_idx + 1}: Total={total_time:.1f}ms "
                              f"(Edge={timings.get('edge_time', 0):.1f}ms, "
                              f"Cloud={timings.get('cloud_time', 0):.1f}ms, "
                              f"Transfer={timings.get('transfer_time', 0):.1f}ms) "
                              f"Pred={pred_class}, True={true_class} ({true_name}) {correct}")
                else:
                    logger.info(f"  Test {test_idx + 1}: Total={total_time:.1f}ms "
                              f"(Edge={timings.get('edge_time', 0):.1f}ms, "
                              f"Cloud={timings.get('cloud_time', 0):.1f}ms, "
                              f"Transfer={timings.get('transfer_time', 0):.1f}ms)")
                
            except Exception as e:
                logger.error(f"  Test {test_idx + 1} failed: {str(e)}")
                continue
        
        if timings_list:
            # Calculate averages
            avg_edge = np.mean([t.get('edge_time', 0) for t in timings_list])
            avg_cloud = np.mean([t.get('cloud_time', 0) for t in timings_list])
            avg_transfer = np.mean([t.get('transfer_time', 0) for t in timings_list])
            avg_total = avg_edge + avg_cloud + avg_transfer
            
            results[split_point] = {
                'avg_edge_time': avg_edge,
                'avg_cloud_time': avg_cloud,
                'avg_transfer_time': avg_transfer,
                'avg_total_time': avg_total,
                'num_tests': len(timings_list)
            }
            
            logger.info(f"  Average: Total={avg_total:.1f}ms "
                       f"(Edge={avg_edge:.1f}ms, Cloud={avg_cloud:.1f}ms, Transfer={avg_transfer:.1f}ms)")
    
    return results

def print_performance_summary(results: Dict[int, Dict]):
    """Print a summary of performance results"""
    if not results:
        logger.warning("No results to summarize")
        return
    
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Split':<6} {'Total(ms)':<12} {'Edge(ms)':<11} {'Cloud(ms)':<12} {'Transfer(ms)':<14} {'Tests':<6}")
    print("-" * 80)
    
    # Sort by total time
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_total_time'])
    
    for split_point, metrics in sorted_results:
        print(f"{split_point:<6} "
              f"{metrics['avg_total_time']:<12.1f} "
              f"{metrics['avg_edge_time']:<11.1f} "
              f"{metrics['avg_cloud_time']:<12.1f} "
              f"{metrics['avg_transfer_time']:<14.1f} "
              f"{metrics['num_tests']:<6}")
    
    # Find optimal split point
    optimal_split = sorted_results[0][0]
    optimal_time = sorted_results[0][1]['avg_total_time']
    
    print("-" * 80)
    print(f"OPTIMAL SPLIT POINT: {optimal_split} (Average time: {optimal_time:.1f}ms)")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='DNN Surgery Client')
    parser.add_argument('--server-address', required=True,
                       help='Server address in format HOST:PORT (e.g., 192.168.1.100:50051)')
    parser.add_argument('--model', choices=['resnet18', 'alexnet'], default='resnet18',
                       help='Model to use for inference (default: resnet18) - only ImageNet-compatible models')
    parser.add_argument('--split-point', type=int, default=None,
                       help='Split point for model partitioning (default: None - use NeuroSurgeon optimization)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference (default: 1)')
    parser.add_argument('--num-batches', type=int, default=1,
                       help='Number of batches to process (default: 1)')
    parser.add_argument('--test-splits', type=str,
                       help='Comma-separated split points to test (e.g., "0,1,2,3,4") - overrides NeuroSurgeon')
    parser.add_argument('--num-tests', type=int, default=3,
                       help='Number of tests per split point (default: 3)')
    parser.add_argument('--test-connection', action='store_true',
                       help='Test connection to server and exit')
    parser.add_argument('--use-neurosurgeon', action='store_true', default=True,
                       help='Use NeuroSurgeon optimization (default: True)')
    parser.add_argument('--no-neurosurgeon', action='store_true',
                       help='Disable NeuroSurgeon optimization')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting DNN Surgery Client")
        logger.info(f"Server address: {args.server_address}")
        logger.info(f"Model: {args.model}")
        
        # Initialize dataset for supported models (ImageNet only)
        if args.model in ['resnet18', 'alexnet']:
            logger.info("Initializing ImageNet mini dataset for image inference...")
            initialize_dataset_loader(args.batch_size)
        elif args.model == 'cnn':
            logger.error("CNN model is not supported with ImageNet mini dataset (incompatible input sizes)")
            logger.error("CNN requires 32x32 CIFAR-10 inputs, but ImageNet mini has 224x224 images")
            logger.error("Please use 'resnet18' or 'alexnet' models instead")
            sys.exit(1)
        
        # Test connection if requested
        if args.test_connection:
            if test_connection(args.server_address):
                logger.info("Connection test passed!")
                sys.exit(0)
            else:
                logger.error("Connection test failed!")
                sys.exit(1)
        
        # Test multiple split points if specified
        if args.test_splits:
            split_points = [int(x.strip()) for x in args.test_splits.split(',')]
            logger.info(f"Testing split points: {split_points}")
            
            results = test_split_points(args.server_address, args.model, split_points, args.num_tests)
            print_performance_summary(results)
            
        # Run batch processing
        elif args.num_batches > 1:
            logger.info(f"Running batch processing: {args.num_batches} batches")
            
            split_point = args.split_point if args.no_neurosurgeon else None
            
            timings_list = run_batch_processing(
                args.server_address, args.model, split_point, 
                args.batch_size, args.num_batches
            )
            
            # Print batch summary
            avg_total = np.mean([t.get('total_wall_time', 0) for t in timings_list])
            logger.info(f"Batch processing completed. Average time per batch: {avg_total:.1f}ms")
            
            # Calculate accuracy since we always have true labels
            if timings_list:
                total_correct = 0
                total_samples = 0
                
                for timing in timings_list:
                    true_labels = timing['true_labels']
                    predicted_classes = timing.get('predicted_classes', [])
                    
                    for true_label, pred_class in zip(true_labels, predicted_classes):
                        if true_label == pred_class:
                            total_correct += 1
                        total_samples += 1
                
                if total_samples > 0:
                    accuracy = (total_correct / total_samples) * 100
                    logger.info(f"Overall accuracy: {accuracy:.2f}% ({total_correct}/{total_samples})")
            
        # Run single inference
        else:
            if args.no_neurosurgeon:
                split_point = args.split_point if args.split_point is not None else 2
                logger.info(f"Running single inference with manual split point {split_point}")
            else:
                split_point = None
                logger.info(f"Running single inference with NeuroSurgeon optimization")
            
            result, timings = run_single_inference(
                args.server_address, args.model, split_point, args.batch_size
            )
            
            total_time = timings.get('edge_time', 0) + timings.get('cloud_time', 0) + timings.get('transfer_time', 0)
            actual_split = timings.get('split_point', 'unknown')
            
            print(f"\n Inference completed successfully!")
            print(f"Results:")
            print(f"   Output shape: {result.shape}")
            print(f"   Split point used: {actual_split}")
            print(f"   Total time: {total_time:.1f}ms")
            print(f"   Edge time: {timings.get('edge_time', 0):.1f}ms")
            print(f"   Cloud time: {timings.get('cloud_time', 0):.1f}ms")
            print(f"   Transfer time: {timings.get('transfer_time', 0):.1f}ms")
            
            # Show prediction for classification with true labels (always available)
            if result.dim() == 2:
                probs = torch.softmax(result, dim=1)
                confidence, predicted_class = probs.max(1)
                
                for i in range(len(predicted_class)):
                    pred_class = predicted_class[i].item()
                    conf = confidence[i].item()
                    
                    true_labels = timings['true_labels']
                    class_names = timings['class_names']
                    
                    if i < len(true_labels) and i < len(class_names):
                        true_class = true_labels[i]
                        true_name = class_names[i]
                        correct = "✓ CORRECT" if pred_class == true_class else "✗ INCORRECT"
                        
                        print(f"   Image {i}: Predicted class: {pred_class}")
                        print(f"   Image {i}: True class: {true_class} ({true_name})")
                        print(f"   Image {i}: Confidence: {conf:.3f}")
                        print(f"   Image {i}: {correct}")
                    else:
                        print(f"   Image {i}: Predicted class: {pred_class}")
                        print(f"   Image {i}: Confidence: {conf:.3f}")
        
        logger.info("Client execution completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Client interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Client error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

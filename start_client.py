#!/usr/bin/env python3
"""
DNN Surgery Client Script

This script runs the client-side of distributed DNN inference.
It can run edge computations locally and send intermediate results to the server.

Usage:
    python start_client.py --server-address HOST:PORT [options]

Examples:
    # Basic inference
    python start_client.py --server-address 192.168.1.100:50051 --model resnet18 --split-point 2

    # Batch processing
    python start_client.py --server-address 192.168.1.100:50051 --model resnet18 --batch-size 5 --split-point 3

    # Performance testing
    python start_client.py --server-address 192.168.1.100:50051 --model resnet18 --test-splits 0,1,2,3,4
"""

import argparse
import logging
import sys
import time
import torch
import numpy as np
from typing import List, Dict, Tuple

from dnn_inference_client import DNNInferenceClient, run_distributed_inference_with_profiling
from dnn_surgery import DNNSurgery
from networks.resnet18 import resnet18
from networks.alexnet import alexnet
from networks.cnn import CNN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def create_sample_input(model_name: str, batch_size: int = 1) -> torch.Tensor:
    """Create sample input tensor based on model requirements"""
    if model_name in ['resnet18', 'alexnet']:
        # ImageNet input size
        return torch.randn(batch_size, 3, 224, 224)
    elif model_name == 'cnn':
        # CIFAR-10 input size  
        return torch.randn(batch_size, 3, 32, 32)
    else:
        # Default ImageNet size
        return torch.randn(batch_size, 3, 224, 224)

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
        model_name: Model name
        split_point: Optional manual split point (if None, uses NeuroSurgeon optimization)
        batch_size: Batch size
        
    Returns:
        Tuple of (result tensor, timing dictionary)
    """
    
    # Create input
    input_tensor = create_sample_input(model_name, batch_size)
    
    # Get model and set up DNN surgery
    model = get_model(model_name)
    dnn_surgery = DNNSurgery(model, model_name)
    
    if split_point is None:
        logger.info(f"Running NeuroSurgeon optimization for model={model_name}, batch_size={batch_size}")
    else:
        logger.info(f"Running inference: model={model_name}, split_point={split_point}, batch_size={batch_size}")
    
    # Run distributed inference
    result, timings = run_distributed_inference_with_profiling(
        model_name, input_tensor, dnn_surgery, split_point, server_address
    )
    
    return result, timings

def run_batch_processing(server_address: str, model_name: str, split_point: int = None, 
                        batch_size: int = 1, num_batches: int = 1) -> List[Dict]:
    """Run multiple batches and collect timing statistics"""
    
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
        input_tensor = create_sample_input(model_name, batch_size)
        
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
        all_timings.append(timings)
        
        logger.info(f"Batch {batch_idx + 1}/{num_batches} completed in {total_time*1000:.1f}ms")
        
        # For classification models, show prediction
        if result.dim() == 2:
            probs = torch.softmax(result, dim=1)
            confidence, predicted_class = probs.max(1)
            logger.info(f"  Prediction: class={predicted_class[0].item()}, confidence={confidence[0].item():.3f}")
    
    return all_timings

def test_split_points(server_address: str, model_name: str, split_points: List[int], 
                     num_tests: int = 3) -> Dict[int, Dict]:
    """Test different split points and return performance comparison"""
    
    model = get_model(model_name)
    dnn_surgery = DNNSurgery(model, model_name)
    
    # Profile the model first
    input_tensor = create_sample_input(model_name)
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
    print(f"🏆 OPTIMAL SPLIT POINT: {optimal_split} (Average time: {optimal_time:.1f}ms)")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='DNN Surgery Client')
    parser.add_argument('--server-address', required=True,
                       help='Server address in format HOST:PORT (e.g., 192.168.1.100:50051)')
    parser.add_argument('--model', choices=['resnet18', 'alexnet', 'cnn'], default='resnet18',
                       help='Model to use for inference (default: resnet18)')
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
        logger.info("🚀 Starting DNN Surgery Client")
        logger.info(f"📡 Server address: {args.server_address}")
        logger.info(f"🤖 Model: {args.model}")
        
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
            logger.info(f"🔬 Testing split points: {split_points}")
            
            results = test_split_points(args.server_address, args.model, split_points, args.num_tests)
            print_performance_summary(results)
            
        # Run batch processing
        elif args.num_batches > 1:
            logger.info(f"📦 Running batch processing: {args.num_batches} batches")
            
            split_point = args.split_point if args.no_neurosurgeon else None
            
            timings_list = run_batch_processing(
                args.server_address, args.model, split_point, 
                args.batch_size, args.num_batches
            )
            
            # Print batch summary
            avg_total = np.mean([t.get('total_wall_time', 0) for t in timings_list])
            logger.info(f"📊 Batch processing completed. Average time per batch: {avg_total:.1f}ms")
            
        # Run single inference
        else:
            if args.no_neurosurgeon:
                split_point = args.split_point if args.split_point is not None else 2
                logger.info(f"🎯 Running single inference with manual split point {split_point}")
            else:
                split_point = None
                logger.info(f"🧠 Running single inference with NeuroSurgeon optimization")
            
            result, timings = run_single_inference(
                args.server_address, args.model, split_point, args.batch_size
            )
            
            total_time = timings.get('edge_time', 0) + timings.get('cloud_time', 0) + timings.get('transfer_time', 0)
            actual_split = timings.get('split_point', 'unknown')
            
            print(f"\n✅ Inference completed successfully!")
            print(f"📊 Results:")
            print(f"   Output shape: {result.shape}")
            print(f"   Split point used: {actual_split}")
            print(f"   Total time: {total_time:.1f}ms")
            print(f"   Edge time: {timings.get('edge_time', 0):.1f}ms")
            print(f"   Cloud time: {timings.get('cloud_time', 0):.1f}ms")
            print(f"   Transfer time: {timings.get('transfer_time', 0):.1f}ms")
            
            # Show prediction for classification
            if result.dim() == 2:
                probs = torch.softmax(result, dim=1)
                confidence, predicted_class = probs.max(1)
                print(f"   Predicted class: {predicted_class[0].item()}")
                print(f"   Confidence: {confidence[0].item():.3f}")
        
        logger.info("🎉 Client execution completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⚠️  Client interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Client error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

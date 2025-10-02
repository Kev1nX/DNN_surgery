#!/usr/bin/env python3

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torchvision.models as models
from torchvision.models import (
    AlexNet_Weights,
    ResNet18_Weights,
    EfficientNet_V2_L_Weights,
    ConvNeXt_Base_Weights,
    ViT_B_16_Weights,
)

from dataset.imagenet_loader import ImageNetMiniLoader
from dnn_inference_client import DNNInferenceClient, resolve_plot_paths, run_distributed_inference
from dnn_surgery import DNNSurgery
from visualization import (
    plot_actual_inference_breakdown,
    plot_actual_split_comparison,
    plot_multi_model_comparison,
)

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
        RuntimeError: If unsupported model is used or dataset fails to load
    """
    global _dataset_iterator, _class_mapping
    
    # Only support pretrained ImageNet models
    supported_models = ['resnet18', 'alexnet', 'efficientnet_v2_l', 'convnext_base', 'vit_b_16']
    if model_name not in supported_models:
        raise RuntimeError(
            f"Model '{model_name}' is not supported. Supported models: {', '.join(supported_models)}"
        )
    
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
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.eval()
        return model
    if model_name == 'alexnet':
        model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        model.eval()
        return model
    if model_name == 'efficientnet_v2_l':
        model = models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
        model.eval()
        return model
    if model_name == 'convnext_base':
        model = models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        model.eval()
        return model
    if model_name == 'vit_b_16':
        model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.eval()
        return model
    raise ValueError(
        f"Unknown model: {model_name}. Supported models: resnet18, alexnet, efficientnet_v2_l, convnext_base, vit_b_16"
    )

def test_connection(server_address: str) -> bool:
    """Test if server is reachable"""
    try:
        logger.info(f"Testing connection to {server_address}...")
        client = DNNInferenceClient(server_address, None)
        
        # Try a simple inference to test connection
        test_input = torch.randn(1, 3, 224, 224)
        client.process_tensor(test_input, 'resnet18')
        
        logger.info("✓ Connection test successful!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Connection test failed: {str(e)}")
        return False

def run_single_inference(
    server_address: str,
    model_name,
    dnn_surgery: DNNSurgery,
    split_point: int = None,
    batch_size: int = 1,
    *,
    auto_plot: bool = True,
    plot_show: bool = True,
    plot_path: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict]:
    """Run a single inference with NeuroSurgeon optimization
    
    Args:
        server_address: Server address
        model_name: Model name
        split_point: Optional manual split point (if None, uses NeuroSurgeon optimization)
        batch_size: Batch size
        
    Returns:
        Tuple of (result tensor, timing dictionary)
        
    Raises:
        RuntimeError: If CNN model is used or dataset fails to load
    """
    
    # Create input - only uses ImageNet images
    input_tensor, true_labels, class_names = get_input_tensor(model_name, batch_size)
    
    
    if split_point is None:
        logger.info(f"Running NeuroSurgeon optimization for model={model_name}, batch_size={batch_size}")
    else:
        logger.info(f"Running inference: model={model_name}, split_point={split_point}, batch_size={batch_size}")
    
    # Log information about the input
    logger.info(f"Using ImageNet images with true labels: {true_labels.tolist()}")
    logger.info(f"True classes: {class_names}")
    
    # Run distributed inference
    result, timings = run_distributed_inference(
        model_name,
        input_tensor,
        dnn_surgery,
        split_point,
        server_address,
        auto_plot=auto_plot,
        plot_show=plot_show,
        plot_path=plot_path,
    )
    
    # Add true label information to timing results for analysis
    timings['true_labels'] = true_labels.tolist()
    timings['class_names'] = class_names

    predicted_plot_path = timings.get('predicted_split_plot_path')
    actual_plot_path = timings.get('actual_split_plot_path')
    if predicted_plot_path:
        logger.info("Predicted split timing chart saved to %s", predicted_plot_path)
    if actual_plot_path:
        logger.info("Measured inference chart saved to %s", actual_plot_path)
    
    return result, timings

def run_batch_processing(
    server_address: str,
    model_name: str,
    split_point: int = None,
    batch_size: int = 1,
    num_batches: int = 1,
    *,
    auto_plot: bool = True,
    plot_show: bool = True,
    plot_path: Optional[str] = None,
) -> List[Dict]:
    """Run multiple batches and collect timing statistics"""
    
    # Ensure model is supported
    supported_models = ['resnet18', 'alexnet', 'efficientnet_v2_l', 'convnext_base', 'vit_b_16']
    if model_name not in supported_models:
        raise RuntimeError(
            f"Model '{model_name}' is not supported. Supported models: {', '.join(supported_models)}"
        )
    
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
    
    should_plot = auto_plot

    for batch_idx in range(num_batches):
        # Get input
        input_tensor, true_labels, class_names = get_input_tensor(model_name, batch_size)
        
        start_time = time.time()
        
        if split_point is not None:
            # Use manual split point
            result, timings = client.process_tensor(input_tensor, model_name)
            if auto_plot and should_plot:
                _, manual_actual_path = resolve_plot_paths(model_name, split_point, plot_path)
                try:
                    manual_actual_path.parent.mkdir(parents=True, exist_ok=True)
                    plot_actual_inference_breakdown(
                        {
                            "edge_time": timings.get("edge_time", 0.0),
                            "transfer_time": timings.get("transfer_time", 0.0),
                            "cloud_time": timings.get("cloud_time", 0.0),
                            "total_batch_processing_time": timings.get("total_batch_processing_time"),
                        },
                        show=plot_show,
                        save_path=str(manual_actual_path),
                        title=f"Measured Inference Breakdown ({model_name}, split {split_point})",
                    )
                    manual_actual_path_resolved = str(manual_actual_path.resolve())
                    timings['actual_split_plot_path'] = manual_actual_path_resolved
                    logger.info("Measured inference chart saved to %s", manual_actual_path_resolved)
                except ImportError as plt_error:
                    logger.warning(
                        "Auto-plot requested but matplotlib is unavailable: %s",
                        plt_error,
                    )
                except Exception as plot_error:
                    logger.error("Failed to render measured inference chart: %s", plot_error)
                should_plot = False
        else:
            # Use NeuroSurgeon (either find optimal or reuse)
            if optimal_split_found is not None:
                # Reuse previously found optimal split
                result, timings = run_distributed_inference(
                    model_name,
                    input_tensor,
                    dnn_surgery,
                    optimal_split_found,
                    server_address,
                    auto_plot=False,
                    plot_show=plot_show,
                    plot_path=plot_path,
                )
            else:
                # Find optimal split for first batch
                result, timings = run_distributed_inference(
                    model_name,
                    input_tensor,
                    dnn_surgery,
                    None,
                    server_address,
                    auto_plot=should_plot,
                    plot_show=plot_show,
                    plot_path=plot_path,
                )
                optimal_split_found = timings.get('split_point', 2)
                should_plot = False
                logger.info(f"Found optimal split point: {optimal_split_found} (will reuse for remaining batches)")
        predicted_plot_path = timings.get('predicted_split_plot_path')
        actual_plot_path = timings.get('actual_split_plot_path')
        if predicted_plot_path:
            logger.info("Predicted split timing chart saved to %s", predicted_plot_path)
        if actual_plot_path:
            logger.info("Measured inference chart saved to %s", actual_plot_path)
        
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


def test_split_points(
    server_address: str,
    model_name: str,
    split_points: List[int],
    num_tests: int = 3,
    *,
    auto_plot: bool = True,
    plot_show: bool = True,
    plot_path: Optional[str] = None,
) -> Dict[int, Dict]:
    """Test different split points and return performance comparison."""

    supported_models = ['resnet18', 'alexnet', 'efficientnet_v2_l', 'convnext_base', 'vit_b_16']
    if model_name not in supported_models:
        raise RuntimeError(
            f"Model '{model_name}' is not supported. Supported models: {', '.join(supported_models)}"
        )

    initialize_dataset_loader(1)  # Use batch size 1 for testing

    model = get_model(model_name)
    dnn_surgery = DNNSurgery(model, model_name)

    input_tensor, _, _ = get_input_tensor(model_name, 1)
    logger.info("Profiling model layers...")
    dnn_surgery.profile_model(input_tensor)

    results: Dict[int, Dict] = {}

    def _init_component_buckets() -> Dict[str, List[float]]:
        return {
            'edge': [],
            'cloud': [],
            'transfer': [],
            'total': [],
        }

    split_actual_components: Dict[int, Dict[str, List[float]]] = defaultdict(_init_component_buckets)

    logger.info(f"Testing split points: {split_points}")

    for split_point in split_points:
        logger.info(f"\n--- Testing split point {split_point} ---")

        timings_list: List[Dict] = []

        for test_idx in range(num_tests):
            try:
                result, timings = run_single_inference(
                    server_address,
                    model_name,
                    dnn_surgery,
                    split_point,
                    auto_plot=auto_plot and test_idx == 0,
                    plot_show=plot_show,
                    plot_path=plot_path,
                )
                timings_list.append(timings)

                edge_time = float(timings.get('edge_time', 0.0))
                cloud_time = float(timings.get('cloud_time', 0.0))
                transfer_time = float(timings.get('transfer_time', 0.0))
                total_time = edge_time + cloud_time + transfer_time

                buckets = split_actual_components[split_point]
                buckets['edge'].append(edge_time)
                buckets['cloud'].append(cloud_time)
                buckets['transfer'].append(transfer_time)
                buckets['total'].append(total_time)

                true_labels = timings['true_labels']
                class_names = timings['class_names']

                if result.dim() == 2:
                    probs = torch.softmax(result, dim=1)
                    _, predicted = probs.max(1)
                    pred_class = predicted[0].item()
                    true_class = true_labels[0]
                    true_name = class_names[0]
                    correct = "✓" if pred_class == true_class else "✗"

                    logger.info(
                        f"  Test {test_idx + 1}: Total={total_time:.1f}ms "
                        f"(Edge={timings.get('edge_time', 0):.1f}ms, "
                        f"Cloud={timings.get('cloud_time', 0):.1f}ms, "
                        f"Transfer={timings.get('transfer_time', 0):.1f}ms) "
                        f"Pred={pred_class}, True={true_class} ({true_name}) {correct}"
                    )
                else:
                    logger.info(
                        f"  Test {test_idx + 1}: Total={total_time:.1f}ms "
                        f"(Edge={timings.get('edge_time', 0):.1f}ms, "
                        f"Cloud={timings.get('cloud_time', 0):.1f}ms, "
                        f"Transfer={timings.get('transfer_time', 0):.1f}ms)"
                    )

            except Exception as e:
                logger.error(f"  Test {test_idx + 1} failed: {str(e)}")
                continue

        if timings_list:
            avg_edge = np.mean([t.get('edge_time', 0) for t in timings_list])
            avg_cloud = np.mean([t.get('cloud_time', 0) for t in timings_list])
            avg_transfer = np.mean([t.get('transfer_time', 0) for t in timings_list])
            avg_total = avg_edge + avg_cloud + avg_transfer

            results[split_point] = {
                'avg_edge_time': avg_edge,
                'avg_cloud_time': avg_cloud,
                'avg_transfer_time': avg_transfer,
                'avg_total_time': avg_total,
                'num_tests': len(timings_list),
            }

            logger.info(
                f"  Average: Total={avg_total:.1f}ms "
                f"(Edge={avg_edge:.1f}ms, Cloud={avg_cloud:.1f}ms, Transfer={avg_transfer:.1f}ms)"
            )

    if auto_plot and split_actual_components:
        _, comparison_seed_path = resolve_plot_paths(model_name, None, plot_path)
        comparison_path = comparison_seed_path.with_name(
            comparison_seed_path.stem.replace("_actual", "") + "_comparison" + comparison_seed_path.suffix
        )
        try:
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            plot_actual_split_comparison(
                split_actual_components,
                show=plot_show,
                save_path=str(comparison_path),
                title=f"Measured Inference Timings ({model_name})",
            )
            logger.info("Split comparison line chart saved to %s", comparison_path.resolve())
        except ImportError as plt_error:
            logger.warning(
                "Auto-plot requested but matplotlib is unavailable: %s",
                plt_error,
            )
        except Exception as plot_error:
            logger.error("Failed to render split comparison chart: %s", plot_error)

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

def test_all_models_all_splits(
    server_address: str,
    num_tests: int = 3,
    *,
    auto_plot: bool = True,
    plot_show: bool = True,
    plot_base_path: Optional[str] = None,
) -> Dict[str, Dict[int, Dict]]:
    """Test all supported models across all their split points.
    
    Args:
        server_address: Server address
        num_tests: Number of test runs per split point
        auto_plot: Whether to generate plots
        plot_show: Whether to show plots interactively
        plot_base_path: Base directory for saving plots
        
    Returns:
        Dictionary mapping model names to their test results
    """
    supported_models = ['resnet18', 'alexnet', 'efficientnet_v2_l', 'convnext_base', 'vit_b_16']
    
    # Initialize dataset once
    initialize_dataset_loader(1)
    
    all_model_results = {}
    all_model_timings = {}  # For multi-model comparison
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE MODEL TESTING")
    logger.info("Testing all models across all split points")
    logger.info("="*80)
    
    for model_name in supported_models:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing model: {model_name}")
        logger.info('='*80)
        
        try:
            # Get model and create DNN Surgery instance
            model = get_model(model_name)
            dnn_surgery = DNNSurgery(model, model_name)
            
            # Profile to determine number of layers
            input_tensor, _, _ = get_input_tensor(model_name, 1)
            dnn_surgery.profile_model(input_tensor)
            num_layers = len(dnn_surgery.splitter.layers)
            
            # Test all split points for this model
            split_points = list(range(num_layers + 1))
            logger.info(f"Model has {num_layers} layers, testing {len(split_points)} split points: {split_points}")
            
            # Determine save path for this model's individual plot
            model_plot_path = None
            if auto_plot and plot_base_path:
                plot_dir = Path(plot_base_path)
                plot_dir.mkdir(parents=True, exist_ok=True)
                model_plot_path = str(plot_dir / f"{model_name}_split_comparison.png")
            
            # Run the split point tests
            results = test_split_points(
                server_address,
                model_name,
                split_points,
                num_tests=num_tests,
                auto_plot=auto_plot,
                plot_show=plot_show,
                plot_path=model_plot_path,
            )
            
            all_model_results[model_name] = results
            
            # Extract timing data for multi-model comparison
            model_timings = {}
            for split_point, metrics in results.items():
                model_timings[split_point] = {
                    'total_time': metrics['avg_total_time'],
                    'client_time': metrics.get('avg_edge_time', 0.0),
                    'server_time': metrics.get('avg_cloud_time', 0.0),
                    'transfer_time': metrics.get('avg_transfer_time', 0.0),
                }
            all_model_timings[model_name] = model_timings
            
            logger.info(f"✓ Completed testing {model_name}")
            
        except Exception as e:
            logger.error(f"✗ Failed to test {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate multi-model comparison plot
    if auto_plot and all_model_timings:
        logger.info("\n" + "="*80)
        logger.info("Generating multi-model comparison plots...")
        logger.info("="*80)
        
        try:
            plot_dir = Path(plot_base_path) if plot_base_path else Path("plots")
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate comparison for each metric
            metrics_to_plot = [
                ('total_time', 'Total Latency'),
                ('client_time', 'Client Execution Time'),
                ('server_time', 'Server Execution Time'),
                ('transfer_time', 'Data Transfer Time'),
            ]
            
            for metric, metric_name in metrics_to_plot:
                save_path = str(plot_dir / f"multi_model_comparison_{metric}.png")
                plot_multi_model_comparison(
                    all_model_timings,
                    show=plot_show,
                    save_path=save_path,
                    title=f"Multi-Model Comparison: {metric_name}",
                    metric=metric,
                )
                logger.info(f"✓ Saved {metric_name} comparison to {save_path}")
            
            logger.info("✓ All comparison plots generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate multi-model comparison: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE TEST SUMMARY")
    logger.info("="*80)
    
    for model_name, results in all_model_results.items():
        if results:
            best_split = min(results.items(), key=lambda x: x[1]['avg_total_time'])
            logger.info(f"{model_name:20s}: Optimal split={best_split[0]:2d}, Time={best_split[1]['avg_total_time']:.1f}ms")
        else:
            logger.info(f"{model_name:20s}: No results")
    
    logger.info("="*80)
    
    return all_model_results

def main():
    parser = argparse.ArgumentParser(description='DNN Surgery Client')
    parser.add_argument('--server-address', required=True,
                       help='Server address in format HOST:PORT (e.g., 192.168.1.100:50051)')
    parser.add_argument(
        '--model',
        choices=['resnet18', 'alexnet', 'efficientnet_v2_l', 'convnext_base', 'vit_b_16'],
        default='resnet18',
        help='Model to use for inference (default: resnet18). Supported: resnet18, alexnet, efficientnet_v2_l, convnext_base, vit_b_16',
    )
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
    parser.add_argument('--test-all-models', action='store_true',
                       help='Test all supported models across all split points and generate comparison plots')
    parser.add_argument('--use-neurosurgeon', action='store_true', default=True,
                       help='Use NeuroSurgeon optimization (default: True)')
    parser.add_argument('--no-neurosurgeon', action='store_true',
                       help='Disable NeuroSurgeon optimization')
    parser.add_argument(
        '--auto-plot',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Automatically generate split timing plots (default: enabled)'
    )
    parser.add_argument(
        '--show-plot',
        dest='plot_show',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Display the split timing plot window (default: enabled)'
    )
    parser.add_argument(
        '--plot-save',
        metavar='PATH',
        default=None,
        help='Optional path to save the generated split timing plot'
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting DNN Surgery Client")
        logger.info(f"Server address: {args.server_address}")
        logger.info(f"Model: {args.model}")
        logger.info(
            "Auto-plot: %s (show=%s, save=%s)",
            "enabled" if args.auto_plot else "disabled",
            args.plot_show,
            args.plot_save or "<none>",
        )
        
        # Initialize dataset for supported models (ImageNet only)
        supported_models = ['resnet18', 'alexnet', 'efficientnet_v2_l', 'convnext_base', 'vit_b_16']
        if args.model not in supported_models:
            logger.error(f"Model '{args.model}' is not supported")
            logger.error(f"Supported models: {', '.join(supported_models)}")
            sys.exit(1)

        logger.info("Initializing ImageNet mini dataset for image inference...")
        initialize_dataset_loader(args.batch_size)
        
        # Test connection if requested
        if args.test_connection:
            if test_connection(args.server_address):
                logger.info("Connection test passed!")
                sys.exit(0)
            else:
                logger.error("Connection test failed!")
                sys.exit(1)
        
        # Test all models across all split points
        if args.test_all_models:
            logger.info("Testing all models across all split points...")
            test_all_models_all_splits(
                args.server_address,
                num_tests=args.num_tests,
                auto_plot=args.auto_plot,
                plot_show=args.plot_show,
                plot_base_path=args.plot_save or "plots/comprehensive",
            )
            logger.info("Comprehensive testing complete!")
            sys.exit(0)
        
        # Test multiple split points if specified
        if args.test_splits:
            split_points = [int(x.strip()) for x in args.test_splits.split(',')]
            logger.info(f"Testing split points: {split_points}")
            
            results = test_split_points(
                args.server_address,
                args.model,
                split_points,
                args.num_tests,
                auto_plot=args.auto_plot,
                plot_show=args.plot_show,
                plot_path=args.plot_save,
            )
            print_performance_summary(results)
            
        # Run batch processing
        elif args.num_batches > 1:
            logger.info(f"Running batch processing: {args.num_batches} batches")
            
            split_point = args.split_point if args.no_neurosurgeon else None
            
            timings_list = run_batch_processing(
                args.server_address,
                args.model,
                split_point,
                args.batch_size,
                args.num_batches,
                auto_plot=args.auto_plot,
                plot_show=args.plot_show,
                plot_path=args.plot_save,
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
            dnn_surgery = DNNSurgery(get_model(args.model), args.model)
            
            result, timings = run_single_inference(
                args.server_address,
                args.model,
                dnn_surgery,
                split_point,
                args.batch_size,
                auto_plot=args.auto_plot,
                plot_show=args.plot_show,
                plot_path=args.plot_save,
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

            predicted_plot_path = timings.get('predicted_split_plot_path')
            actual_plot_path = timings.get('actual_split_plot_path')
            if predicted_plot_path:
                print(f"   Predicted split chart saved to: {predicted_plot_path}")
            if actual_plot_path:
                print(f"   Measured inference chart saved to: {actual_plot_path}")
            
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

#!/usr/bin/env python3

import argparse
import logging
import sys
import time
import traceback
import warnings
import os

# Suppress warnings BEFORE importing torch
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'  # Suppress OpenMP warnings
os.environ['MKL_NUM_THREADS'] = '1'  # Suppress MKL warnings

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torchvision.models as models
from torchvision.models import (
    AlexNet_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    GoogLeNet_Weights,
    EfficientNet_B2_Weights,
    MobileNet_V3_Large_Weights,
)

# Disable PyTorch warnings after import
torch.set_warn_always(False)

from dataset.imagenet_loader import ImageNetMiniLoader
from dnn_inference_client import DNNInferenceClient, resolve_plot_paths, run_distributed_inference
from dnn_surgery import DNNSurgery
from early_exit import EarlyExitInferenceClient, EarlyExitConfig, run_distributed_inference_with_early_exit
from visualization import (
    plot_actual_inference_breakdown,
    plot_actual_split_comparison,
    plot_multi_model_comparison,
    plot_model_comparison_bar,
    plot_throughput_from_timing,
    plot_split_throughput_comparison,
    plot_multi_model_throughput_comparison,
    plot_model_throughput_comparison_bar,
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
    
    validate_model(model_name)
    
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

# Supported models configuration
SUPPORTED_MODELS = ['resnet18', 'resnet50', 'alexnet', 'googlenet', 'efficientnet_b2', 'mobilenet_v3_large']

MODEL_REGISTRY = {
    'resnet18': (models.resnet18, ResNet18_Weights.DEFAULT),
    'resnet50': (models.resnet50, ResNet50_Weights.DEFAULT),
    'alexnet': (models.alexnet, AlexNet_Weights.DEFAULT),
    'googlenet': (models.googlenet, GoogLeNet_Weights.DEFAULT),
    'efficientnet_b2': (models.efficientnet_b2, EfficientNet_B2_Weights.DEFAULT),
    'mobilenet_v3_large': (models.mobilenet_v3_large, MobileNet_V3_Large_Weights.DEFAULT),
}

def get_model(model_name: str):
    """Get model instance by name"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Supported models: {', '.join(SUPPORTED_MODELS)}"
        )
    model_fn, weights = MODEL_REGISTRY[model_name]
    return model_fn(weights=weights).eval()

def calculate_timing_averages(timings_list: List[Dict]) -> Dict[str, float]:
    """Calculate average timings from a list of timing dictionaries"""
    if not timings_list:
        return {}
    
    avg_edge = np.mean([t.get('edge_time', 0) for t in timings_list])
    avg_cloud = np.mean([t.get('cloud_time', 0) for t in timings_list])
    avg_transfer = np.mean([t.get('transfer_time', 0) for t in timings_list])
    avg_total = avg_edge + avg_cloud + avg_transfer
    
    return {
        'avg_edge_time': avg_edge,
        'avg_cloud_time': avg_cloud,
        'avg_transfer_time': avg_transfer,
        'avg_total_time': avg_total,
        'num_tests': len(timings_list),
    }

def validate_model(model_name: str) -> None:
    """Validate that a model is supported"""
    if model_name not in SUPPORTED_MODELS:
        raise RuntimeError(
            f"Model '{model_name}' is not supported. Supported models: {', '.join(SUPPORTED_MODELS)}"
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
    use_early_exit: bool = False,
) -> List[Dict]:
    """Run multiple batches and collect timing statistics"""
    validate_model(model_name)
    
    # Initialize dataset
    initialize_dataset_loader(batch_size)
    
    model = get_model(model_name)
    dnn_surgery = DNNSurgery(model, model_name)
    
    # Setup early exit if requested
    exit_config = None
    if use_early_exit:
        logger.info("Early exit enabled for batch processing")
        exit_config = EarlyExitConfig(
            enabled=True,
            confidence_threshold=0.7,  # Exit at first opportunity
            max_exits=3,
        )
    
    if split_point is not None:
        dnn_surgery.splitter.set_split_point(split_point)
        edge_model = dnn_surgery.splitter.get_edge_model()
        client = DNNInferenceClient(server_address, edge_model)
        logger.info(f"Using manual split point: {split_point}")
    else:
        # Use NeuroSurgeon for first batch, then reuse the optimal split
        config_msg = "NeuroSurgeon optimization" + (" with early exit" if use_early_exit else "")
        logger.info(f"Using {config_msg} for batch processing")
    
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
                    timing_data = {
                        "edge_time": timings.get("edge_time", 0.0),
                        "transfer_time": timings.get("transfer_time", 0.0),
                        "cloud_time": timings.get("cloud_time", 0.0),
                        "total_batch_processing_time": timings.get("total_batch_processing_time"),
                    }
                    plot_actual_inference_breakdown(
                        timing_data,
                        show=plot_show,
                        save_path=str(manual_actual_path),
                        title=f"Measured Inference Breakdown ({model_name}, split {split_point})",
                    )
                    manual_actual_path_resolved = str(manual_actual_path.resolve())
                    timings['actual_split_plot_path'] = manual_actual_path_resolved
                    logger.info("Measured inference chart saved to %s", manual_actual_path_resolved)
                    
                    # Generate companion throughput plot
                    manual_throughput_path = manual_actual_path.with_name(
                        manual_actual_path.stem.replace("_actual", "_throughput") + manual_actual_path.suffix
                    )
                    plot_throughput_from_timing(
                        timing_data,
                        show=plot_show,
                        save_path=str(manual_throughput_path),
                        title=f"Inference Throughput ({model_name}, split {split_point})",
                    )
                    logger.info("Throughput chart saved to %s", manual_throughput_path.resolve())
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
            if use_early_exit:
                # Use early exit version
                if optimal_split_found is not None:
                    # Reuse previously found optimal split
                    result, timings = run_distributed_inference_with_early_exit(
                        model_name,
                        input_tensor,
                        dnn_surgery,
                        exit_config=exit_config,
                        split_point=optimal_split_found,
                        server_address=server_address,
                    )
                else:
                    # Find optimal split for first batch (with early exit profiling)
                    result, timings = run_distributed_inference_with_early_exit(
                        model_name,
                        input_tensor,
                        dnn_surgery,
                        exit_config=exit_config,
                        split_point=None,
                        server_address=server_address,
                    )
                    optimal_split_found = timings.get('split_point', 2)
                    should_plot = False
                    logger.info(f"Found optimal split point with early exit: {optimal_split_found} (will reuse for remaining batches)")
            else:
                # Standard inference without early exit
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
    validate_model(model_name)
    initialize_dataset_loader(1)

    model = get_model(model_name)
    dnn_surgery = DNNSurgery(model, model_name)

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
                    auto_plot=False,  # Don't generate bar charts for individual split points
                    plot_show=False,
                    plot_path=None,
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
            results[split_point] = calculate_timing_averages(timings_list)
            avg = results[split_point]
            logger.info(
                f"  Average: Total={avg['avg_total_time']:.1f}ms "
                f"(Edge={avg['avg_edge_time']:.1f}ms, Cloud={avg['avg_cloud_time']:.1f}ms, Transfer={avg['avg_transfer_time']:.1f}ms)"
            )

    if auto_plot and split_actual_components:
        _, comparison_seed_path = resolve_plot_paths(model_name, None, plot_path)
        comparison_path = comparison_seed_path.with_name(
            comparison_seed_path.stem.replace("_actual", "") + "_comparison" + comparison_seed_path.suffix
        )
        throughput_comparison_path = comparison_seed_path.with_name(
            comparison_seed_path.stem.replace("_actual", "") + "_throughput_comparison" + comparison_seed_path.suffix
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
            
            # Generate throughput comparison plot
            plot_split_throughput_comparison(
                split_actual_components,
                show=plot_show,
                save_path=str(throughput_comparison_path),
                title=f"Inference Throughput ({model_name})",
            )
            logger.info("Throughput comparison chart saved to %s", throughput_comparison_path.resolve())
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

def test_all_models_single_split(
    server_address: str,
    split_point: int,
    num_tests: int = 3,
    *,
    auto_plot: bool = True,
    plot_show: bool = True,
    plot_path: Optional[str] = None,
) -> Dict[str, Dict]:
    """Test all supported models at a single split point and compare their performance.
    
    Args:
        server_address: Server address
        split_point: Split point to test across all models
        num_tests: Number of test runs per model
        auto_plot: Whether to generate plots
        plot_show: Whether to show plots interactively
        plot_path: Path for saving the plot
        
    Returns:
        Dictionary mapping model names to their timing results
    """
    initialize_dataset_loader(1)
    
    all_model_timings = {}
    
    logger.info("="*80)
    logger.info(f"TESTING ALL MODELS AT SPLIT POINT {split_point}")
    logger.info("="*80)
    
    for model_name in SUPPORTED_MODELS:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing model: {model_name} at split point {split_point}")
        logger.info('='*80)
        
        try:
            # Get model and create DNN Surgery instance
            model = get_model(model_name)
            dnn_surgery = DNNSurgery(model, model_name)
            
            # Get number of layers
            num_layers = len(dnn_surgery.splitter.layers)
            
            # Validate split point
            if split_point < 0 or split_point > num_layers:
                logger.error(f"Invalid split point {split_point} for {model_name} (has {num_layers} layers)")
                continue
            
            # Run multiple tests
            edge_times = []
            transfer_times = []
            cloud_times = []
            total_times = []
            
            for test_idx in range(num_tests):
                logger.info(f"  Test run {test_idx + 1}/{num_tests}")
                
                try:
                    result, timings = run_single_inference(
                        server_address,
                        model_name,
                        dnn_surgery,
                        split_point,
                        batch_size=1,
                        auto_plot=False,  # Don't generate individual plots
                        plot_show=False,
                        plot_path=None,
                    )
                    
                    edge_time = timings.get("edge_time", 0.0)
                    transfer_time = timings.get("transfer_time", 0.0)
                    cloud_time = timings.get("cloud_time", 0.0)
                    total_time = timings.get("total_batch_processing_time", edge_time + transfer_time + cloud_time)
                    
                    edge_times.append(edge_time)
                    transfer_times.append(transfer_time)
                    cloud_times.append(cloud_time)
                    total_times.append(total_time)
                    
                    logger.info(
                        f"    Edge: {edge_time:.2f}ms | Transfer: {transfer_time:.2f}ms | "
                        f"Cloud: {cloud_time:.2f}ms | Total: {total_time:.2f}ms"
                    )
                    
                except Exception as e:
                    logger.error(f"  Test failed: {str(e)}")
                    continue
            
            # Calculate averages
            if total_times:
                avg_edge = sum(edge_times) / len(edge_times)
                avg_transfer = sum(transfer_times) / len(transfer_times)
                avg_cloud = sum(cloud_times) / len(cloud_times)
                avg_total = sum(total_times) / len(total_times)
                
                all_model_timings[model_name] = {
                    'edge_time': avg_edge,
                    'transfer_time': avg_transfer,
                    'cloud_time': avg_cloud,
                    'total_time': avg_total,
                    'num_tests': len(total_times),
                }
                
                logger.info(f"\n{model_name} Average times:")
                logger.info(
                    f"  Edge: {avg_edge:.2f}ms | Transfer: {avg_transfer:.2f}ms | "
                    f"Cloud: {avg_cloud:.2f}ms | Total: {avg_total:.2f}ms"
                )
            
        except Exception as e:
            logger.error(f"Failed to test {model_name}: {str(e)}")
            continue
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info(f"SUMMARY - All Models at Split Point {split_point}")
    logger.info("="*80)
    logger.info(f"{'Model':<20} {'Total(ms)':<12} {'Edge(ms)':<11} {'Transfer(ms)':<14} {'Cloud(ms)':<12}")
    logger.info("-" * 80)
    
    # Sort by total time
    sorted_models = sorted(all_model_timings.items(), key=lambda x: x[1]['total_time'])
    for model_name, timings in sorted_models:
        logger.info(
            f"{model_name:<20} {timings['total_time']:<12.1f} {timings['edge_time']:<11.1f} "
            f"{timings['transfer_time']:<14.1f} {timings['cloud_time']:<12.1f}"
        )
    
    logger.info("="*80)
    
    # Generate comparison plots
    if auto_plot and all_model_timings:
        _save_comparison_plots(
            Path(plot_path or "plots"), all_model_timings, 
            f"Model Comparison at Split {split_point}", f"all_models_split_{split_point}", plot_show
        )
    
    return all_model_timings

def _calculate_batch_accuracy(result: torch.Tensor, true_labels: torch.Tensor, batch_idx: int) -> Tuple[int, int]:
    """Calculate accuracy for a batch of predictions"""
    if result.dim() != 2:
        return 0, 0
    probs = torch.softmax(result, dim=1)
    _, predicted = probs.max(1)
    correct = (predicted == true_labels).sum().item()
    total = len(true_labels)
    logger.debug(f"Batch {batch_idx}: {correct}/{total} correct")
    return correct, total

def _save_comparison_plots(plot_dir: Path, model_timings: Dict, title_prefix: str, filename_prefix: str, 
                          show: bool = True, timestamp: str = None) -> None:
    """Helper to save bar chart and throughput plots"""
    timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        bar_path = plot_dir / f"{filename_prefix}_{timestamp}.png"
        throughput_path = plot_dir / f"{filename_prefix}_throughput_{timestamp}.png"
        
        plot_model_comparison_bar(model_timings, show=show, save_path=str(bar_path), title=title_prefix)
        plot_model_throughput_comparison_bar(model_timings, show=show, save_path=str(throughput_path), 
                                            title=f"{title_prefix} - Throughput")
        logger.info(f"✓ Charts saved: {bar_path.name}, {throughput_path.name}")
    except Exception as e:
        logger.error(f"Plot generation failed: {e}")

def test_all_models_neurosurgeon(
    server_address: str,
    num_batches: int = 3,
    *,
    auto_plot: bool = True,
    plot_show: bool = True,
    plot_path: Optional[str] = None,
    enable_quantization: bool = False,
    use_early_split: bool = False,
    batch_size: int = 1,
) -> Dict[str, Dict]:
    """Test all supported models using NeuroSurgeon-determined optimal split points.
    
    This function runs each model with NeuroSurgeon optimization to find the optimal
    split point, then performs multiple batch inferences for stable measurements.
    
    Args:
        server_address: Server address
        num_batches: Number of batches to run per model for stable measurements (default: 3)
        auto_plot: Whether to generate plots
        plot_show: Whether to show plots interactively
        plot_path: Path for saving the plot
        enable_quantization: Whether to enable INT8 quantization for models
        use_early_split: Whether to enable early exit with intermediate classifiers
        batch_size: Batch size for each inference run (default: 1)
        
    Returns:
        Dictionary mapping model names to their optimal split timing results
    """
    initialize_dataset_loader(batch_size)
    
    all_model_timings = {}
    
    # Build configuration description
    config_parts = []
    if enable_quantization:
        config_parts.append("Quantization")
    if use_early_split:
        config_parts.append("Early Exit")
    
    config_desc = " + ".join(config_parts) if config_parts else "Standard"
    
    logger.info("="*80)
    logger.info(f"TESTING ALL MODELS WITH NEUROSURGEON OPTIMIZATION ({config_desc})")
    logger.info("="*80)
    if enable_quantization:
        logger.info("Quantization: ENABLED (INT8 dynamic quantization)")
    if use_early_split:
        logger.info("Early Exit: ENABLED (intermediate classifiers with confidence threshold)")
    logger.info(f"Running {num_batches} batch(es) per model (batch_size={batch_size}) for stable measurements")
    logger.info("="*80)
    
    for model_name in SUPPORTED_MODELS:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing model: {model_name} with NeuroSurgeon")
        logger.info('='*80)
        
        try:
            # Get model and create DNN Surgery instance
            model = get_model(model_name)
            dnn_surgery = DNNSurgery(model, model_name, enable_quantization=enable_quantization)
            
            # Configure early exit if requested
            exit_config = None
            if use_early_split:
                logger.info(f"Using early exit configuration with intermediate classifiers")
                exit_config = EarlyExitConfig(
                    enabled=True,
                    confidence_threshold=0.7,  # Lower threshold - 0% confidence to exit (always exit at first opportunity)
                    max_exits=3,  # Limit to 3 early exit points
                )
            else:
                logger.info(f"Running NeuroSurgeon optimization for {model_name}...")
            
            # Initialize timing lists and accuracy tracking
            edge_times = []
            transfer_times = []
            cloud_times = []
            total_times = []
            correct_predictions = 0
            total_samples = 0
            early_exit_counts = []
            early_exit_rates = []
            optimal_split = None
            
            # Run all batches for stable measurements
            logger.info(f"Running {num_batches} batch(es) for stable measurements...")
            for batch_idx in range(num_batches):
                # Get fresh input for each batch
                input_tensor, true_labels, class_names = get_input_tensor(model_name, batch_size)
                
                # Determine split point: None for first batch (NeuroSurgeon optimization), reuse optimal_split for subsequent batches
                current_split = None if batch_idx == 0 else optimal_split
                
                if use_early_split:
                    result, timings = run_distributed_inference_with_early_exit(
                        model_name,
                        input_tensor,
                        dnn_surgery,
                        exit_config=exit_config,
                        split_point=current_split,  # None for first batch, then reuse optimal split
                        server_address=server_address,
                    )
                    early_exit_counts.append(timings.get('early_exit_count', 0))
                    early_exit_rates.append(timings.get('early_exit_rate', 0.0))
                else:
                    result, timings = run_distributed_inference(
                        model_name,
                        input_tensor,
                        dnn_surgery,
                        split_point=current_split,
                        server_address=server_address,
                        auto_plot=False,
                        plot_show=False,
                        plot_path=None,
                    )
                
                # Store optimal split from first batch
                if batch_idx == 0:
                    optimal_split = timings.get('split_point')
                    logger.info(f"{'Early exit configuration' if use_early_split else 'NeuroSurgeon'} determined optimal split point: {optimal_split}")
                
                # Collect timing metrics
                edge_times.append(timings.get('edge_time', 0))
                transfer_times.append(timings.get('transfer_time', 0))
                cloud_times.append(timings.get('cloud_time', 0))
                total_times.append(timings.get('edge_time', 0) + timings.get('transfer_time', 0) + timings.get('cloud_time', 0))
                
                # Calculate accuracy for this batch
                batch_correct, batch_total = _calculate_batch_accuracy(result, true_labels, batch_idx + 1)
                correct_predictions += batch_correct
                total_samples += batch_total
                
                logger.info(f"  Batch {batch_idx + 1}/{num_batches}: Total={total_times[-1]:.1f}ms (Edge={edge_times[-1]:.1f}ms, Transfer={transfer_times[-1]:.1f}ms, Cloud={cloud_times[-1]:.1f}ms)")
        
            # Calculate averages and statistics
            avg_edge = np.mean(edge_times) if edge_times else 0.0
            avg_transfer = np.mean(transfer_times) if transfer_times else 0.0
            avg_cloud = np.mean(cloud_times) if cloud_times else 0.0
            avg_total = np.mean(total_times) if total_times else 0.0
            std_total = np.std(total_times) if len(total_times) > 1 else 0.0
            accuracy = (correct_predictions / total_samples * 100) if total_samples > 0 else 0.0
            
            all_model_timings[model_name] = {
                'optimal_split': optimal_split,
                'edge_time': avg_edge,
                'transfer_time': avg_transfer,
                'cloud_time': avg_cloud,
                'total_time': avg_total,
                'std_total_time': std_total,
                'num_batches': len(total_times) if total_times else 0,
                'throughput': 1000.0 / avg_total if avg_total > 0 else 0.0,  # inferences per second
                'accuracy': accuracy,
                'correct_predictions': correct_predictions,
                'total_samples': total_samples,
            }
            
            # Add early exit statistics if enabled
            if use_early_split and early_exit_counts:
                all_model_timings[model_name]['avg_early_exit_count'] = np.mean(early_exit_counts)
                all_model_timings[model_name]['avg_early_exit_rate'] = np.mean(early_exit_rates)
            
            # Simplified logging
            logger.info(f"✓ {model_name}: split={optimal_split}, time={avg_total:.1f}±{std_total:.1f}ms, "
                       f"accuracy={accuracy:.1f}%, breakdown: E={avg_edge:.1f} T={avg_transfer:.1f} C={avg_cloud:.1f}ms")
            if use_early_split and early_exit_counts:
                logger.info(f"  Early exits: {np.mean(early_exit_rates)*100:.1f}% (avg {np.mean(early_exit_counts):.1f}/batch)")
            
        except Exception as e:
            logger.error(f"Failed to test {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info(f"NEUROSURGEON OPTIMIZATION SUMMARY ({config_desc})")
    logger.info("="*80)
    
    # Check if any model has early exit data
    has_early_exits = any('avg_early_exit_rate' in timings for timings in all_model_timings.values())
    
    if has_early_exits:
        logger.info(f"{'Model':<20} {'Split':<7} {'Total(ms)':<13} {'Std(ms)':<10} {'Accuracy':<10} {'EE Rate':<10} {'Batches':<8}")
        logger.info("-" * 80)
        
        # Sort by total time
        sorted_models = sorted(all_model_timings.items(), key=lambda x: x[1]['total_time'])
        for model_name, timings in sorted_models:
            ee_rate = timings.get('avg_early_exit_rate', 0) * 100
            logger.info(
                f"{model_name:<20} {timings['optimal_split']:<7} {timings['total_time']:<13.1f} "
                f"{timings.get('std_total_time', 0):<10.1f} {timings.get('accuracy', 0):<9.1f}% "
                f"{ee_rate:<9.1f}% {timings['num_batches']:<8}"
            )
    else:
        logger.info(f"{'Model':<20} {'Split':<7} {'Total(ms)':<13} {'Std(ms)':<10} {'Accuracy':<10} {'Batches':<8}")
        logger.info("-" * 80)
        
        # Sort by total time
        sorted_models = sorted(all_model_timings.items(), key=lambda x: x[1]['total_time'])
        for model_name, timings in sorted_models:
            logger.info(
                f"{model_name:<20} {timings['optimal_split']:<7} {timings['total_time']:<13.1f} "
                f"{timings.get('std_total_time', 0):<10.1f} {timings.get('accuracy', 0):<9.1f}% {timings['num_batches']:<8}"
            )
    
    logger.info("="*80)
    
    # Print detailed early exit statistics if available
    if has_early_exits:
        logger.info("\nEARLY EXIT DETAILS:")
        logger.info("-" * 80)
        for model_name, timings in sorted_models:
            if 'avg_early_exit_rate' in timings:
                ee_rate = timings.get('avg_early_exit_rate', 0) * 100
                ee_count = timings.get('avg_early_exit_count', 0)
                logger.info(
                    f"{model_name:<20} Exit Rate: {ee_rate:>5.1f}%  "
                    f"Avg Exits/Batch: {ee_count:>4.1f}"
                )
        logger.info("="*80 + "\n")
    
    # Generate comparison plots
    if auto_plot and all_model_timings:
        suffix = "_".join(filter(None, ["quantized" if enable_quantization else "", "earlyexit" if use_early_split else ""]))
        suffix = f"_{suffix}" if suffix else ""
        _save_comparison_plots(
            Path(plot_path or "plots"), all_model_timings,
            f"NeuroSurgeon Optimal Split ({config_desc})", f"all_models_neurosurgeon{suffix}", plot_show
        )
    
    return all_model_timings

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
    initialize_dataset_loader(1)
    
    all_model_results = {}
    all_model_timings = {}  # For multi-model comparison
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE MODEL TESTING")
    logger.info("Testing all models across all split points")
    logger.info("="*80)
    
    for model_name in SUPPORTED_MODELS:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing model: {model_name}")
        logger.info('='*80)
        
        try:
            # Get model and create DNN Surgery instance
            model = get_model(model_name)
            dnn_surgery = DNNSurgery(model, model_name)
            
            # Get number of layers
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
            
            # Generate throughput comparison plot
            throughput_save_path = str(plot_dir / f"multi_model_throughput_comparison.png")
            plot_multi_model_throughput_comparison(
                all_model_timings,
                show=plot_show,
                save_path=throughput_save_path,
                title="Multi-Model Throughput Comparison",
            )
            logger.info(f"✓ Saved throughput comparison to {throughput_save_path}")
            
            logger.info("✓ All comparison plots generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate multi-model comparison: {str(e)}")
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
        choices=['resnet18', 'resnet50', 'alexnet', 'googlenet', 'efficientnet_b2', 'mobilenet_v3_large'],
        default='resnet18',
        help='Model to use for inference (default: resnet18). Supported: resnet18, resnet50, alexnet, googlenet, efficientnet_b2, mobilenet_v3_large',
    )
    parser.add_argument('--split-point', type=int, default=None,
                       help='Split point for model partitioning (default: None - use NeuroSurgeon optimization)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference (default: 1)')
    parser.add_argument('--num-batches', type=int, default=3,
                       help='Number of batches to process for stable measurements (default: 3)')
    parser.add_argument('--test-splits', type=str,
                       help='Comma-separated split points to test (e.g., "0,1,2,3,4") - overrides NeuroSurgeon')
    parser.add_argument('--num-tests', type=int, default=3,
                       help='Number of tests per split point when testing specific splits (default: 3)')
    parser.add_argument('--test-connection', action='store_true',
                       help='Test connection to server and exit')
    parser.add_argument('--test-all-models', action='store_true',
                       help='Test all supported models across all split points and generate comparison plots')
    parser.add_argument('--test-all-models-split', type=int, default=None,
                       help='Test all models at a specific split point and generate a bar chart comparison (e.g., --test-all-models-split 0)')
    parser.add_argument('--test-all-models-neurosurgeon', action='store_true',
                       help='Test all models using NeuroSurgeon-determined optimal split points and generate comparison plots')
    parser.add_argument('--neurosurgeon-quantize', action='store_true',
                       help='Enable INT8 quantization when testing all models with NeuroSurgeon')
    parser.add_argument('--neurosurgeon-early-split', action='store_true',
                       help='Enable early exit with intermediate classifiers (confidence-based exits at shallow layers)')
    parser.add_argument('--use-neurosurgeon', action='store_true', default=True,
                       help='Use NeuroSurgeon optimization (default: True)')
    parser.add_argument('--no-neurosurgeon', action='store_true',
                       help='Disable NeuroSurgeon optimization (requires --split-point)')
    parser.add_argument('--quantize', action='store_true',
                       help='Enable INT8 dynamic quantization for edge model (reduces memory and improves speed)')
    parser.add_argument('--no-plot', action='store_true',
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
        
        # Validate model is supported
        validate_model(args.model)

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
        
        # Test all models at a single split point
        if args.test_all_models_split is not None:
            logger.info(f"Testing all models at split point {args.test_all_models_split}...")
            test_all_models_single_split(
                args.server_address,
                args.test_all_models_split,
                num_tests=args.num_tests,
                auto_plot=args.auto_plot,
                plot_show=args.plot_show,
                plot_path=args.plot_save or "plots",
            )
            logger.info("All models testing complete!")
            sys.exit(0)
        
        # Test all models with NeuroSurgeon optimization
        if args.test_all_models_neurosurgeon:
            logger.info("Testing all models with NeuroSurgeon optimization...")
            test_all_models_neurosurgeon(
                args.server_address,
                num_batches=args.num_batches,
                auto_plot=args.auto_plot,
                plot_show=args.plot_show,
                plot_path=args.plot_save or "plots",
                enable_quantization=args.neurosurgeon_quantize,
                use_early_split=args.neurosurgeon_early_split,
                batch_size=args.batch_size,
            )
            logger.info("NeuroSurgeon testing complete!")
            sys.exit(0)
        
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
            
            if args.neurosurgeon_early_split:
                logger.info("Early exit enabled for batch processing")
            
            timings_list = run_batch_processing(
                args.server_address,
                args.model,
                split_point,
                args.batch_size,
                args.num_batches,
                auto_plot=args.auto_plot,
                plot_show=args.plot_show,
                plot_path=args.plot_save,
                use_early_exit=args.neurosurgeon_early_split,
            )
            
            # Print batch summary
            avg_total = np.mean([t.get('total_wall_time', 0) for t in timings_list])
            logger.info(f"Batch processing completed. Average time per batch: {avg_total:.1f}ms")
            
            # Calculate accuracy and early exit stats
            if timings_list:
                total_correct = 0
                total_samples = 0
                early_exit_counts = []
                early_exit_rates = []
                
                for timing in timings_list:
                    true_labels = timing['true_labels']
                    predicted_classes = timing.get('predicted_classes', [])
                    
                    for true_label, pred_class in zip(true_labels, predicted_classes):
                        if true_label == pred_class:
                            total_correct += 1
                        total_samples += 1
                    
                    # Collect early exit stats if available
                    if 'early_exit_count' in timing:
                        early_exit_counts.append(timing.get('early_exit_count', 0))
                        early_exit_rates.append(timing.get('early_exit_rate', 0.0))
                
                if total_samples > 0:
                    accuracy = (total_correct / total_samples) * 100
                    logger.info(f"Overall accuracy: {accuracy:.2f}% ({total_correct}/{total_samples})")
                
                # Show early exit summary if available
                if early_exit_counts and args.neurosurgeon_early_split:
                    avg_exit_rate = np.mean(early_exit_rates) * 100
                    logger.info(f"Average early exit rate: {avg_exit_rate:.1f}% (avg {np.mean(early_exit_counts):.1f} exits per batch)")
            
        # Run single inference
        else:
            if args.no_neurosurgeon:
                split_point = args.split_point if args.split_point is not None else 2
                logger.info(f"Running single inference with manual split point {split_point}")
            else:
                split_point = None
                logger.info(f"Running single inference with NeuroSurgeon optimization")
            
            if args.quantize:
                logger.info("Quantization enabled: Edge model will use INT8 dynamic quantization")
            
            if args.neurosurgeon_early_split:
                logger.info("Early exit enabled: Will use confidence-based early exits at intermediate layers")
            
            dnn_surgery = DNNSurgery(get_model(args.model), args.model, enable_quantization=args.quantize)
            
            # Run inference (with or without early exit)
            if args.neurosurgeon_early_split:
                exit_config = EarlyExitConfig(enabled=True, confidence_threshold=0.7, max_exits=3)
                input_tensor, true_labels, class_names = get_input_tensor(args.model, args.batch_size)
                result, timings = run_distributed_inference_with_early_exit(
                    args.model, input_tensor, dnn_surgery, exit_config=exit_config,
                    split_point=split_point, server_address=args.server_address
                )
                timings['true_labels'] = true_labels.tolist()
                timings['class_names'] = class_names
                
                # Generate plot for early exit
                if args.auto_plot:
                    try:
                        plot_path = Path(args.plot_save or "plots")
                        if not plot_path.suffix:
                            plot_path = plot_path / f"{args.model}_split{timings.get('split_point', 'X')}_earlyexit_{datetime.now():%Y%m%d-%H%M%S}.png"
                        plot_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        plot_actual_inference_breakdown(
                            {k: timings.get(k, 0) for k in ['edge_time', 'transfer_time', 'cloud_time', 'total_batch_processing_time']},
                            show=args.plot_show, save_path=str(plot_path),
                            title=f"Early Exit Inference - {args.model}"
                        )
                        timings['actual_split_plot_path'] = str(plot_path.resolve())
                        logger.info(f"Plot saved to {plot_path}")
                    except Exception as e:
                        logger.error(f"Plot failed: {e}")
            else:
                result, timings = run_single_inference(
                    args.server_address, args.model, dnn_surgery, split_point, args.batch_size,
                    auto_plot=args.auto_plot, plot_show=args.plot_show, plot_path=args.plot_save
                )
            
            total_time = timings.get('edge_time', 0) + timings.get('cloud_time', 0) + timings.get('transfer_time', 0)
            actual_split = timings.get('split_point', 'unknown')
            
            print(f"\n✓ Inference completed successfully!")
            print(f"Results:")
            print(f"   Output shape: {result.shape}")
            print(f"   Split point used: {actual_split}")
            print(f"   Total time: {total_time:.1f}ms")
            print(f"   Edge time: {timings.get('edge_time', 0):.1f}ms")
            print(f"   Cloud time: {timings.get('cloud_time', 0):.1f}ms")
            print(f"   Transfer time: {timings.get('transfer_time', 0):.1f}ms")
            
            # Show early exit info if enabled
            if args.neurosurgeon_early_split:
                early_exit_count = timings.get('early_exit_count', 0)
                early_exit_rate = timings.get('early_exit_rate', 0.0)
                print(f"   Early exits: {int(early_exit_count)}/{args.batch_size} ({early_exit_rate*100:.1f}%)")
                
                if 'total_exits' in timings:
                    total_exits = int(timings.get('total_exits', 0))
                    if total_exits > 0:
                        print(f"   Most frequent exit layer: {int(timings.get('most_frequent_exit_layer', -1))}")

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

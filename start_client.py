#!/usr/bin/env python3

import argparse
import logging
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import gc
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

from dataset.imagenet_loader import ImageNetMiniLoader
from dnn_inference_client import DNNInferenceClient, resolve_plot_paths
from dnn_surgery import DNNSurgery
from early_exit import EarlyExitConfig, run_distributed_inference_with_early_exit
from visualization import (
    plot_actual_inference_breakdown,
    plot_actual_split_comparison,
    plot_model_comparison_bar,
    plot_throughput_from_timing,
    plot_split_throughput_comparison,
    plot_model_throughput_comparison_bar,
    plot_quantization_size_reduction,
    plot_quantization_comparison_bar,
    plot_split_timing,
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
_calibration_dataloader = None

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

def get_calibration_dataloader(batch_size: int = 4, num_workers: int = 0):
    """Get a calibration DataLoader for quantization.
    
    Args:
        batch_size: Batch size for calibration (default: 4)
        num_workers: Number of worker processes (default: 0)
        
    Returns:
        DataLoader for calibration data
    """
    global _calibration_dataloader
    
    if _calibration_dataloader is None:
        logger.info(f"Creating calibration DataLoader (batch_size={batch_size})...")
        calibration_loader = ImageNetMiniLoader(batch_size=batch_size, num_workers=num_workers)
        # Use validation set for calibration
        _calibration_dataloader, _ = calibration_loader.get_loader(train=False)
    
    return _calibration_dataloader

def run_distributed_inference(
    model_id: str,
    input_tensor: torch.Tensor,
    dnn_surgery: DNNSurgery,
    split_point: int | None = None,
    server_address: str = "localhost:50051",
    *,
    auto_plot: bool = True,
    plot_show: bool = True,
    plot_path: str | None = None,
    calibration_dataloader = None,
    num_calibration_batches: int = 10,
) -> Tuple[torch.Tensor, Dict]:
    """Run distributed inference with NeuroSurgeon optimization
    
    Args:
        model_id: Model identifier
        input_tensor: Input tensor
        dnn_surgery: DNNSurgery instance for model splitting
        split_point: Optional manual split point (if None, will use NeuroSurgeon optimization)
        server_address: Server address
        calibration_dataloader: Optional DataLoader for quantization calibration.
            Required if quantization is enabled and quantizer not yet initialized.
        num_calibration_batches: Number of batches to use for calibration (default: 10)
        
    Returns:
        Tuple of (result tensor, timing dictionary)
    """
    try:
        split_summary: Optional[List[Dict[str, float]]] = None
        predicted_plot_path: Optional[Path] = None
        actual_plot_path: Optional[Path] = None
        predicted_plot_path_resolved: Optional[str] = None
        actual_plot_path_resolved: Optional[str] = None

        # Use NeuroSurgeon approach if no manual split point provided
        if split_point is None:
            optimal_split, analysis = dnn_surgery.find_optimal_split(
                input_tensor, 
                server_address,
                calibration_dataloader=calibration_dataloader,
                num_calibration_batches=num_calibration_batches
            )
            split_point = optimal_split
            logger.info(f"NeuroSurgeon optimal split point: {split_point}")
            logger.info(f"Predicted total time: {analysis['min_total_time']:.2f}ms")
            split_summary = analysis.get("split_summary")
        else:
            logger.info(f"Using manual split point: {split_point}")
            success = dnn_surgery._send_split_decision_to_server(split_point, server_address)
            if not success:
                logger.error("Failed to send manual split decision to server")

        if auto_plot:
            predicted_plot_path, actual_plot_path = resolve_plot_paths(model_id, split_point, plot_path)

            if split_summary is None:
                logger.warning("Split summary not available; skipping predicted split chart")
            else:
                logger.info("Auto-plot enabled; generating split timing chart")
                try:
                    predicted_plot_path.parent.mkdir(parents=True, exist_ok=True)
                    plot_split_timing(
                        split_summary,
                        show=plot_show,
                        save_path=str(predicted_plot_path),
                        title=f"Predicted Split Timing ({model_id})",
                    )
                    predicted_plot_path_resolved = str(predicted_plot_path.resolve())
                    logger.info(
                        "Predicted split timing plot saved to %s",
                        predicted_plot_path_resolved,
                    )
                except ImportError as plt_error:
                    logger.warning(
                        "Auto-plot requested but matplotlib is unavailable: %s",
                        plt_error,
                    )
                except Exception as plot_error:
                    logger.error("Failed to render predicted split chart: %s", plot_error)

        dnn_surgery.splitter.set_split_point(split_point)
        edge_model = dnn_surgery.splitter.get_edge_model()
        cloud_model = dnn_surgery.splitter.get_cloud_model()
        requires_cloud_processing = cloud_model is not None
        
        if not requires_cloud_processing:
            logger.info(f"Split point {split_point} means all inference on client side - no server communication needed")
        
        # Create client with edge model (pass quantize_transfer from dnn_surgery)
        client = DNNInferenceClient(server_address, edge_model, quantize_transfer=dnn_surgery.quantize_transfer)
        
        # Run inference
        result, timings = client.process_tensor(input_tensor, model_id, requires_cloud_processing)
        
        # Get timing summary
        timing_summary = client.get_timing_summary()
        timings.update(timing_summary)

        total_metrics = {
            "edge_time": timings.get("edge_time", 0.0),
            "transfer_time": timings.get("transfer_time", 0.0),
            "cloud_time": timings.get("cloud_time", 0.0),
            "total_batch_processing_time": timings.get("total_batch_processing_time"),
        }

        if auto_plot and actual_plot_path is not None:
            try:
                actual_plot_path.parent.mkdir(parents=True, exist_ok=True)
                plot_actual_inference_breakdown(
                    total_metrics,
                    show=plot_show,
                    save_path=str(actual_plot_path),
                    title=f"Measured Inference Breakdown ({model_id})",
                )
                actual_plot_path_resolved = str(actual_plot_path.resolve())
                logger.info(
                    "Measured inference plot saved to %s",
                    actual_plot_path_resolved,
                )
                
                # Generate companion throughput plot
                throughput_plot_path = actual_plot_path.with_name(
                    actual_plot_path.stem.replace("_actual", "_throughput") + actual_plot_path.suffix
                )
                plot_throughput_from_timing(
                    total_metrics,
                    show=plot_show,
                    save_path=str(throughput_plot_path),
                    title=f"Inference Throughput ({model_id})",
                )
                logger.info(
                    "Throughput plot saved to %s",
                    throughput_plot_path.resolve(),
                )
            except ImportError as plt_error:
                logger.warning(
                    "Auto-plot requested but matplotlib is unavailable: %s",
                    plt_error,
                )
            except Exception as plot_error:
                logger.error("Failed to render measured inference chart: %s", plot_error)

        if predicted_plot_path_resolved:
            timings['predicted_split_plot_path'] = predicted_plot_path_resolved
        if actual_plot_path_resolved:
            timings['actual_split_plot_path'] = actual_plot_path_resolved
        
        # Add split point info
        timings['split_point'] = split_point
        
        # Add analysis data if available (from NeuroSurgeon optimization)
        if split_point is not None and 'analysis' in locals():
            timings['split_analysis'] = analysis
        
        return result, timings
        
    except Exception as e:
        logger.error(f"Distributed inference failed: {str(e)}")
        raise

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
    
    # Get calibration dataloader if quantization is enabled
    calibration_dataloader = None
    if dnn_surgery.enable_quantization:
        calibration_dataloader = get_calibration_dataloader(batch_size=4)
    
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
        calibration_dataloader=calibration_dataloader,
        num_calibration_batches=10,
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
    enable_quantization: bool = False,
    quantize_transfer: bool = False,
) -> List[Dict]:
    """Run multiple batches and collect timing statistics"""
    validate_model(model_name)
    
    # Initialize dataset
    initialize_dataset_loader(batch_size)
    
    model = get_model(model_name)
    dnn_surgery = DNNSurgery(model, model_name, enable_quantization=enable_quantization, quantize_transfer=quantize_transfer)
    
    # Get calibration dataloader if quantization is enabled
    calibration_dataloader = None
    if enable_quantization:
        calibration_dataloader = get_calibration_dataloader(batch_size=4)
    
    # Setup early exit if requested
    exit_config = None
    if use_early_exit:
        logger.info("Early exit enabled for batch processing")
        exit_config = EarlyExitConfig(
            enabled=True,
            entropy_threshold=0.3,  # Maximum entropy for early exit (lower = more confident)
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
                        calibration_dataloader=calibration_dataloader,
                        num_calibration_batches=10,
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
                        calibration_dataloader=calibration_dataloader,
                        num_calibration_batches=10,
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
        
        # Clean up tensors and free memory after each batch
        del input_tensor, result, true_labels
        gc.collect()
    
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
    quantize_transfer: bool = False,
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
        quantize_transfer: Whether to enable INT8 quantization for intermediate tensor transfers
        use_early_split: Whether to enable early exit with intermediate classifiers
        batch_size: Batch size for each inference run (default: 1)
        
    Returns:
        Dictionary mapping model names to their optimal split timing results
    """
    initialize_dataset_loader(batch_size)
    
    all_model_timings = {}
    all_quantization_metrics = {}  # Collect quantization metrics from all models
    
    # Build configuration description
    config_parts = []
    if enable_quantization:
        config_parts.append("Model Quantization")
    if quantize_transfer:
        config_parts.append("Transfer Quantization")
    if use_early_split:
        config_parts.append("Early Exit")
    
    config_desc = " + ".join(config_parts) if config_parts else "Standard"
    
    for model_name in SUPPORTED_MODELS:
        try:
            # Get model and create DNN Surgery instance
            model = get_model(model_name)
            dnn_surgery = DNNSurgery(model, model_name, enable_quantization=enable_quantization, quantize_transfer=quantize_transfer)
            
            # Get calibration dataloader if quantization is enabled
            calibration_dataloader = None
            if enable_quantization:
                calibration_dataloader = get_calibration_dataloader(batch_size=4)
            
            # Configure early exit if requested
            exit_config = None
            if use_early_split:
                logger.info(f"Using early exit configuration with intermediate classifiers")
                exit_config = EarlyExitConfig(
                    enabled=True,
                    entropy_threshold=0.3,  # Maximum entropy for early exit (lower = more confident)
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
                
                # Wrap inference in no_grad to prevent gradient tracking
                with torch.no_grad():
                    if use_early_split:
                        result, timings = run_distributed_inference_with_early_exit(
                            model_name,
                            input_tensor,
                            dnn_surgery,
                            exit_config=exit_config,
                            split_point=current_split,  # None for first batch, then reuse optimal split
                            server_address=server_address,
                            auto_plot=False,  # Disable auto-plotting during batch runs
                            plot_show=False,
                            plot_path=None,
                            calibration_dataloader=calibration_dataloader,
                            num_calibration_batches=10,
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
                            calibration_dataloader=calibration_dataloader,
                            num_calibration_batches=10,
                        )
                
                    # Store optimal split from first batch
                    if batch_idx == 0:
                        optimal_split = timings.get('split_point')
                    
                    # Collect timing metrics
                    edge_times.append(timings.get('edge_time', 0))
                    transfer_times.append(timings.get('transfer_time', 0))
                    cloud_times.append(timings.get('cloud_time', 0))
                    total_times.append(timings.get('edge_time', 0) + timings.get('transfer_time', 0) + timings.get('cloud_time', 0))
                    
                    # Calculate accuracy for this batch
                    batch_correct, batch_total = _calculate_batch_accuracy(result, true_labels, batch_idx + 1)
                    correct_predictions += batch_correct
                    total_samples += batch_total
                    
                    # Clean up batch tensors immediately
                    del result, input_tensor, true_labels
                
                # Force garbage collection every few batches to prevent memory buildup
                if (batch_idx + 1) % 5 == 0:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

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
            
            # Collect quantization metrics if enabled
            if enable_quantization:
                model_metrics = dnn_surgery.quantizer.get_size_metrics()
                if model_metrics:
                    all_quantization_metrics.update(model_metrics)
                    logger.info(f"  Quantization metrics collected: {len(model_metrics)} entries")
            
        except Exception as e:
            logger.error(f"Failed to test {model_name}: {str(e)}")
            traceback.print_exc()
            continue
        finally:
            # Clean up model and DNN surgery instance after each model test
            if 'model' in locals():
                del model
            if 'dnn_surgery' in locals():
                del dnn_surgery
            # Force garbage collection and clear CUDA cache
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
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
        
        # Generate quantization plots if model quantization was enabled
        if enable_quantization:
            logger.info("\n" + "="*80)
            logger.info("QUANTIZATION METRICS")
            logger.info("="*80)
            
            if all_quantization_metrics:
                # Log quantization summary
                for model_name in SUPPORTED_MODELS:
                    edge_key = f"{model_name}_edge"
                    cloud_key = f"{model_name}_cloud"
                    
                    if edge_key in all_quantization_metrics:
                        metrics = all_quantization_metrics[edge_key]
                        logger.info(f"{model_name} (edge model):")
                        logger.info(f"  Original: {metrics['original_size_mb']:.2f} MB → Quantized: {metrics['quantized_size_mb']:.2f} MB")
                        logger.info(f"  Compression: {metrics['compression_ratio']:.2f}x ({metrics['num_quantizable_layers']} layers)")
                    
                    if cloud_key in all_quantization_metrics:
                        metrics = all_quantization_metrics[cloud_key]
                        logger.info(f"{model_name} (cloud model):")
                        logger.info(f"  Original: {metrics['original_size_mb']:.2f} MB → Quantized: {metrics['quantized_size_mb']:.2f} MB")
                        logger.info(f"  Compression: {metrics['compression_ratio']:.2f}x ({metrics['num_quantizable_layers']} layers)")
                
                logger.info("="*80)
                
                # Generate quantization visualization plots
                try:
                    plot_dir = Path(plot_path or "plots")
                    plot_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    
                    # Bar chart comparison
                    quant_bar_path = plot_dir / f"quantization_comparison_{timestamp}.png"
                    plot_quantization_comparison_bar(
                        all_quantization_metrics,
                        show=plot_show,
                        save_path=str(quant_bar_path),
                        title="Model Quantization Compression (INT8)"
                    )
                    logger.info(f"✓ Quantization comparison chart saved: {quant_bar_path.name}")
                    
                    # Detailed size reduction plot
                    quant_detail_path = plot_dir / f"quantization_size_reduction_{timestamp}.png"
                    plot_quantization_size_reduction(
                        all_quantization_metrics,
                        show=plot_show,
                        save_path=str(quant_detail_path),
                        title="Model Size Reduction via Quantization"
                    )
                    logger.info(f"✓ Quantization size reduction chart saved: {quant_detail_path.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate quantization plots: {e}")
                    traceback.print_exc()
            else:
                logger.warning("No quantization metrics collected from any model")
    
    return all_model_timings


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
    parser.add_argument('--test-all-models-split', type=int, default=None,
                       help='Test all models at a specific split point and generate a bar chart comparison (e.g., --test-all-models-split 0)')
    parser.add_argument('--test-all-models-neurosurgeon', action='store_true',
                       help='Test all models using NeuroSurgeon-determined optimal split points and generate comparison plots')
    parser.add_argument('--neurosurgeon-quantize', action='store_true',
                       help='Enable INT8 model quantization (weights) for edge/cloud models - works with all testing modes')
    parser.add_argument('--neurosurgeon-early-split', action='store_true',
                       help='Enable early exit with intermediate classifiers (confidence-based exits at shallow layers)')
    parser.add_argument('--use-neurosurgeon', action='store_true', default=True,
                       help='Use NeuroSurgeon optimization (default: True)')
    parser.add_argument('--quantize-transfer', action='store_true',
                       help='Enable INT8 quantization for intermediate tensors during transfer (reduces bandwidth by ~4x)')
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
                quantize_transfer=args.quantize_transfer,
                use_early_split=args.neurosurgeon_early_split,
                batch_size=args.batch_size,
            )
            logger.info("NeuroSurgeon testing complete!")
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
            
        # Run batch processing
        elif args.num_batches > 1:
            logger.info(f"Running batch processing: {args.num_batches} batches")
            
            split_point = args.split_point 
            
            if args.neurosurgeon_early_split:
                logger.info("Early exit enabled for batch processing")
            
            if args.neurosurgeon_quantize:
                logger.info("Model quantization enabled: Edge/cloud models will use INT8 post-training static quantization (PTQ)")
            
            if args.quantize_transfer:
                logger.info("Transfer quantization enabled: Intermediate tensors will be quantized during transfer")
            
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
                enable_quantization=args.neurosurgeon_quantize,
                quantize_transfer=args.quantize_transfer,
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
            if args.split_point is not None:
                split_point = args.split_point
                logger.info(f"Running single inference with manual split point {split_point}")
            else:
                split_point = None
                logger.info(f"Running single inference with NeuroSurgeon optimization")

            if args.neurosurgeon_quantize:
                logger.info("Quantization enabled: Edge/cloud models will use INT8 post-training static quantization (PTQ)")
            
            if args.quantize_transfer:
                logger.info("Transfer quantization enabled: Intermediate tensors will be quantized (FP32→INT8) during transfer")
            
            if args.neurosurgeon_early_split:
                logger.info("Early exit enabled: Will use confidence-based early exits at intermediate layers")
            
            dnn_surgery = DNNSurgery(get_model(args.model), args.model, enable_quantization=args.neurosurgeon_quantize, quantize_transfer=args.quantize_transfer)
            
            # Run inference (with or without early exit)
            if args.neurosurgeon_early_split:
                exit_config = EarlyExitConfig(enabled=True, entropy_threshold=0.3, max_exits=3)
                input_tensor, true_labels, class_names = get_input_tensor(args.model, args.batch_size)
                result, timings = run_distributed_inference_with_early_exit(
                    args.model, input_tensor, dnn_surgery, exit_config=exit_config,
                    split_point=split_point, server_address=args.server_address,
                    auto_plot=args.auto_plot, plot_show=args.plot_show, plot_path=args.plot_save
                )
                timings['true_labels'] = true_labels.tolist()
                timings['class_names'] = class_names
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

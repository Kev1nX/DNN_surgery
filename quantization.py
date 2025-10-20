"""Model quantization utilities for DNN Surgery.

Implements dynamic quantization (INT8) for both client and server models.
Dynamic quantization converts weights to INT8 while keeping activations in FP32,
providing a good balance between performance and accuracy for inference.
"""

import logging
from typing import Optional, Set
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelQuantizer:
    """Handles model quantization for edge and cloud deployment.
    
    Uses dynamic quantization (INT8) which:
    - Converts weights to INT8 (reduces model size by ~4x)
    - Keeps activations in FP32 (maintains accuracy)
    - Reduces memory bandwidth requirements
    - Improves inference speed on CPU
    """
    
    # Layers that support dynamic quantization
    QUANTIZABLE_LAYERS = {nn.Linear, nn.LSTM, nn.GRU, nn.LSTMCell, nn.GRUCell, nn.RNNCell}
    
    def __init__(self):
        self._quantized_models = {}
        logger.info("Initialized ModelQuantizer with dynamic INT8 quantization")
    
    def quantize_model(
        self,
        model: nn.Module,
        model_name: str = "unknown",
        inplace: bool = False
    ) -> nn.Module:
        """Apply dynamic INT8 quantization to a model.
        
        Args:
            model: PyTorch model to quantize
            model_name: Name identifier for the model
            inplace: If True, modify the model in-place; otherwise create a copy
            
        Returns:
            Quantized model
            
        Raises:
            RuntimeError: If quantization fails
        """
        try:
            if not inplace:
                model = self._prepare_model_copy(model)
            
            # Ensure model is in eval mode
            model.eval()
            
            # Count quantizable layers before quantization
            num_quantizable = self._count_quantizable_layers(model)
            
            if num_quantizable == 0:
                logger.warning(
                    f"Model {model_name} has no quantizable layers. "
                    "Dynamic quantization typically works on Linear/LSTM/GRU layers."
                )
                return model
            
            logger.info(f"Quantizing {model_name}: {num_quantizable} quantizable layers found")
            
            # Apply dynamic quantization
            quantized_model = quantize_dynamic(
                model,
                qconfig_spec=self.QUANTIZABLE_LAYERS,
                dtype=torch.qint8
            )
            
            # Calculate compression ratio
            original_size = self._calculate_model_size(model)
            quantized_size = self._calculate_model_size(quantized_model)
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
            
            logger.info(
                f"Quantization complete for {model_name}:\n"
                f"  Original size: {original_size / 1e6:.2f} MB\n"
                f"  Quantized size: {quantized_size / 1e6:.2f} MB\n"
                f"  Compression ratio: {compression_ratio:.2f}x\n"
                f"  Memory saved: {(original_size - quantized_size) / 1e6:.2f} MB"
            )
            
            # Cache the quantized model
            self._quantized_models[model_name] = quantized_model
            
            return quantized_model
            
        except Exception as e:
            error_msg = f"Failed to quantize model {model_name}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _prepare_model_copy(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model for quantization.
        
        Args:
            model: Original model
            
        Returns:
            Deep copy of the model
        """
        import copy
        return copy.deepcopy(model)
    
    def _count_quantizable_layers(self, model: nn.Module) -> int:
        """Count the number of quantizable layers in the model.
        
        Args:
            model: Model to analyze
            
        Returns:
            Number of quantizable layers
        """
        count = 0
        for module in model.modules():
            if type(module) in self.QUANTIZABLE_LAYERS:
                count += 1
        return count
    
    def _calculate_model_size(self, model: nn.Module) -> int:
        """Calculate the size of a model in bytes.
        
        Args:
            model: Model to measure
            
        Returns:
            Size in bytes
        """
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        for buffer in model.buffers():
            total_size += buffer.numel() * buffer.element_size()
        return total_size
    
    def get_quantized_model(self, model_name: str) -> Optional[nn.Module]:
        """Retrieve a cached quantized model.
        
        Args:
            model_name: Name of the quantized model
            
        Returns:
            Quantized model if found, None otherwise
        """
        return self._quantized_models.get(model_name)
    
    def clear_cache(self) -> None:
        """Clear all cached quantized models."""
        self._quantized_models.clear()
        logger.info("Cleared quantization cache")


def quantize_for_inference(
    model: nn.Module,
    model_name: str = "model"
) -> nn.Module:
    """Convenience function to quantize a model for inference.
    
    Args:
        model: PyTorch model to quantize
        model_name: Name identifier for logging
        
    Returns:
        Quantized model ready for inference
        
    Example:
        >>> model = models.resnet18(pretrained=True)
        >>> quantized_model = quantize_for_inference(model, "resnet18")
        >>> result = quantized_model(input_tensor)
    """
    quantizer = ModelQuantizer()
    return quantizer.quantize_model(model, model_name, inplace=False)


def compare_model_sizes(
    original_model: nn.Module,
    quantized_model: nn.Module,
    model_name: str = "model"
) -> dict:
    """Compare sizes and compression ratio between original and quantized models.
    
    Args:
        original_model: Original FP32 model
        quantized_model: Quantized INT8 model
        model_name: Name for logging
        
    Returns:
        Dictionary with size comparison metrics
    """
    quantizer = ModelQuantizer()
    
    original_size = quantizer._calculate_model_size(original_model)
    quantized_size = quantizer._calculate_model_size(quantized_model)
    
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
    memory_saved = original_size - quantized_size
    
    comparison = {
        "model_name": model_name,
        "original_size_mb": original_size / 1e6,
        "quantized_size_mb": quantized_size / 1e6,
        "compression_ratio": compression_ratio,
        "memory_saved_mb": memory_saved / 1e6,
        "size_reduction_percent": (memory_saved / original_size * 100) if original_size > 0 else 0
    }
    
    logger.info(
        f"Size comparison for {model_name}:\n"
        f"  Original: {comparison['original_size_mb']:.2f} MB\n"
        f"  Quantized: {comparison['quantized_size_mb']:.2f} MB\n"
        f"  Compression: {compression_ratio:.2f}x\n"
        f"  Saved: {comparison['memory_saved_mb']:.2f} MB ({comparison['size_reduction_percent']:.1f}%)"
    )
    
    return comparison


__all__ = [
    "ModelQuantizer",
    "quantize_for_inference",
    "compare_model_sizes",
]

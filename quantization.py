"""Model quantization utilities for DNN Surgery.

Implements dynamic quantization (INT8) for both client and server models.
Dynamic quantization converts weights to INT8 while keeping activations in FP32,
providing a good balance between performance and accuracy for inference.

IMPORTANT: This implementation uses DYNAMIC quantization:
- Quantizes: Model WEIGHTS (Linear, LSTM, GRU layers) → INT8
- Does NOT quantize: ACTIVATIONS/intermediate tensors → remain FP32
- Use case: Reduce model size and memory bandwidth for transfer

For full INT8 quantization (including activations), static quantization 
with calibration is required
"""

import copy
import logging
from typing import Optional
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
    
    Note: Dynamic quantization is optimized for layers with large weight matrices.
    Convolutional layers (Conv1d, Conv2d, Conv3d) are NOT supported by dynamic
    quantization and require static quantization instead.
    """
    
    # Layers that support dynamic quantization (per PyTorch 2.9+ documentation)
    # See: https://pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html
    QUANTIZABLE_LAYERS = {nn.Linear, nn.LSTM, nn.GRU, nn.LSTMCell, nn.GRUCell, nn.RNNCell}
    
    def __init__(self):
        self._quantized_models = {}
        self._size_metrics = {}  # Track original and quantized sizes
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
            
            # Cache the quantized model and size metrics
            self._quantized_models[model_name] = quantized_model
            self._size_metrics[model_name] = {
                'original_size_bytes': original_size,
                'quantized_size_bytes': quantized_size,
                'original_size_mb': original_size / 1e6,
                'quantized_size_mb': quantized_size / 1e6,
                'compression_ratio': compression_ratio,
                'memory_saved_mb': (original_size - quantized_size) / 1e6,
                'num_quantizable_layers': num_quantizable,
            }
            
            return quantized_model
            
        except Exception as e:
            error_msg = f"Failed to quantize model {model_name}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _prepare_model_copy(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model for quantization."""
        return copy.deepcopy(model)
    
    def _count_quantizable_layers(self, model: nn.Module) -> int:
        """Count the number of quantizable layers in the model."""
        count = 0
        for module in model.modules():
            if type(module) in self.QUANTIZABLE_LAYERS:
                count += 1
        return count
    
    def _calculate_model_size(self, model: nn.Module) -> int:
        """Calculate the size of a model in bytes."""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        for buffer in model.buffers():
            total_size += buffer.numel() * buffer.element_size()
        return total_size
    
    def get_quantized_model(self, model_name: str) -> Optional[nn.Module]:
        """Retrieve a cached quantized model."""
        return self._quantized_models.get(model_name)
    
    def get_size_metrics(self, model_name: str = None):
        """Retrieve size metrics for a specific model or all models.
        
        Args:
            model_name: Optional model name. If None, returns metrics for all models.
            
        Returns:
            Dictionary of size metrics for the specified model, or dict of all metrics.
        """
        if model_name is not None:
            return self._size_metrics.get(model_name)
        return self._size_metrics.copy()
    
    def clear_cache(self) -> None:
        """Clear all cached quantized models."""
        self._quantized_models.clear()
        self._size_metrics.clear()
        logger.info("Cleared quantization cache")
    
    @staticmethod
    def quantize_tensor(
        tensor: torch.Tensor,
        dtype: torch.dtype = torch.qint8
    ) -> torch.Tensor:
        """Quantize a single tensor for efficient transfer/storage.
        
        Converts FP32 tensors to INT8 using linear scaling. This reduces
        tensor size by ~4x for network transfer between edge and cloud.
        
        Args:
            tensor: Input tensor to quantize (must be float type)
            dtype: Target quantized dtype (default: torch.qint8)
            
        Returns:
            Quantized tensor with built-in scale and zero-point parameters
            
        Note:
            Dequantization is handled by PyTorch's built-in .dequantize() method
            which reconstructs the FP32 approximation: tensor_fp32 = scale * (tensor_int8 - zero_point)
        """
        if not tensor.is_floating_point():
            raise ValueError(f"Can only quantize float tensors, got {tensor.dtype}")
        
        # Calculate scale and zero_point
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Determine quantization parameters based on dtype
        if dtype == torch.qint8:
            qmin, qmax = -128, 127
        elif dtype == torch.quint8:
            qmin, qmax = 0, 255
        else:
            raise ValueError(f"Unsupported quantization dtype: {dtype}")
        
        # Compute scale and zero_point
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - torch.round(min_val / scale).to(torch.int)
        
        # Clamp zero_point to valid range
        zero_point = torch.clamp(zero_point, qmin, qmax).item()
        
        # Quantize the tensor
        quantized = torch.quantize_per_tensor(
            tensor,
            scale=scale.item(),
            zero_point=int(zero_point),
            dtype=dtype
        )
        
        return quantized
    
    @staticmethod
    def calculate_tensor_compression_ratio(original: torch.Tensor, quantized: torch.Tensor) -> float:
        """Calculate compression ratio achieved by tensor quantization.
        
        Args:
            original: Original FP32 tensor
            quantized: Quantized tensor
            
        Returns:
            Compression ratio (e.g., 4.0 means 4x smaller)
        """
        original_size = original.numel() * original.element_size()
        quantized_size = quantized.numel() * quantized.element_size()
        return original_size / quantized_size if quantized_size > 0 else 1.0


__all__ = ["ModelQuantizer"]

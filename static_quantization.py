"""Static quantization implementation for DNN Surgery.

Static quantization (INT8) requires calibration data and converts both weights
and activations to INT8, providing better performance than dynamic quantization
but requiring a calibration step.
"""

import logging
import torch
import torch.nn as nn
from torch.quantization import get_default_qconfig, prepare, convert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StaticQuantizer:
    """Handles static INT8 quantization for models."""
    
    def __init__(self):
        self._size_metrics = {}
        logger.info("Initialized StaticQuantizer")
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: torch.utils.data.DataLoader,
        model_name: str = "unknown",
        backend: str = "fbgemm"
    ) -> nn.Module:
        """Apply static INT8 quantization to a model.
        
        Args:
            model: PyTorch model to quantize
            calibration_data: DataLoader with calibration samples
            model_name: Name identifier for the model
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
            
        Returns:
            Quantized model
        """
        try:
            model = model.cpu()
            model.eval()
            
            # Set quantization backend
            torch.backends.quantized.engine = backend
            
            # Configure quantization
            model.qconfig = get_default_qconfig(backend)
            
            # Prepare model for calibration
            model_prepared = prepare(model, inplace=False)
            
            # Calibrate with sample data
            logger.info(f"Calibrating {model_name}...")
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(calibration_data):
                    if batch_idx >= 10:  # Use 10 batches for calibration
                        break
                    model_prepared(data.cpu())
            
            # Convert to quantized model
            quantized_model = convert(model_prepared, inplace=False)
            
            # Calculate metrics
            original_size = self._calculate_model_size(model)
            quantized_size = self._calculate_model_size(quantized_model)
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
            
            logger.info(
                f"Static quantization complete for {model_name}:\n"
                f"  Original size: {original_size / 1e6:.2f} MB\n"
                f"  Quantized size: {quantized_size / 1e6:.2f} MB\n"
                f"  Compression ratio: {compression_ratio:.2f}x"
            )
            
            self._size_metrics[model_name] = {
                'original_size_mb': original_size / 1e6,
                'quantized_size_mb': quantized_size / 1e6,
                'compression_ratio': compression_ratio,
            }
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Static quantization failed for {model_name}: {str(e)}")
            raise RuntimeError(f"Static quantization failed: {str(e)}") from e
    
    def _calculate_model_size(self, model: nn.Module) -> int:
        """Calculate model size in bytes."""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        for buffer in model.buffers():
            total_size += buffer.numel() * buffer.element_size()
        return total_size
    
    def get_size_metrics(self, model_name: str = None):
        """Retrieve size metrics."""
        if model_name is not None:
            return self._size_metrics.get(model_name)
        return self._size_metrics.copy()


__all__ = ["StaticQuantizer"]

"""Static quantization implementation for DNN Surgery.

Static quantization (INT8) requires calibration data and converts both weights
and activations to INT8, providing better performance than dynamic quantization
but requiring a calibration step.
"""

import logging
import copy
import torch
import torch.nn as nn
from torch.quantization import get_default_qconfig, prepare, convert, fuse_modules
from typing import Optional

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
            # Make a copy to avoid modifying original
            model = copy.deepcopy(model)
            model = model.cpu()
            model.eval()
            
            # Set quantization backend
            torch.backends.quantized.engine = backend
            
            # Add quantization stubs
            model = self._add_quantization_stubs(model)
            
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
    
    def _add_quantization_stubs(self, model: nn.Module) -> nn.Module:
        """Add quantization/dequantization stubs to model."""
        class QuantizableModel(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.quant = torch.quantization.QuantStub()
                self.model = base_model
                self.dequant = torch.quantization.DeQuantStub()
            
            def forward(self, x):
                x = self.quant(x)
                x = self.model(x)
                x = self.dequant(x)
                return x
        
        return QuantizableModel(model)
    
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
    
    def quantize_split_models(
        self,
        edge_model: Optional[nn.Module],
        cloud_model: Optional[nn.Module],
        calibration_data: torch.utils.data.DataLoader,
        model_name: str = "unknown",
        backend: str = "fbgemm"
    ) -> tuple[Optional[nn.Module], Optional[nn.Module]]:
        """Quantize both edge and cloud models after splitting.
        
        Args:
            edge_model: Edge model to quantize (None if split_point=0)
            cloud_model: Cloud model to quantize (None if all edge)
            calibration_data: DataLoader with calibration samples
            model_name: Base model name
            backend: Quantization backend
            
        Returns:
            Tuple of (quantized_edge_model, quantized_cloud_model)
        """
        quantized_edge = None
        quantized_cloud = None
        
        # Quantize edge model if present
        if edge_model is not None:
            logger.info(f"Quantizing edge model for {model_name}...")
            edge_name = f"{model_name}_edge"
            
            try:
                # Create calibration data for edge
                edge_calibration = self._prepare_edge_calibration(
                    edge_model, calibration_data
                )
                quantized_edge = self.quantize_model(
                    edge_model, edge_calibration, edge_name, backend
                )
            except Exception as e:
                logger.error(f"Edge quantization failed: {e}")
                quantized_edge = edge_model
        
        # Quantize cloud model if present
        if cloud_model is not None:
            logger.info(f"Quantizing cloud model for {model_name}...")
            cloud_name = f"{model_name}_cloud"
            
            try:
                # Create calibration data for cloud (using edge outputs)
                if edge_model is not None:
                    cloud_calibration = self._prepare_cloud_calibration(
                        edge_model, cloud_model, calibration_data
                    )
                else:
                    cloud_calibration = calibration_data
                
                quantized_cloud = self.quantize_model(
                    cloud_model, cloud_calibration, cloud_name, backend
                )
            except Exception as e:
                logger.error(f"Cloud quantization failed: {e}")
                quantized_cloud = cloud_model
        
        return quantized_edge, quantized_cloud
    
    def _prepare_edge_calibration(
        self,
        edge_model: nn.Module,
        calibration_data: torch.utils.data.DataLoader
    ) -> list:
        """Prepare calibration data for edge model."""
        edge_outputs = []
        edge_model.eval()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(calibration_data):
                if batch_idx >= 10:
                    break
                edge_outputs.append((data.cpu(), target))
        
        # Return as simple list that can be iterated
        return edge_outputs
    
    def _prepare_cloud_calibration(
        self,
        edge_model: nn.Module,
        cloud_model: nn.Module,
        calibration_data: torch.utils.data.DataLoader
    ) -> list:
        """Prepare calibration data for cloud model using edge outputs."""
        cloud_inputs = []
        edge_model.eval()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(calibration_data):
                if batch_idx >= 10:
                    break
                # Get edge output to use as cloud input
                edge_output = edge_model(data.cpu())
                cloud_inputs.append((edge_output, target))
        
        return cloud_inputs


__all__ = ["StaticQuantizer"]

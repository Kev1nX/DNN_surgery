"""Model quantization utilities for DNN Surgery.
"""

import copy
import logging
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import types
from torch.quantization import prepare, convert, get_default_qconfig, fuse_modules, QuantStub, DeQuantStub
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelQuantizer:
    """Handles model quantization for edge and cloud deployment.
    
    Quantizes full models offline and saves them to disk. At runtime, loads
    pre-quantized models and splits them (no runtime quantization).
    """
    
    QUANTIZABLE_LAYERS = {nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.ReLU6}
    
    def __init__(self, calibration_dataloader: Optional[DataLoader] = None, num_calibration_batches: int = 10, checkpoint_dir: str = "checkpoints/quantized"):
        """Initialize ModelQuantizer with post-training static quantization.
        
        Args:
            calibration_dataloader: DataLoader providing representative calibration data.
            num_calibration_batches: Number of batches to use for calibration (default: 10)
            checkpoint_dir: Directory to save/load quantized models (default: checkpoints/quantized)
        """
        self._quantized_models = {}
        self._size_metrics = {}
        self._calibration_dataloader = calibration_dataloader
        self._num_calibration_batches = num_calibration_batches
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized ModelQuantizer: {num_calibration_batches} calibration batches, checkpoint_dir={checkpoint_dir}")
    
    def load_or_quantize_model(self, model: nn.Module, model_name: str, calibration_dataloader: Optional[DataLoader] = None) -> nn.Module:
        """Load pre-quantized model from disk, or quantize and save if not exists.
        
        Args:
            model: PyTorch model to quantize (only used if checkpoint doesn't exist)
            model_name: Model identifier for checkpoint naming
            calibration_dataloader: DataLoader for calibration (only used if quantizing)
            
        Returns:
            Quantized full model ready for splitting
        """
        checkpoint_path = self._checkpoint_dir / f"{model_name}_quantized.pt"
        
        if checkpoint_path.exists():
            logger.info(f"Loading pre-quantized model from {checkpoint_path}")
            try:
                quantized_model = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                quantized_model.eval()
                logger.info(f"Successfully loaded quantized {model_name}")
                return quantized_model
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Re-quantizing...")
        quantized_model = self._quantize_model(model, model_name, calibration_dataloader)
        
        logger.info(f"Saving quantized model to {checkpoint_path}")
        torch.save(quantized_model, checkpoint_path)
        
        return quantized_model
    
    def _quantize_model(
        self,
        model: nn.Module,
        model_name: str = "unknown",
        calibration_dataloader: Optional[DataLoader] = None
    ) -> nn.Module:
        try:
            model = self._prepare_model_copy(model)
            model = self._wrap_with_quant_stubs(model)
            
            # Ensure model is in eval mode and on CPU (required for quantization)
            model.eval()
            model = model.cpu()
            
            torch.backends.quantized.engine = 'fbgemm'
            
            # Count quantizable layers before quantization
            num_quantizable = self._count_quantizable_layers(model)
            
            if num_quantizable == 0:
                return model
            
            logger.info(f"Quantizing {model_name}: {num_quantizable} quantizable layers found")
            
            # Fuse modules for better performance (Conv+BN+ReLU, Conv+ReLU, etc.)
            model = self._fuse_modules(model, model_name)
            
            # Set quantization configuration based on backend
            backend = torch.backends.quantized.engine
            model.qconfig = get_default_qconfig(backend)
            logger.info(f"Using quantization backend: {backend}")
            
            # Prepare model for quantization (insert observers)
            model = prepare(model, inplace=True)
            
            # Calibrate the model with representative data
            dataloader = calibration_dataloader or self._calibration_dataloader
            if dataloader is not None:
                logger.info(f"Running calibration for {model_name} using DataLoader...")
                self._calibrate_with_dataloader(model, dataloader)
            else:
                logger.warning(
                    f"No calibration dataloader provided for {model_name}. "
                    "Using default random data calibration."
                )
                self._default_calibration(model)
            
            # Convert to quantized model
            quantized_model = convert(model, inplace=True)
            
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
            
            # Cache only size metrics (not the model itself to save memory)
            self._size_metrics[model_name] = {
                'original_size_bytes': original_size,
                'quantized_size_bytes': quantized_size,
                'original_size_mb': original_size / 1e6,
                'quantized_size_mb': quantized_size / 1e6,
                'compression_ratio': compression_ratio,
                'memory_saved_mb': (original_size - quantized_size) / 1e6,
                'num_quantizable_layers': num_quantizable,
            }
            
            # Don't cache the quantized model to save memory (can be regenerated if needed)
            # Only cache if explicitly requested via get_quantized_model
            
            return quantized_model
            
        except Exception as e:
            error_msg = f"Failed to quantize model {model_name}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _prepare_model_copy(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model for quantization."""
        return copy.deepcopy(model)
    
    def _wrap_with_quant_stubs(self, model: nn.Module) -> nn.Module:
        """Add QuantStub and DeQuantStub to model for proper quantized tensor handling.
        
        This is required to convert FP32 input tensors to quantized tensors before
        processing by quantized layers, and convert back to FP32 at the output.
        add stubs into the model's forward method
        

        """
        # Add stubs as model attributes
        model.quant = QuantStub()  # Converts FP32 -> INT8 quantized
        model.dequant = DeQuantStub()  # Converts INT8 quantized -> FP32
        
        # Store original forward method
        original_forward = model.forward
        
        # Create new forward that uses stubs
        def forward_with_stubs(self, x):
            x = self.quant(x)  # Quantize input
            x = original_forward(x)  # Process with model
            x = self.dequant(x)  # Dequantize output
            return x
        
        # Replace forward method (bind to model instance)

        model.forward = types.MethodType(forward_with_stubs, model)
        
        return model
    
    def _fuse_modules(self, model: nn.Module, model_name: str) -> nn.Module:
        """Fuse common module patterns for better quantization performance.
        
        Args:
            model: Model to fuse
            model_name: Name of the model for logging
            
        Returns:
            Model with fused modules
        """
        try:
            # Common fusion patterns - try to fuse but don't fail if not possible
            fused_count = 0
            
            # Walk through modules and try to fuse common patterns
            for name, module in model.named_modules():
                if isinstance(module, nn.Sequential):
                    # Try to identify and fuse patterns within Sequential blocks
                    children = list(module.children())
                    patterns = []
                    
                    for i in range(len(children)):
                        # Conv + BN + ReLU pattern
                        if (i + 2 < len(children) and
                            isinstance(children[i], nn.Conv2d) and
                            isinstance(children[i + 1], nn.BatchNorm2d) and
                            isinstance(children[i + 2], (nn.ReLU, nn.ReLU6))):
                            patterns.append([str(i), str(i + 1), str(i + 2)])
                            fused_count += 1
                        # Conv + BN pattern
                        elif (i + 1 < len(children) and
                              isinstance(children[i], nn.Conv2d) and
                              isinstance(children[i + 1], nn.BatchNorm2d)):
                            patterns.append([str(i), str(i + 1)])
                            fused_count += 1
                        # Conv + ReLU pattern
                        elif (i + 1 < len(children) and
                              isinstance(children[i], nn.Conv2d) and
                              isinstance(children[i + 1], (nn.ReLU, nn.ReLU6))):
                            patterns.append([str(i), str(i + 1)])
                            fused_count += 1
                    
                    # Apply fusion if patterns found
                    if patterns:
                        try:
                            fuse_modules(module, patterns, inplace=True)
                        except Exception as e:
                            logger.debug(f"Could not fuse pattern in {name}: {e}")
            
            return model
            
        except Exception as e:
            logger.warning(f"Module fusion failed for {model_name}: {e}. Continuing without fusion.")
            return model
    
    def _calibrate_with_dataloader(self, model: nn.Module, dataloader: DataLoader) -> None:
        """Run calibration using a DataLoader with real data.
        
        Args:
            model: Model to calibrate
            dataloader: DataLoader providing calibration data
        """
        logger.info(f"Starting calibration with {self._num_calibration_batches} batches...")
        
        model.eval()
        device = next(model.parameters()).device
        
        batch_count = 0
        with torch.no_grad():
            for batch_data in dataloader:
                if batch_count >= self._num_calibration_batches:
                    break
                
                # Handle different batch formats
                if isinstance(batch_data, (tuple, list)):
                    inputs = batch_data[0]  # Assume first element is input tensor
                else:
                    inputs = batch_data
                
                # Move to correct device
                inputs = inputs.to(device)
                
                try:
                    _ = model(inputs)
                    batch_count += 1
                    if batch_count % 5 == 0:
                        logger.info(f"Calibrated {batch_count}/{self._num_calibration_batches} batches")
                except Exception as e:
                    logger.warning(f"Calibration batch {batch_count} failed: {e}")
                    continue
        
        logger.info(f"Calibration complete: processed {batch_count} batches")
    
    def _default_calibration(self, model: nn.Module, num_batches: int = 10) -> None:
        """Run default calibration with random data (fallback only).
        
        Args:
            model: Model to calibrate
            num_batches: Number of calibration batches
        """
        logger.info("Running default calibration with random data...")
        
        # Try to infer input shape from the first layer
        input_shape = [1, 3, 224, 224]  # Default ImageNet shape
        
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                input_shape = [1, module.in_channels, 224, 224]
                break
            elif isinstance(module, nn.Linear):
                input_shape = [1, module.in_features]
                break
        
        # Run calibration batches
        with torch.no_grad():
            for _ in range(num_batches):
                dummy_input = torch.randn(input_shape)
                try:
                    _ = model(dummy_input)
                except Exception as e:
                    logger.warning(f"Calibration batch failed: {e}")
                    break
    
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
    
    def has_quantized_checkpoint(self, model_name: str) -> bool:
        """Check if a pre-quantized checkpoint exists for the model."""
        checkpoint_path = self._checkpoint_dir / f"{model_name}_quantized.pt"
        return checkpoint_path.exists()
    
    def get_checkpoint_path(self, model_name: str) -> Path:
        """Get the checkpoint path for a model."""
        return self._checkpoint_dir / f"{model_name}_quantized.pt"
    
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

__all__ = ["ModelQuantizer"]

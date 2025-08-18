import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

def get_layer_parameter_sizes(model: Module) -> Dict[str, np.ndarray]:
    """Get parameter sizes for each named module in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary mapping layer names to their parameter size arrays
    """
    sizes = {}
    for name, module in model.named_modules():
        if len(list(module.parameters())) > 0:  # Only include parameterized layers
            param_sizes = []
            for param in module.parameters():
                param_sizes.append(np.array(param.size()))
            sizes[name] = param_sizes
    return sizes

def get_layer_parameter_size(model: Module, bits: int = 32) -> Dict[str, int]:
    """Calculate parameter size in bits for each layer.
    
    Args:
        model: PyTorch model
        bits: Bits per parameter
        
    Returns:
        Dictionary mapping layer names to their total parameter size in bits
    """
    sizes = get_layer_parameter_sizes(model)
    layer_bits = {}
    
    for name, param_sizes in sizes.items():
        total_params = 0
        for size in param_sizes:
            total_params += np.prod(size)
        layer_bits[name] = int(total_params * bits)
        
    return layer_bits

def calculate_total_parameter_size(model: Module, bits: int = 32) -> int:
    """Calculate total model size in bits.
    
    Args:
        model: PyTorch model
        bits: Bits per parameter
        
    Returns:
        Total model size in bits
    """
    layer_bits = get_layer_parameter_size(model, bits)
    total_bits = sum(layer_bits.values())
    
    logger.debug(f"Parameter sizes by layer:")
    for name, bits in layer_bits.items():
        logger.debug(f"{name}: {bits/8/1024:.2f} KB")
    logger.debug(f"Total model size: {total_bits/8/1024/1024:.2f} MB")
    
    return total_bits

def estimate_activation_size(model: Module, input_size: Tuple[int, ...], 
                           device: torch.device = torch.device('cpu')) -> Dict[str, int]:
    """Estimate intermediate activation sizes for each layer.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (N, C, H, W)
        device: Device to run estimation on
        
    Returns:
        Dictionary mapping layer names to activation sizes in bytes
    """
    activation_sizes = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activation_sizes[name] = output.nelement() * output.element_size()
            elif isinstance(output, tuple):
                size = sum(t.nelement() * t.element_size() for t in output if isinstance(t, torch.Tensor))
                activation_sizes[name] = size
        return hook
    
    # Register hooks for all named modules
    for name, module in model.named_modules():
        if not list(module.children()):  # Only leaf modules
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Run sample input through model
    try:
        model.to(device)
        x = torch.randn(input_size).to(device)
        with torch.no_grad():
            _ = model(x)
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    logger.debug(f"Activation sizes by layer:")
    for name, size in activation_sizes.items():
        logger.debug(f"{name}: {size/1024:.2f} KB")
    
    return activation_sizes

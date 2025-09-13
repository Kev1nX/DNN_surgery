import torch
import torch.nn as nn
import time
import logging
from typing import List, Dict, Tuple, Optional, Union
import psutil
import platform
from datetime import datetime
import gRPC.protobuf.dnn_inference_pb2 as dnn_inference_pb2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LayerProfiler:
    def __init__(self):
        self.profiles = []
        
    def profile_layer(self, layer: nn.Module, input_tensor: torch.Tensor, 
                     layer_idx: int, layer_name: str) -> Dict:
        # Profile execution time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output = layer(input_tensor)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        execution_time = (end_time - start_time) * 1000  # Convert to ms
    
        
        # Calculate tensor sizes
        input_size = list(input_tensor.shape)
        output_size = list(output.shape)
        data_transfer_size = output.numel() * output.element_size()  # bytes
        
        profile = {
            'layer_idx': layer_idx,
            'layer_name': layer_name,
            'execution_time': execution_time,
            'input_size': input_size,
            'output_size': output_size,
            'data_transfer_size': data_transfer_size,
            'computation_complexity': output.numel()  # Simple complexity measure
        }
        
        self.profiles.append(profile)
        logger.info(f"Layer {layer_idx} ({layer_name}): {execution_time:.2f}ms, "
                   f"Output shape: {output.shape}")
        
        return profile
        
    def get_profiles(self) -> List[Dict]:
        """Get all collected profiles"""
        return self.profiles
        
    def clear_profiles(self):
        """Clear all collected profiles"""
        self.profiles = []

class ModelSplitter:
    """Handles splitting models between edge and cloud execution"""
    
    def __init__(self, model: nn.Module, model_name: str = "unknown"):
        self.model = model
        self.model_name = model_name
        self.split_point = 0
        
        # Extract layers if model has gen_network method
        if hasattr(model, 'gen_network'):
            self.layers = model.gen_network()
        else:
            # Fallback: use model children
            self.layers = list(model.children())
            
        logger.info(f"Model {model_name} split into {len(self.layers)} layers")
        
    def set_split_point(self, split_point: int):
        """Set where to split the model between edge and cloud
        
        Args:
            split_point: Layer index where to split (0 = all cloud, len(layers) = all edge)
        """
        if split_point < 0 or split_point > len(self.layers):
            raise ValueError(f"Split point must be between 0 and {len(self.layers)}")
        self.split_point = split_point
        logger.info(f"Split point set to layer {split_point}")
        
    def get_edge_model(self) -> Optional[nn.Module]:
        """Get the edge part of the model (layers 0 to split_point-1)"""
        if self.split_point == 0:
            return None
            
        edge_layers = self.layers[:self.split_point]
        
        class EdgeModel(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = layers
                
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
                
        return EdgeModel(edge_layers)
        
    def get_cloud_model(self) -> Optional[nn.Module]:
        """Get the cloud part of the model (layers split_point to end)"""
        if self.split_point >= len(self.layers):
            return None
            
        cloud_layers = self.layers[self.split_point:]
        
        class CloudModel(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = layers
                
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
                
        return CloudModel(cloud_layers)


class DNNSurgery:
    """Main class for distributed DNN inference with optimal splitting"""
    
    def __init__(self, model: nn.Module, model_name: str = "unknown"):
        self.model = model.eval()  # Ensure model is in eval mode
        self.model_name = model_name
        self.splitter = ModelSplitter(model, model_name)
        self.profiler = LayerProfiler()
        
        logger.info(f"Initialized DNNSurgery for model: {model_name}")
        
    def profile_model(self, input_tensor: torch.Tensor) -> List[Dict]:
        """Profile the entire model layer by layer
        
        Args:
            input_tensor: Input tensor for profiling
        Returns:
            List of layer profiles
        """
        logger.info(f"Starting model profiling with input shape: {input_tensor.shape}")
        
        self.profiler.clear_profiles()
        current_tensor = input_tensor
        
        # Profile each layer
        for idx, layer in enumerate(self.splitter.layers):
            layer_name = layer.__class__.__name__
            if hasattr(layer, '_get_name'):
                layer_name = layer._get_name()
                
            profile = self.profiler.profile_layer(
                layer, current_tensor, idx, layer_name
            )
            
            # Execute layer to get output for next layer
            with torch.no_grad():
                current_tensor = layer(current_tensor)
                
        return self.profiler.get_profiles()
        
    def create_client_profile(self, input_tensor: torch.Tensor) -> dnn_inference_pb2.ClientProfile:
        """Create a protobuf ClientProfile message
        
        Args:
            input_tensor: Input tensor used for profiling
            
        Returns:
            ClientProfile protobuf message
        """
        # Run profiling
        layer_profiles = self.profile_model(input_tensor)
        
        # Create layer profile messages
        layer_metrics = []
        for profile in layer_profiles:
            layer_metric = dnn_inference_pb2.LayerProfile(
                layer_idx=profile['layer_idx'],
                layer_name=profile['layer_name'],
                execution_time=profile['execution_time'],
                input_size=profile['input_size'],
                output_size=profile['output_size'],
                data_transfer_size=profile['data_transfer_size']
            )
            layer_metrics.append(layer_metric)
        
        # Create client profile (simplified structure)
        client_profile = dnn_inference_pb2.ClientProfile(
            model_name=self.model_name,
            layer_metrics=layer_metrics,
            input_size=list(input_tensor.shape),
            total_layers=len(layer_profiles)
        )
        
        return client_profile
        
    def find_optimal_split(self, target_edge_percentage: float = 0.5) -> int:
        """Find optimal split point based on execution time
        
        Args:
            target_edge_percentage: Target percentage of computation on edge (0.0 to 1.0)
            
        Returns:
            Optimal split point (layer index)
        """
        if not self.profiler.profiles:
            logger.warning("No profiling data available. Run profile_model first.")
            return 0
            
        total_time = sum(p['execution_time'] for p in self.profiler.profiles)
        target_time = total_time * target_edge_percentage
        
        cumulative_time = 0
        for i, profile in enumerate(self.profiler.profiles):
            cumulative_time += profile['execution_time']
            if cumulative_time >= target_time:
                return i + 1  # Split after this layer
                
        return len(self.profiler.profiles)  # All layers on edge
    
import torch
import torch.nn as nn
import time
import logging
import io
import socket
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import gRPC.protobuf.dnn_inference_pb2 as dnn_inference_pb2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkProfiler:
    """Measures network bandwidth and latency between client and server"""
    
    def __init__(self):
        self.bandwidth_mbps = None
        self.latency_ms = None
        
    def measure_network_performance(self, server_address: str, test_data_sizes: List[int] = None) -> Dict[str, float]:
        """Measure network bandwidth and latency
        
        Args:
            server_address: Server address in format "host:port"
            test_data_sizes: List of data sizes in bytes for bandwidth testing
            
        Returns:
            Dictionary with bandwidth (Mbps) and latency (ms)
        """
        if test_data_sizes is None:
            # Use various sizes to get accurate bandwidth measurement
            test_data_sizes = [1024, 4096, 16384, 65536, 262144]  # 1KB to 256KB
            
        try:
            host, port = server_address.split(':')
            port = int(port)
            
            # Measure latency with small packets
            latency_ms = self._measure_latency(host, port)
            
            # Measure bandwidth with larger data transfers
            bandwidth_mbps = self._measure_bandwidth(host, port, test_data_sizes)
            
            self.bandwidth_mbps = bandwidth_mbps
            self.latency_ms = latency_ms
            
            logger.info(f"Network performance measured - Bandwidth: {bandwidth_mbps:.2f} Mbps, Latency: {latency_ms:.2f} ms")
            
            return {
                'bandwidth_mbps': bandwidth_mbps,
                'latency_ms': latency_ms
            }
            
        except Exception as e:
            logger.error(f"Failed to measure network performance: {str(e)}")
            return {'bandwidth_mbps': 10.0, 'latency_ms': 50.0}  # Conservative fallback
    
    def _measure_latency(self, host: str, port: int, num_pings: int = 10) -> float:
        """Measure network latency using socket connections"""
        latencies = []
        
        for _ in range(num_pings):
            try:
                start_time = time.perf_counter()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # Set socket options to handle connection issues
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.settimeout(5.0)
                sock.connect((host, port))
                sock.close()
                end_time = time.perf_counter()
                
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
                
            except Exception as e:
                logger.warning(f"Latency measurement failed: {str(e)}")
                latencies.append(100.0)  # Conservative fallback
                
        return sum(latencies) / len(latencies) if latencies else 100.0
    
    def _measure_bandwidth(self, host: str, port: int, test_sizes: List[int]) -> float:
        """Measure bandwidth by sending data of various sizes"""
        bandwidth_measurements = []
        
        for size in test_sizes:
            try:
                # Create test data
                test_data = b'0' * size
                
                start_time = time.perf_counter()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # Set socket options to handle connection issues
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.settimeout(10.0)
                sock.connect((host, port))
                sock.sendall(test_data)
                sock.close()
                end_time = time.perf_counter()
                
                transfer_time = end_time - start_time
                if transfer_time > 0:
                    bandwidth_bps = (size * 8) / transfer_time  # bits per second
                    bandwidth_mbps = bandwidth_bps / (1024 * 1024)  # Convert to Mbps
                    bandwidth_measurements.append(bandwidth_mbps)
                    
            except Exception as e:
                logger.warning(f"Bandwidth measurement failed for size {size}: {str(e)}")
                continue
        
        return sum(bandwidth_measurements) / len(bandwidth_measurements) if bandwidth_measurements else 10.0
    
    def estimate_transfer_time(self, data_size_bytes: int) -> float:
        """Estimate transfer time for given data size
        
        Args:
            data_size_bytes: Size of data to transfer in bytes
            
        Returns:
            Estimated transfer time in milliseconds
        """
        if self.bandwidth_mbps is None or self.latency_ms is None:
            logger.warning("Network not profiled. Using conservative estimates.")
            return (data_size_bytes * 8) / (10 * 1024 * 1024) * 1000 + 50  # 10 Mbps + 50ms latency
            
        # Transfer time = latency + (data_size_bits / bandwidth_bps)
        data_size_bits = data_size_bytes * 8
        bandwidth_bps = self.bandwidth_mbps * 1024 * 1024
        transfer_time_ms = self.latency_ms + (data_size_bits / bandwidth_bps) * 1000
        
        return transfer_time_ms


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
    """Handles splitting models between edge and cloud execution using model's natural structure"""
    
    def __init__(self, model: nn.Module, model_name: str = "unknown"):
        self.model = model
        self.model_name = model_name
        self.split_point = 0
        
        # Use model's natural hierarchical structure instead of flattening
        self.layers = self._get_model_layers(model)
        logger.info(f"Model {model_name} has {len(self.layers)} major components")
        
        for i, layer in enumerate(self.layers):
            logger.debug(f"Component {i}: {layer.__class__.__name__}")
    
    def _get_model_layers(self, model: nn.Module) -> List[nn.Module]:
        """Get model layers using the model's natural structure"""
        # For most pretrained models, just use the direct children
        # This preserves the model's intended structure
        layers = list(model.children())
        
        # If the model has very few top-level children, we might want to go one level deeper
        if len(layers) <= 2:
            expanded_layers = []
            for layer in layers:
                if hasattr(layer, '__len__') and len(list(layer.children())) > 1:
                    # If this layer has multiple children, expand them
                    expanded_layers.extend(list(layer.children()))
                else:
                    expanded_layers.append(layer)
            layers = expanded_layers
        
        return layers
        
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
        """Get the edge part of the model using the original model's forward logic"""
        if self.split_point == 0:
            return None
            
        edge_layers = self.layers[:self.split_point]
        
        class EdgeModel(nn.Module):
            def __init__(self, layers, original_model, model_name):
                super().__init__()
                self.layers = nn.ModuleList(layers)
                self.original_model = original_model
                self.model_name = model_name
                
            def forward(self, x):
                # Execute layers sequentially, but let each layer handle its own forward logic
                for layer in self.layers:
                    x = layer(x)
                return x
                
        return EdgeModel(edge_layers, self.model, self.model_name)
        
    def get_cloud_model(self) -> Optional[nn.Module]:
        """Get the cloud part of the model using the original model's forward logic"""
        if self.split_point >= len(self.layers):
            return None
            
        cloud_layers = self.layers[self.split_point:]
        split_point = self.split_point
        
        class CloudModel(nn.Module):
            def __init__(self, layers, original_model, model_name, split_point):
                super().__init__()
                self.layers = nn.ModuleList(layers)
                self.original_model = original_model
                self.model_name = model_name
                self.split_point = split_point
                
            def forward(self, x):
                # Execute remaining layers sequentially
                for i, layer in enumerate(self.layers):
                    # Handle flattening for pretrained models
                    if self._needs_flattening(layer, x):
                        x = torch.flatten(x, 1)
                    x = layer(x)
                return x
                
            def _needs_flattening(self, layer, input_tensor):
                """Check if we need to flatten the input before applying this layer"""
                # Check if this is a Linear layer expecting 2D input but receiving 4D input
                if isinstance(layer, nn.Linear) and input_tensor.dim() > 2:
                    return True
                    
                # Check if this is a Sequential classifier that starts with Linear/Dropout
                if isinstance(layer, nn.Sequential) and len(layer) > 0:
                    first_layer = layer[0]
                    if isinstance(first_layer, (nn.Linear, nn.Dropout)) and input_tensor.dim() > 2:
                        return True
                
                return False
                
        return CloudModel(cloud_layers, self.model, self.model_name, split_point)


class DNNSurgery:
    """Main class for distributed DNN inference with optimal splitting using NeuroSurgeon approach"""
    
    def __init__(self, model: nn.Module, model_name: str = "unknown"):
        self.model = model.eval()  # Ensure model is in eval mode
        self.model_name = model_name
        self.splitter = ModelSplitter(model, model_name)
        self.profiler = LayerProfiler()
        self.network_profiler = NetworkProfiler()
        
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
    
    def find_optimal_split_neurosurgeon(self, input_tensor: torch.Tensor, server_address: str) -> Tuple[int, Dict]:
        """Find optimal split point using NeuroSurgeon approach
        
        Args:
            input_tensor: Input tensor for profiling
            server_address: Server address for network profiling
            
        Returns:
            Tuple of (optimal_split_point, timing_analysis)
        """
        logger.info("Finding optimal split point using NeuroSurgeon approach...")
        
        # Step 1: Profile the model layers
        layer_profiles = self.profile_model(input_tensor)
        
        # Step 2: Measure network performance
        network_metrics = self.network_profiler.measure_network_performance(server_address)
        
        # Step 3: Calculate total time for each possible split point
        split_analysis = {}
        min_total_time = float('inf')
        optimal_split = 0
        
        for split_point in range(len(self.splitter.layers) + 1):
            timing = self._calculate_split_timing(layer_profiles, split_point, input_tensor)
            split_analysis[split_point] = timing
            
            if timing['total_time'] < min_total_time:
                min_total_time = timing['total_time']
                optimal_split = split_point
        
        logger.info(f"Optimal split point found: {optimal_split} with total time: {min_total_time:.2f}ms")
        
        return optimal_split, {
            'optimal_split': optimal_split,
            'min_total_time': min_total_time,
            'network_metrics': network_metrics,
            'all_splits': split_analysis
        }
    
    def _calculate_split_timing(self, layer_profiles: List[Dict], split_point: int, input_tensor: torch.Tensor) -> Dict:
        """Calculate timing for a specific split point using NeuroSurgeon formula
        
        Args:
            layer_profiles: List of layer profiling data
            split_point: Split point to analyze
            input_tensor: Input tensor
            
        Returns:
            Dictionary with timing breakdown
        """
        # Client execution time (layers 0 to split_point-1)
        client_time = sum(layer_profiles[i]['execution_time'] for i in range(min(split_point, len(layer_profiles))))
        
        # Server execution time (layers split_point to end)
        server_time = sum(layer_profiles[i]['execution_time'] for i in range(split_point, len(layer_profiles)))
        
        # Calculate data transfer sizes
        if split_point == 0:
            # All on server - transfer original input
            input_transfer_size = input_tensor.numel() * input_tensor.element_size()
        elif split_point >= len(layer_profiles):
            # All on client - no intermediate transfer, but need to send final result back
            input_transfer_size = 0
        else:
            # Transfer intermediate result
            input_transfer_size = layer_profiles[split_point - 1]['data_transfer_size']
        
        # Output transfer size (final result back to client)
        if split_point >= len(layer_profiles):
            # All on client - no server output to transfer back
            output_transfer_size = 0
        else:
            # Transfer final server output back to client
            output_transfer_size = layer_profiles[-1]['data_transfer_size']
        
        # Calculate transfer times
        input_transfer_time = self.network_profiler.estimate_transfer_time(input_transfer_size)
        output_transfer_time = self.network_profiler.estimate_transfer_time(output_transfer_size)
        
        # Total time = Client Execution + Input Transfer + Server Execution + Output Transfer
        total_time = client_time + input_transfer_time + server_time + output_transfer_time
        
        return {
            'split_point': split_point,
            'client_time': client_time,
            'server_time': server_time,
            'input_transfer_time': input_transfer_time,
            'output_transfer_time': output_transfer_time,
            'input_transfer_size': input_transfer_size,
            'output_transfer_size': output_transfer_size,
            'total_time': total_time
        }
        
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
        """Find optimal split point based on execution time (legacy method)
        
        Args:
            target_edge_percentage: Target percentage of computation on edge (0.0 to 1.0)
            
        Returns:
            Optimal split point (layer index)
        """
        logger.warning("Using legacy split method. Consider using find_optimal_split_neurosurgeon() instead.")
        
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


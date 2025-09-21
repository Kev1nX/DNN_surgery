import torch
import torch.nn as nn
import time
import logging
import io
import socket
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import grpc
import gRPC.protobuf.dnn_inference_pb2 as dnn_inference_pb2
import gRPC.protobuf.dnn_inference_pb2_grpc as dnn_inference_pb2_grpc

def cuda_sync():
    """Helper function to synchronize CUDA operations if available"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkProfiler:
    """Measures network bandwidth and latency between client and server"""
    
    def __init__(self):
        self.bandwidth_mbps = None
        self.latency_ms = None
        
    def measure_network(self, server_address: str, test_data_sizes: List[int] = None) -> Dict[str, float]:
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
        cuda_sync()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output = layer(input_tensor)

        cuda_sync()
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
                # Execute layers sequentially with proper shape handling
                for i, layer in enumerate(self.layers):
                    # Handle flattening before Linear layers if needed
                    if self._needs_flattening(layer, x):
                        x = torch.flatten(x, 1)
                    x = layer(x)
                return x
                
            def _needs_flattening(self, layer, input_tensor):
                """Check if we need to flatten the input before applying this layer"""
                # Check if this is a Linear layer expecting 2D input but receiving >2D input
                if isinstance(layer, nn.Linear) and input_tensor.dim() > 2:
                    return True
                    
                # Check if this is a Sequential that starts with Linear/Dropout
                if isinstance(layer, nn.Sequential) and len(layer) > 0:
                    first_layer = layer[0]
                    if isinstance(first_layer, (nn.Linear, nn.Dropout)) and input_tensor.dim() > 2:
                        return True
                
                return False
                
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
                # Execute remaining layers sequentially with proper shape handling
                for i, layer in enumerate(self.layers):
                    # Handle flattening for pretrained models
                    if self._needs_flattening(layer, x):
                        original_shape = x.shape
                        x = torch.flatten(x, 1)
                        logger.debug(f"Flattened tensor from {original_shape} to {x.shape} before {layer.__class__.__name__}")
                    x = layer(x)
                return x
                
            def _needs_flattening(self, layer, input_tensor):
                """Check if we need to flatten the input before applying this layer"""
                # Check if this is a Linear layer expecting 2D input but receiving >2D input
                if isinstance(layer, nn.Linear) and input_tensor.dim() > 2:
                    return True
                    
                # Check if this is a Sequential classifier that starts with Linear/Dropout
                if isinstance(layer, nn.Sequential) and len(layer) > 0:
                    first_layer = layer[0]
                    if isinstance(first_layer, (nn.Linear, nn.Dropout)) and input_tensor.dim() > 2:
                        return True
                
                # Check for adaptive pooling followed by flatten
                # Some models have this pattern: AdaptiveAvgPool2d -> flatten -> Linear
                if hasattr(layer, '__class__') and 'Linear' in str(layer.__class__):
                    if input_tensor.dim() > 2:
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
                
            # Before profiling, handle tensor shape transitions
            profile_input = current_tensor
            if isinstance(layer, nn.Linear) and current_tensor.dim() > 2:
                # For Linear layers, flatten the input for profiling
                profile_input = torch.flatten(current_tensor, 1)
                
            profile = self.profiler.profile_layer(
                layer, profile_input, idx, layer_name
            )
            
            # Execute layer with the actual input (including potential flattening logic)
            with torch.no_grad():
                if isinstance(layer, nn.Linear) and current_tensor.dim() > 2:
                    # Apply flattening for the actual execution too
                    current_tensor = torch.flatten(current_tensor, 1)
                    
                current_tensor = layer(current_tensor)
                
        return self.profiler.get_profiles()
    
    def find_optimal_split(self, input_tensor: torch.Tensor, server_address: str) -> Tuple[int, Dict]:
        """Find optimal split point using NeuroSurgeon approach with server-side profiling
        
        Args:
            input_tensor: Input tensor for profiling
            server_address: Server address for network profiling
            
        Returns:
            Tuple of (optimal_split_point, timing_analysis)
        """
        logger.info("Finding optimal split point using NeuroSurgeon approach...")
        
        # Step 1: Profile the model layers on CLIENT
        logger.info("Step 1: Profiling model layers on client...")
        client_layer_profiles = self.profile_model(input_tensor)
        
        # Step 2: Get SERVER-SIDE profiling results via gRPC
        logger.info("Step 2: Requesting server-side profiling...")
        server_layer_profiles = self._get_server_profile(input_tensor, server_address)
        
        if server_layer_profiles is None:
            logger.warning("Failed to get server profiling. Falling back to client-only estimation.")
            server_layer_profiles = client_layer_profiles  # Fallback
        
        # Step 3: Measure network performance
        logger.info("Step 3: Measuring network performance...")
        network_metrics = self.network_profiler.measure_network(server_address)
        
        # Step 4: Calculate total time for each possible split point using ACTUAL server timings
        logger.info("Step 4: Analyzing split points with actual client and server performance...")
        split_analysis = {}
        min_total_time = float('inf')
        optimal_split = 0
        
        logger.info("=== NeuroSurgeon Split Point Analysis (Client vs Server Performance) ===")
        logger.info(f"{'Split':<5} {'Client':<8} {'Server':<8} {'Transfer':<10} {'Total':<8} {'Description'}")
        logger.info("-" * 70)
        
        for split_point in range(len(self.splitter.layers) + 1):
            timing = self._calculate_split_timing(
                client_layer_profiles, server_layer_profiles, split_point, input_tensor
            )
            split_analysis[split_point] = timing
            
            # Create description
            if split_point == 0:
                desc = "All cloud"
            elif split_point >= len(self.splitter.layers):
                desc = "All client"
            else:
                desc = f"Split after layer {split_point-1}"
            
            total_transfer = timing['input_transfer_time'] + timing['output_transfer_time']
            
            logger.info(f"{split_point:<5} {timing['client_time']:<8.1f} {timing['server_time']:<8.1f} "
                       f"{total_transfer:<10.1f} {timing['total_time']:<8.1f} {desc}")
            
            if timing['total_time'] < min_total_time:
                min_total_time = timing['total_time']
                optimal_split = split_point
        
        logger.info("-" * 70)
        logger.info(f"Optimal split point: {optimal_split} with total time: {min_total_time:.2f}ms")
        
        # Log detailed breakdown of optimal split
        optimal_timing = split_analysis[optimal_split]
        logger.info(f"Optimal split breakdown:")
        logger.info(f"  Client time: {optimal_timing['client_time']:.2f}ms (measured on client)")
        logger.info(f"  Server time: {optimal_timing['server_time']:.2f}ms (measured on server)")
        logger.info(f"  Input transfer: {optimal_timing['input_transfer_time']:.2f}ms ({optimal_timing['input_transfer_size']} bytes)")
        logger.info(f"  Output transfer: {optimal_timing['output_transfer_time']:.2f}ms ({optimal_timing['output_transfer_size']} bytes)")
        logger.info("=" * 50)
        
        return optimal_split, {
            'optimal_split': optimal_split,
            'min_total_time': min_total_time,
            'network_metrics': network_metrics,
            'client_profiles': client_layer_profiles,
            'server_profiles': server_layer_profiles,
            'all_splits': split_analysis
        }
    
    def _get_server_profile(self, input_tensor: torch.Tensor, server_address: str) -> Optional[List[Dict]]:
        """Get server-side profiling results via gRPC
        
        Args:
            input_tensor: Input tensor for profiling
            server_address: Server address
            
        Returns:
            List of server-side layer profiles or None if failed
        """
        try:
            # Create gRPC channel
            channel = grpc.insecure_channel(server_address)
            stub = dnn_inference_pb2_grpc.DNNInferenceStub(channel)
            
            # Create profiling request
            client_profile = self.create_client_profile(input_tensor)
            request = dnn_inference_pb2.ProfilingRequest(
                profile=client_profile,
                client_id="profiling_client"
            )
            
            logger.info("Sending profiling request to server...")
            response = stub.SendProfilingData(request)
            
            if response.success:
                logger.info(f"Server profiling completed: {response.message}")
                
                # Parse server profiling data from response
                server_profiles = self._parse_server_response(response.updated_split_config)
                if server_profiles:
                    logger.info("Successfully received server profiling data")
                    return server_profiles
                else:
                    logger.warning("Failed to parse server profiling data")
                    return None
            else:
                logger.error(f"Server profiling failed: {response.message}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get server profiling: {str(e)}")
            return None
    
    def _parse_server_response(self, config_string: str) -> Optional[List[Dict]]:
        """Parse server profiling data from the response string
        
        Args:
            config_string: Configuration string from server response
            
        Returns:
            List of server-side layer profiles or None if parsing failed
        """
        try:
            # Parse the config string: "split_point:X;server_profile:time1,time2,time3..."
            parts = config_string.split(';')
            server_profile_part = None
            
            for part in parts:
                if part.startswith('server_profile:'):
                    server_profile_part = part.split(':', 1)[1]
                    break
            
            if not server_profile_part:
                logger.warning("No server profile data found in response")
                return None
            
            # Parse the comma-separated execution times
            server_times = [float(t.strip()) for t in server_profile_part.split(',')]
            
            # Convert to the same format as client profiles
            server_profiles = []
            for i, execution_time in enumerate(server_times):
                profile = {
                    'layer_idx': i,
                    'layer_name': f'Layer_{i}',  # We don't have layer names from server
                    'execution_time': execution_time,
                    'input_size': [],  # Not needed for timing calculation
                    'output_size': [],  # Not needed for timing calculation
                    'data_transfer_size': 0,  # Will be filled from client profiles
                    'computation_complexity': 0  # Not needed
                }
                server_profiles.append(profile)
            
            logger.info(f"Parsed {len(server_profiles)} server layer profiles")
            return server_profiles
            
        except Exception as e:
            logger.error(f"Failed to parse server profile response: {str(e)}")
            return None
    
    def _calculate_split_timing(self, client_profiles: List[Dict], 
                               server_profiles: List[Dict], 
                               split_point: int, 
                               input_tensor: torch.Tensor) -> Dict:
        """Calculate timing for a specific split point using actual client and server profiles
        
        Args:
            client_profiles: List of client-side layer profiling data
            server_profiles: List of server-side layer profiling data
            split_point: Split point to analyze
            input_tensor: Input tensor
            
        Returns:
            Dictionary with timing breakdown
        """
        # Client execution time (layers 0 to split_point-1) using CLIENT measurements
        client_time = sum(client_profiles[i]['execution_time'] for i in range(min(split_point, len(client_profiles))))
        
        # Server execution time (layers split_point to end) using SERVER measurements
        server_time = sum(server_profiles[i]['execution_time'] for i in range(split_point, len(server_profiles)))
        
        # Calculate data transfer sizes (same as before)
        if split_point == 0:
            # All on server - transfer original input
            input_transfer_size = input_tensor.numel() * input_tensor.element_size()
        elif split_point >= len(client_profiles):
            # All on client - no intermediate transfer, but need to send final result back
            input_transfer_size = 0
        else:
            # Transfer intermediate result - use client profile since that's where data comes from
            input_transfer_size = client_profiles[split_point - 1]['data_transfer_size']
        
        # Output transfer size (final result back to client) - use server profile for final output size
        if split_point >= len(server_profiles):
            # All on client - no server output to transfer back
            output_transfer_size = 0
        else:
            # Transfer final server output back to client
            output_transfer_size = server_profiles[-1]['data_transfer_size']
        
        # Calculate transfer times
        input_transfer_time = self.network_profiler.estimate_transfer_time(input_transfer_size)
        output_transfer_time = self.network_profiler.estimate_transfer_time(output_transfer_size)
        
        # Total time = Client Execution + Input Transfer + Server Execution + Output Transfer
        total_time = client_time + input_transfer_time + server_time + output_transfer_time
        
        # Log detailed calculation for debugging
        logger.debug(f"Split {split_point} timing calculation (with server profiles):")
        logger.debug(f"  Client layers: 0 to {split_point-1 if split_point > 0 else 'none'}")
        logger.debug(f"  Server layers: {split_point} to {len(server_profiles)-1 if split_point < len(server_profiles) else 'none'}")
        logger.debug(f"  Client time: {client_time:.2f}ms (measured on client)")
        logger.debug(f"  Server time: {server_time:.2f}ms (measured on server)")
        logger.debug(f"  Input transfer: {input_transfer_size} bytes -> {input_transfer_time:.2f}ms")
        logger.debug(f"  Output transfer: {output_transfer_size} bytes -> {output_transfer_time:.2f}ms")
        logger.debug(f"  Total time: {total_time:.2f}ms")
        
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


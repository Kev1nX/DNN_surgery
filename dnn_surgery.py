import torch
import torch.nn as nn
import time
import logging
from typing import List, Dict, Tuple, Optional
import uuid
import grpc
import gRPC.protobuf.dnn_inference_pb2 as dnn_inference_pb2
import gRPC.protobuf.dnn_inference_pb2_grpc as dnn_inference_pb2_grpc
from config import GRPC_SETTINGS
from visualization import build_split_timing_summary, format_split_summary

def cuda_sync():
    """Helper function to synchronize CUDA operations if available"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkProfiler:
    """Measures actual tensor transfer time between client and server."""

    def __init__(self):
        self.probe_client_id = f"bandwidth_probe_{uuid.uuid4()}"
        self._transfer_cache: Dict[Tuple[str, int], float] = {}

    def clear_cache(self) -> None:
        """Reset cached transfer measurements."""
        self._transfer_cache.clear()

    def measure_tensor_transfer(self, tensor: torch.Tensor, server_address: str, echo: bool = False) -> float:
        """Send the given tensor to the server and measure total round-trip time.

        Args:
            tensor: Tensor to transfer.
            server_address: Server address ("host:port").
            echo: Whether the server should echo the tensor back.

        Returns:
            Observed transfer time in milliseconds.
        """
        payload_size = tensor.numel() * tensor.element_size()
        if payload_size <= 0:
            logger.debug("Tensor payload size is zero; returning 0 transfer time")
            return 0.0

        cache_key = (server_address, payload_size)
        if cache_key in self._transfer_cache:
            return self._transfer_cache[cache_key]

        channel = None
        try:
            channel = grpc.insecure_channel(server_address, options=GRPC_SETTINGS.channel_options)
            stub = dnn_inference_pb2_grpc.DNNInferenceStub(channel)

            payload_bytes = bytes(payload_size)
            start_time = time.perf_counter()
            response = stub.MeasureBandwidth(
                dnn_inference_pb2.BandwidthProbeRequest(
                    client_id=self.probe_client_id,
                    payload=payload_bytes,
                    echo=echo
                ),
                timeout=15.0
            )
            end_time = time.perf_counter()

            if not response.success:
                logger.warning(f"Transfer probe failed: {response.message}")
                return 0.0

            elapsed_ms = (end_time - start_time) * 1000
            self._transfer_cache[cache_key] = elapsed_ms
            return elapsed_ms
        except Exception as exc:
            logger.error(f"Failed to measure tensor transfer: {exc}")
            return 0.0
        finally:
            if channel is not None:
                try:
                    channel.close()
                except Exception:
                    pass

    def get_cached_measurements(self) -> Dict[Tuple[str, int], float]:
        """Return a copy of the cached transfer measurements keyed by (server, payload)."""
        return dict(self._transfer_cache)


class LayerProfiler:
    def __init__(self):
        self.profiles = []
        
    def profile_layer(self, layer: nn.Module, input_tensor: torch.Tensor, 
                     layer_idx: int, layer_name: str) -> Dict:
        # Profile execution time with multiple runs for accuracy
        times = []
        for _ in range(3):  # Multiple measurements for accuracy
            cuda_sync()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = layer(input_tensor)

            cuda_sync()
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Use median time to avoid outliers
        execution_time = sorted(times)[len(times)//2]
    
        
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
                    
                # Check if this is a Sequential classifier (like AlexNet's classifier)
                # The classifier Sequential may contain Dropout, Linear, etc.
                if isinstance(layer, nn.Sequential) and len(layer) > 0 and input_tensor.dim() > 2:
                    # Check if any layer in the sequential is Linear - if so, we need to flatten
                    for sublayer in layer:
                        if isinstance(sublayer, nn.Linear):
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
                    
                # Check if this is a Sequential classifier (like AlexNet's classifier)
                # The classifier Sequential may contain Dropout, Linear, etc.
                if isinstance(layer, nn.Sequential) and len(layer) > 0 and input_tensor.dim() > 2:
                    # Check if any layer in the sequential is Linear - if so, we need to flatten
                    for sublayer in layer:
                        if isinstance(sublayer, nn.Linear):
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
        self.layer_outputs: List[torch.Tensor] = []
        self.input_sample: Optional[torch.Tensor] = None
        
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
        self.layer_outputs = []
        self.input_sample = input_tensor.detach().cpu()
        current_tensor = input_tensor
        
        # Profile each layer
        for idx, layer in enumerate(self.splitter.layers):
            layer_name = layer.__class__.__name__
            if hasattr(layer, '_get_name'):
                layer_name = layer._get_name()
                
            # Before profiling, handle tensor shape transitions
            profile_input = current_tensor
            needs_flatten = self._needs_flattening_for_layer(layer, current_tensor)
            
            if needs_flatten:
                # For layers that need flattening, flatten the input for profiling
                profile_input = torch.flatten(current_tensor, 1)

            self.profiler.profile_layer(
                layer, profile_input, idx, layer_name
            )
            
            # Execute layer with the actual input (including potential flattening logic)
            with torch.no_grad():
                if needs_flatten:
                    # Apply flattening for the actual execution too
                    current_tensor = torch.flatten(current_tensor, 1)
                    
                current_tensor = layer(current_tensor)
                self.layer_outputs.append(current_tensor.detach().cpu())
                
        return self.profiler.get_profiles()
    
    def _needs_flattening_for_layer(self, layer: nn.Module, input_tensor: torch.Tensor) -> bool:
        """Check if we need to flatten the input before applying this layer
        
        Args:
            layer: The layer to check
            input_tensor: The input tensor
            
        Returns:
            True if flattening is needed, False otherwise
        """
        # Check if this is a Linear layer expecting 2D input but receiving >2D input
        if isinstance(layer, nn.Linear) and input_tensor.dim() > 2:
            return True
            
        # Check if this is a Sequential classifier (like AlexNet's classifier)
        # The classifier Sequential may contain Dropout, Linear, etc.
        if isinstance(layer, nn.Sequential) and len(layer) > 0 and input_tensor.dim() > 2:
            # Check if any layer in the sequential is Linear - if so, we need to flatten
            for sublayer in layer:
                if isinstance(sublayer, nn.Linear):
                    return True
        
        return False
    
    def find_optimal_split(self, input_tensor: torch.Tensor, server_address: str) -> Tuple[int, Dict]:
        """Find optimal split point using client-side calculation with server execution times
        
        Args:
            input_tensor: Input tensor for profiling
            server_address: Server address for network profiling
            
        Returns:
            Tuple of (optimal_split_point, timing_analysis)
        """
        logger.info("=== CLIENT-DECIDES ARCHITECTURE ===")
        logger.info("Step 1: Profiling model layers on client...")
        
        # Step 1: Profile the model layers on CLIENT
        client_layer_profiles = self.profile_model(input_tensor)
        
        # Step 2: Get SERVER-SIDE execution times via gRPC (no split recommendation)
        logger.info("Step 2: Requesting server-side execution times...")
        server_layer_profiles = self._get_server_profile(input_tensor, client_layer_profiles, server_address)
        
        if server_layer_profiles is None:
            logger.warning("Failed to get server execution times. Using client times as fallback.")
            server_layer_profiles = client_layer_profiles
        
        # Step 3: CLIENT calculates optimal split point using direct transfer measurements
        logger.info("Step 3: Client calculating optimal split point with live transfer probes...")
        optimal_split, split_analysis = self._calculate_optimal_split_client_side(
            client_layer_profiles, server_layer_profiles, input_tensor, server_address
        )
        
        # Step 4: Send optimal split decision back to server
        logger.info(f"Step 4: Sending split decision to server: split_point={optimal_split}")
        split_config_success = self._send_split_decision_to_server(optimal_split, server_address)
        
        if not split_config_success:
            logger.error("Failed to configure server with split decision!")
        else:
            logger.info("Server successfully configured with client's split decision")
        
        split_summary = build_split_timing_summary(split_analysis)

        return optimal_split, {
            'optimal_split': optimal_split,
            'min_total_time': split_analysis[optimal_split]['total_time'],
            'transfer_measurements_ms': self.network_profiler.get_cached_measurements(),
            'client_profiles': client_layer_profiles,
            'server_profiles': server_layer_profiles,
            'all_splits': split_analysis,
            'split_summary': split_summary,
            'split_summary_table': format_split_summary(split_summary, sort_by_total_time=False),
            'recommended_by_server': False,
            'split_config_success': split_config_success
        }

    def _get_server_profile(self, input_tensor: torch.Tensor, profile, server_address: str) -> Optional[List[Dict]]:
        """Get server-side execution times via gRPC (NO split recommendation)
        
        Args:
            input_tensor: Input tensor for profiling
            server_address: Server address
            
        Returns:
            List of server layer profiles, or None if failed
        """
        try:
            # Create gRPC channel with larger message size limits
            channel = grpc.insecure_channel(
                server_address, options=GRPC_SETTINGS.channel_options
            )
            stub = dnn_inference_pb2_grpc.DNNInferenceStub(channel)
            
            # Create profiling request
            client_profile = self.create_client_profile(input_tensor, profile)
            request = dnn_inference_pb2.ProfilingRequest(
                profile=client_profile,
                client_id="profiling_client"
            )
            
            logger.info("Sending profiling request to server...")
            response = stub.SendProfilingData(request)
            
            if response.success:
                logger.info("Server profiling completed successfully")
                
                # Parse server execution times from response
                server_profiles = self._parse_server_response(response.updated_split_config)
                if server_profiles:
                    logger.info(f"Successfully received {len(server_profiles)} server execution times")
                    return server_profiles
                else:
                    logger.warning("Failed to parse server execution times")
                    return None
            else:
                logger.error(f"Server profiling failed: {response.message}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get server profiling: {str(e)}")
            return None

    def _calculate_optimal_split_client_side(self, client_profiles: List[Dict], 
                                           server_profiles: List[Dict], 
                                           input_tensor: torch.Tensor,
                                           server_address: str) -> Tuple[int, Dict]:
        """Calculate optimal split point on client side using server execution times
        
        Args:
            client_profiles: Client-side layer profiling data
            server_profiles: Server-side layer profiling data  
            input_tensor: Input tensor
            server_address: Server address for transfer probes
            
        Returns:
            Tuple of (optimal_split_point, all_split_analysis)
        """
        logger.info("=== Client-Side Split Point Analysis ===")
        logger.info("Transfer probes will use real tensors captured during profiling")
        logger.info(f"{'Split':<5} {'Client':<8} {'Server':<8} {'Transfer':<10} {'Total':<8} {'Description'}")
        logger.info("-" * 75)
        
        split_analysis = {}
        min_total_time = float('inf')
        optimal_split = 0
        
        # Ensure server profiles have the right data transfer sizes from client profiles
        for i, server_profile in enumerate(server_profiles):
            if i < len(client_profiles):
                server_profile['data_transfer_size'] = client_profiles[i]['data_transfer_size']
        
        for split_point in range(len(self.splitter.layers) + 1):
            timing = self._calculate_split_timing(
                client_profiles, server_profiles, split_point, input_tensor, server_address
            )
            split_analysis[split_point] = timing
            
            desc = f"Split after layer {split_point}"
            
            total_transfer = timing['input_transfer_time'] + timing['output_transfer_time']
            
            logger.info(f"{split_point:<5} {timing['client_time']:<8.1f} {timing['server_time']:<8.1f} "
                       f"{total_transfer:<10.1f} {timing['total_time']:<8.1f} {desc}")
            
            if timing['total_time'] < min_total_time:
                min_total_time = timing['total_time']
                optimal_split = split_point
        
        logger.info("-" * 75)
        logger.info(f"OPTIMAL SPLIT: {optimal_split} with total time: {min_total_time:.2f}ms")
        
        return optimal_split, split_analysis

    def _send_split_decision_to_server(self, split_point: int, server_address: str) -> bool:
        """Send the client's split decision to the server
        
        Args:
            split_point: The optimal split point decided by client
            server_address: Server address
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create gRPC channel with larger message size limits
            channel = grpc.insecure_channel(
                server_address, options=GRPC_SETTINGS.channel_options
            )
            stub = dnn_inference_pb2_grpc.DNNInferenceStub(channel)
            
            # Create split configuration request
            request = dnn_inference_pb2.SplitConfigRequest(
                model_name=self.model_name,
                split_point=split_point
            )
            
            logger.info(f"Sending split decision to server: {self.model_name} -> split_point={split_point}")
            response = stub.set_split_point(request)
            
            if response.success:
                logger.info(f"Split configuration sent to server: {response.message}")
                return True
            else:
                logger.error(f"Server rejected split configuration: {response.message}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send split decision to server: {str(e)}")
            return False
    
    def _parse_server_response(self, config_string: str) -> Optional[List[Dict]]:
        """Parse server profiling data from the response string (NO split point recommendation)
        
        Args:
            config_string: Configuration string from server response
            
        Returns:
            List of server layer profiles, or None if parsing failed
        """
        try:
            # Parse the config string: "server_profile:time1,time2,time3..."
            if not config_string.startswith('server_profile:'):
                logger.warning("Invalid server response format - expected 'server_profile:' prefix")
                return None
            
            server_profile_part = config_string.split(':', 1)[1]
            
            if not server_profile_part:
                logger.warning("No server profile data found in response")
                return None
            
            # Parse the comma-separated execution times
            server_times = [float(t.strip()) for t in server_profile_part.split(',')]
            logger.info(f"Parsed server execution times: {server_times}")
            
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
                               input_tensor: torch.Tensor,
                               server_address: str) -> Dict:
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
        
        # Measure transfer times using actual tensors
        input_transfer_time = 0.0
        output_transfer_time = 0.0

        input_tensor_sample: Optional[torch.Tensor] = None
        if split_point == 0:
            input_tensor_sample = self.input_sample
        elif 0 < split_point < len(self.layer_outputs):
            input_tensor_sample = self.layer_outputs[split_point - 1]

        if input_tensor_sample is not None:
            input_transfer_time = self.network_profiler.measure_tensor_transfer(
                input_tensor_sample, server_address
            )

        if split_point < len(self.layer_outputs) and self.layer_outputs:
            output_tensor_sample = self.layer_outputs[-1]
            output_transfer_time = self.network_profiler.measure_tensor_transfer(
                output_tensor_sample, server_address
            )
        
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
        
    def create_client_profile(self, input_tensor, profile) -> dnn_inference_pb2.ClientProfile:
        """Create a protobuf ClientProfile message
        
        Args:
            input_tensor: Input tensor used for profiling
            
        Returns:
            ClientProfile protobuf message
        """
        # Run profiling
        layer_profiles = profile
        
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


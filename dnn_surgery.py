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
from quantization import ModelQuantizer
import warnings
warnings.filterwarnings('ignore')
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
        """Get model layers with intelligent grouping to respect skip connections
        
        This method extracts layers while keeping residual blocks and other
        skip-connection structures together as atomic units.
        """
        layers = []
        
        def _has_skip_connection(module: nn.Module) -> bool:
            """Detect if a module has skip connections by analyzing its structure
            
            A module likely has skip connections if:
            1. It's not a Sequential (which is purely linear)
            2. It has multiple child modules (suggesting parallel paths)
            3. It's not a simple wrapper (like ModuleList, ModuleDict)
            """
            # Sequential containers are always linear, no skip connections
            if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                return False
            
            # Get child modules
            children = list(module.children())
            
            # If it has 2+ children and isn't just a container, it likely has skip connections
            # This catches ResNet blocks, attention modules, etc.
            if len(children) >= 2:
                return True
                
            return False
        
        def _is_atomic_layer(module: nn.Module) -> bool:
            """Check if a module should be treated as a single atomic unit"""
            # Basic layers that shouldn't be split further
            atomic_types = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, 
                          nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
                          nn.Dropout, nn.Flatten, nn.LayerNorm, nn.GELU,
                          nn.BatchNorm1d, nn.ReLU6, nn.Sigmoid, nn.Tanh)
            
            if isinstance(module, atomic_types):
                return True
            
            # Modules with skip connections are atomic
            if _has_skip_connection(module):
                return True
                
            return False
        
        def _flatten_sequential(seq: nn.Sequential) -> List[nn.Module]:
            """Flatten a Sequential container, respecting skip connections"""
            result = []
            for child in seq.children():
                if _is_atomic_layer(child):
                    # Keep as single unit
                    result.append(child)
                elif isinstance(child, nn.Sequential):
                    # Recursively flatten nested Sequential
                    result.extend(_flatten_sequential(child))
                else:
                    # Complex module - keep as single unit
                    result.append(child)
            return result
        
        # Process top-level children
        for child in model.children():
            if _is_atomic_layer(child):
                # Keep as single unit (e.g., ResNet blocks, basic layers)
                layers.append(child)
            elif isinstance(child, nn.Sequential):
                # Flatten Sequential containers while respecting skip connections
                flattened = _flatten_sequential(child)
                layers.extend(flattened)
            else:
                # Complex module - keep as single unit
                layers.append(child)
        
        return layers

    def _is_linear_like(self, module: nn.Module) -> bool:
        """Return True for modules that behave like Linear layers.

        This is robust to dynamic/quantized replacements where the runtime
        class may not be exactly ``nn.Linear`` but still needs 2D inputs
        (e.g. quantized dynamic Linear modules). We use the class name as a
        heuristic because quantized/dynamic Linear classes usually include
        'Linear' in their class name.
        """
        try:
            name = module.__class__.__name__.lower()
        except Exception:
            return False

        if 'linear' in name:
            return True

        # Fallback to isinstance check for regular nn.Linear
        if isinstance(module, nn.Linear):
            return True

        return False
        
    def set_split_point(self, split_point: int):
        """Set where to split the model between edge and cloud
        
        Args:
            split_point: Layer index where to split (0 = all cloud, len(layers) = all edge)
        """
        if split_point < 0 or split_point > len(self.layers):
            raise ValueError(f"Split point must be between 0 and {len(self.layers)}")
        self.split_point = split_point
        logger.info(f"Split point set to layer {split_point}")
        
    def get_edge_model(self, quantize: bool = False, quantizer: Optional[ModelQuantizer] = None) -> Optional[nn.Module]:
        """Get the edge part of the model using the original model's forward logic
        
        Args:
            quantize: Whether to apply quantization to the edge model
            quantizer: ModelQuantizer instance (required if quantize=True)
            
        Returns:
            Edge model (optionally quantized)
        """
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
            
            def _is_linear_like(self, module: nn.Module) -> bool:
                """Return True for modules that behave like Linear layers."""
                try:
                    name = module.__class__.__name__.lower()
                except Exception:
                    return False
                if 'linear' in name:
                    return True
                if isinstance(module, nn.Linear):
                    return True
                return False
                
            def _needs_flattening(self, layer, input_tensor):
                """Check if we need to flatten the input before applying this layer"""
                # Check if this is a Linear-like layer expecting 2D input but receiving >2D input
                # Use the outer ModelSplitter's heuristic to detect quantized/dynamic Linear modules
                if self._is_linear_like(layer) and input_tensor.dim() > 2:
                    return True
                    
                # Check if this is a Sequential classifier (like AlexNet's classifier)
                # The classifier Sequential may contain Dropout, Linear, etc.
                if isinstance(layer, nn.Sequential) and len(layer) > 0 and input_tensor.dim() > 2:
                    # First check if the Sequential already has a Flatten layer - if so, don't flatten
                    for sublayer in layer:
                        if isinstance(sublayer, nn.Flatten):
                            return False  # Sequential will handle flattening itself
                    
                    # Check if any layer in the sequential is Linear - if so, we need to flatten
                    for sublayer in layer:
                        # Check for linear-like sublayers (handles quantized/dynamic Linear)
                        if self._is_linear_like(sublayer):
                            return True
                
                return False
                
        edge_model = EdgeModel(edge_layers, self.model, self.model_name)
        
        # Apply quantization if requested
        if quantize:
            if quantizer is None:
                logger.warning("Quantization requested but no quantizer provided. Returning non-quantized edge model.")
            else:
                edge_model = quantizer.quantize_edge_model(edge_model, self.model_name)
        
        return edge_model
        
    def get_cloud_model(self, quantize: bool = False, quantizer: Optional[ModelQuantizer] = None) -> Optional[nn.Module]:
        """Get the cloud part of the model using the original model's forward logic
        
        Args:
            quantize: Whether to apply quantization to the cloud model
            quantizer: ModelQuantizer instance (required if quantize=True)
            
        Returns:
            Cloud model (optionally quantized)
        """
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
            
            def _is_linear_like(self, module: nn.Module) -> bool:
                """Return True for modules that behave like Linear layers."""
                try:
                    name = module.__class__.__name__.lower()
                except Exception:
                    return False
                if 'linear' in name:
                    return True
                if isinstance(module, nn.Linear):
                    return True
                return False
                
            def _needs_flattening(self, layer, input_tensor):
                """Check if we need to flatten the input before applying this layer"""
                # Check if this is a Linear-like layer expecting 2D input but receiving >2D input
                if self._is_linear_like(layer) and input_tensor.dim() > 2:
                    return True
                    
                # Check if this is a Sequential classifier (like AlexNet's classifier)
                # The classifier Sequential may contain Dropout, Linear, etc.
                if isinstance(layer, nn.Sequential) and len(layer) > 0 and input_tensor.dim() > 2:
                    # First check if the Sequential already has a Flatten layer - if so, don't flatten
                    for sublayer in layer:
                        if isinstance(sublayer, nn.Flatten):
                            return False  # Sequential will handle flattening itself
                    
                    # Check if any layer in the sequential is Linear - if so, we need to flatten
                    for sublayer in layer:
                        if self._is_linear_like(sublayer):
                            return True
                
                return False
                
        cloud_model = CloudModel(cloud_layers, self.model, self.model_name, split_point)
        
        # Apply quantization if requested
        if quantize:
            if quantizer is None:
                logger.warning("Quantization requested but no quantizer provided. Returning non-quantized cloud model.")
            else:
                cloud_model = quantizer.quantize_cloud_model(cloud_model, self.model_name)
        
        return cloud_model


class DNNSurgery:
    """Main class for distributed DNN inference with optimal splitting using NeuroSurgeon approach"""
    
    def __init__(self, model: nn.Module, model_name: str = "unknown", enable_quantization: bool = False):
        self.model = model.eval()  # Ensure model is in eval mode
        self.model_name = model_name
        self.splitter = ModelSplitter(model, model_name)
        self.network_profiler = NetworkProfiler()
        self.enable_quantization = enable_quantization
        self.quantizer = ModelQuantizer() if enable_quantization else None
        
        if enable_quantization:
            logger.info(f"Initialized DNNSurgery for model: {model_name} with quantization enabled")
        else:
            logger.info(f"Initialized DNNSurgery for model: {model_name}")
    
    def get_split_layer_names(self) -> List[str]:
        """Get layer names for all possible split points
        
        Returns:
            List of layer names where index i corresponds to split point i
            Split point 0 = "Input" (all on cloud)
            Split point i = layer name after which the split occurs
            Split point len(layers) = "Output" (all on edge)
        """
        layer_names = ["Input"]  # Split point 0
        
        for layer in self.splitter.layers:
            layer_name = layer.__class__.__name__
            if hasattr(layer, '_get_name'):
                layer_name = layer._get_name()
            layer_names.append(layer_name)
        
        layer_names.append("Output")  # Split point after last layer
        return layer_names
    
    def find_optimal_split(self, input_tensor: torch.Tensor, server_address: str) -> Tuple[int, Dict]:
        """Find optimal split point by measuring actual inference at each split point
        
        Args:
            input_tensor: Input tensor for inference
            server_address: Server address
            
        Returns:
            Tuple of (optimal_split_point, timing_analysis)
        """
        # Import here to avoid circular dependency
        from dnn_inference_client import DNNInferenceClient
        
        num_splits = len(self.splitter.layers) + 1
        logger.info(f"=== Finding Optimal Split Point ===")
        logger.info(f"Testing {num_splits} split points (0 to {num_splits - 1})")
        
        split_analysis = {}
        
        # Test each split point by running actual inference
        for split_point in range(num_splits):
            logger.info(f"Testing split point {split_point}/{num_splits - 1}...")
            
            # Configure split point
            self.splitter.set_split_point(split_point)
            
            # Get edge model
            edge_model = self.splitter.get_edge_model(
                quantize=self.enable_quantization,
                quantizer=self.quantizer
            )
            
            # Create client
            client = DNNInferenceClient(server_address, edge_model)
            
            # Check if cloud processing is needed
            requires_cloud = self.splitter.get_cloud_model() is not None
            
            # Configure server with this split point
            self._send_split_decision_to_server(split_point, server_address)
            
            # Run inference and measure
            try:
                _, timings = client.process_tensor(input_tensor, self.model_name, requires_cloud)
                
                edge_time = timings.get('edge_time', 0.0)
                transfer_time = timings.get('transfer_time', 0.0)
                cloud_time = timings.get('cloud_time', 0.0)
                total_time = edge_time + transfer_time + cloud_time
                
                split_analysis[split_point] = {
                    'edge_time': edge_time,
                    'transfer_time': transfer_time,
                    'cloud_time': cloud_time,
                    'total_time': total_time
                }
                
                logger.info(f"  Split {split_point}: Total={total_time:.1f}ms (Edge={edge_time:.1f}ms, "
                          f"Transfer={transfer_time:.1f}ms, Cloud={cloud_time:.1f}ms)")
                
            except Exception as e:
                logger.error(f"Failed to test split point {split_point}: {e}")
                split_analysis[split_point] = {
                    'edge_time': 0.0,
                    'transfer_time': 0.0,
                    'cloud_time': 0.0,
                    'total_time': float('inf')
                }
        
        # Find optimal split point
        optimal_split = min(split_analysis.keys(), key=lambda k: split_analysis[k]['total_time'])
        min_time = split_analysis[optimal_split]['total_time']
        
        logger.info(f"=== Optimal Split Point Found ===")
        logger.info(f"Optimal split: {optimal_split} with total time: {min_time:.1f}ms")
        
        # Configure server with optimal split
        self._send_split_decision_to_server(optimal_split, server_address)
        
        # Build summary
        split_summary = build_split_timing_summary(split_analysis, self.get_split_layer_names())

        return optimal_split, {
            'optimal_split': optimal_split,
            'min_total_time': min_time,
            'all_splits': split_analysis,
            'split_summary': split_summary,
            'split_summary_table': format_split_summary(split_summary, sort_by_total_time=False),
            'layer_names': self.get_split_layer_names(),
            'split_config_success': True
        }



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
    



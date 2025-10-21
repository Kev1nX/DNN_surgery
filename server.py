import grpc
import logging
import warnings
from concurrent import futures
import torch
import torch.nn as nn
import time
from typing import Dict, Optional, List, Tuple
import gRPC.protobuf.dnn_inference_pb2 as dnn_inference_pb2
import gRPC.protobuf.dnn_inference_pb2_grpc as dnn_inference_pb2_grpc
from config import GRPC_SETTINGS
from dnn_surgery import DNNSurgery, ModelSplitter, NetworkProfiler
from grpc_utils import proto_to_tensor, tensor_to_proto
from quantization import ModelQuantizer

# Suppress NNPACK warnings
torch.backends.nnpack.enabled = False
warnings.filterwarnings('ignore')
def cuda_sync():
    """Helper function to synchronize CUDA operations if available"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class DNNInferenceServicer(dnn_inference_pb2_grpc.DNNInferenceServicer):
    """Implements the DNNInference gRPC service for distributed DNN computation."""
    
    def __init__(self, enable_quantization: bool = False):
        self.models: Dict[str, torch.nn.Module] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_profiles: Dict[str, dnn_inference_pb2.ClientProfile] = {}
        
        # Additional attributes for handling split models
        self.dnn_surgery_instances: Dict[str, DNNSurgery] = {}
        self.cloud_models: Dict[str, torch.nn.Module] = {}
        self.client_split_points: Dict[str, int] = {}
        
        # Network profiler for transfer time calculations (using same class as client)
        self.network_profiler = NetworkProfiler()
        
        # Quantization support
        self.enable_quantization = enable_quantization
        self.quantizer = ModelQuantizer() if enable_quantization else None
        
        quant_status = "with quantization enabled" if enable_quantization else "without quantization"
        logging.info(f"Initializing DNNInferenceServicer with device: {self.device} {quant_status}")
        
    def MeasureBandwidth(self, request: dnn_inference_pb2.BandwidthProbeRequest,
                         context: grpc.ServicerContext) -> dnn_inference_pb2.BandwidthProbeResponse:
        """Echo payloads back to the client for bandwidth/latency probing."""
        try:
            processing_start = time.perf_counter()

            response_kwargs = {
                "success": True,
                "message": "Bandwidth probe successful",
            }

            if request.HasField("tensor"):
                _ = proto_to_tensor(request.tensor, device=self.device)
                if request.echo:
                    response_kwargs["tensor"] = request.tensor
            else:
                if request.HasField("payload") and request.echo:
                    response_kwargs["payload"] = request.payload
            server_overhead_ms = (time.perf_counter() - processing_start) * 1000

            response_kwargs["server_overhead_ms"] = server_overhead_ms

            return dnn_inference_pb2.BandwidthProbeResponse(**response_kwargs)
        except Exception as exc:
            error_msg = f"Bandwidth probe failed: {exc}"
            logging.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return dnn_inference_pb2.BandwidthProbeResponse(
                success=False,
                message=error_msg,
                server_overhead_ms=0.0
            )

    def register_model(self, model_id: str, model: torch.nn.Module) -> None:
        """Register a model for inference
        
        Args:
            model_id: Unique identifier for the model
            model: PyTorch model to register
        """
        try:
            model = model.to(self.device).eval()  # Ensure model is in eval mode
            self.models[model_id] = model
            
            # Create DNN Surgery instance for this model
            self.dnn_surgery_instances[model_id] = DNNSurgery(model, model_id)
            
            # Log model details
            num_parameters = sum(p.numel() for p in model.parameters())
            requires_grad = any(p.requires_grad for p in model.parameters())
            
            logging.info(f"Registered model: {model_id}")
            logging.info(f"Model details - Parameters: {num_parameters:,}, Device: {self.device}, Requires Grad: {requires_grad}")
            logging.debug(f"Model architecture:\n{model}")
            
        except Exception as e:
            logging.error(f"Failed to register model {model_id}: {str(e)}")
            raise

    def ProcessTensor(self, request: dnn_inference_pb2.InferenceRequest, 
                     context: grpc.ServicerContext) -> dnn_inference_pb2.InferenceResponse:
        """Process a tensor using the specified model with timing
        
        Args:
            request: Inference request containing tensor and model ID
            context: gRPC context
            
        Returns:
            Inference response with processed tensor
        """
        try:
            model = None
            split_point = self.client_split_points.get(request.model_id)
            logging.info(f"Looking for cloud model for request.model_id: '{request.model_id}'")
            logging.info(f"Stored split points: {self.client_split_points}")
            if split_point is not None:
                cloud_key = f"{request.model_id}_split_{split_point}"
                model = self.cloud_models.get(cloud_key)
                if model is not None:
                    logging.info(f"Using cached cloud model for key '{cloud_key}'")
                else:
                    logging.info(f"Cloud model '{cloud_key}' missing. Rebuilding from base model.")
                    base_model = self.models.get(request.model_id)
                    if base_model is None:
                        error_msg = f"Base model {request.model_id} not registered"
                        logging.error(error_msg)
                        context.set_code(grpc.StatusCode.NOT_FOUND)
                        context.set_details(error_msg)
                        return dnn_inference_pb2.InferenceResponse(
                            success=False,
                            error_message=error_msg
                        )
                    splitter = ModelSplitter(base_model, request.model_id)
                    splitter.set_split_point(split_point)
                    model = splitter.get_cloud_model(
                        quantize=self.enable_quantization,
                        quantizer=self.quantizer
                    )
                    if model is None:
                        logging.warning(
                            "Split point %s results in no cloud model for %s; using full model",
                            split_point,
                            request.model_id,
                        )
                        model = base_model
                    else:
                        model = model.to(self.device)
                        self.cloud_models[cloud_key] = model
                        logging.info(f"Recreated and cached cloud model under key '{cloud_key}'")

            if model is None:
                error_msg = f"Model {request.model_id} not found"
                logging.error(error_msg)
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(error_msg)
                return dnn_inference_pb2.InferenceResponse(
                    success=False,
                    error_message=error_msg
                )
                
            # Convert proto to tensor (automatically dequantizes if quantized)
            request_start_time = time.perf_counter()
            input_tensor = proto_to_tensor(request.tensor, device=self.device, dequantize=True)
            
            # Log input tensor stats
            logging.info("=== Cloud Model Processing ===")
            logging.info(f"Input tensor shape: {input_tensor.shape}")
            # Run inference with timing
            compute_start_time = time.perf_counter()
            with torch.no_grad():
                try:
                    # logging.info(f"Cloud model structure:\n{model}")
                    output_tensor = model(input_tensor)
                    
                except Exception as e:
                    error_msg = f"Model inference failed: {str(e)}"
                    logging.error(error_msg)
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details(error_msg)
                    return dnn_inference_pb2.InferenceResponse(
                        success=False,
                        error_message=error_msg
                    )
            
            compute_end_time = time.perf_counter()
            execution_time = (compute_end_time - compute_start_time) * 1000  # ms
            
            logging.info(f"Cloud execution time: {execution_time:.2f}ms")
            logging.info(f"Output tensor shape: {output_tensor.shape}")
            logging.info(f"Output tensor stats - Min: {output_tensor.min().item():.3f}, Max: {output_tensor.max().item():.3f}, Mean: {output_tensor.mean().item():.3f}")
            
            # For classification outputs
            if output_tensor.dim() == 2:
                probs = torch.softmax(output_tensor, dim=1)
                max_prob, pred = probs.max(1)
                logging.info(f"Prediction confidence: {max_prob.item():.3f}, Predicted class: {pred.item()}")
                
            # Convert result back to proto
            response_tensor = tensor_to_proto(output_tensor, ensure_cpu=True)
            response_ready_time = time.perf_counter()
            total_processing_time = (response_ready_time - request_start_time) * 1000  # ms
            
            return dnn_inference_pb2.InferenceResponse(
                tensor=response_tensor,
                success=True,
                server_execution_time_ms=execution_time,
                server_total_time_ms=total_processing_time
            )
            
        except Exception as e:
            error_msg = f"Error processing tensor: {str(e)}"
            logging.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return dnn_inference_pb2.InferenceResponse(
                success=False,
                error_message=error_msg
            )
    
    def SendProfilingData(self, request: dnn_inference_pb2.ProfilingRequest,
                         context: grpc.ServicerContext) -> dnn_inference_pb2.ProfilingResponse:
        """Receive profiling data, run server-side profiling, and return server execution times
        
        The client will use this data to determine the optimal split point.
        
        Args:
            request: Profiling request containing client profile data
            context: gRPC context
            
        Returns:
            Profiling response with server execution times (no split decision)
        """
        try:
            client_id = request.client_id
            client_profile = request.profile
            model_name = client_profile.model_name
            
            # Store the client profile
            self.client_profiles[client_id] = client_profile
            
            logging.info(f"Received profiling data from client: {client_id}")
            logging.info(f"Model: {model_name}, Total layers: {client_profile.total_layers}")
            
            # Run SERVER-SIDE profiling
            logging.info("Running server-side model profiling...")
            server_times = self._profile_server_model(model_name, client_profile)
            
            if server_times is None:
                logging.error("Failed to profile model on server")
                return dnn_inference_pb2.ProfilingResponse(
                    success=False,
                    message="Failed to profile model on server",
                    updated_split_config=""
                )
            
            # Create response with server profiling data ONLY
            # The client will decide the optimal split point
            server_profile_data = self._create_server_response(server_times)
            
            logging.info("Server profiling completed. Sending execution times to client for split decision.")
            
            return dnn_inference_pb2.ProfilingResponse(
                success=True,
                message="Server profiling completed. Client will determine optimal split point.",
                updated_split_config=f"server_profile:{server_profile_data}"
            )
            
        except Exception as e:
            error_msg = f"Error processing profiling data: {str(e)}"
            logging.error(error_msg)
            import traceback
            logging.error(traceback.format_exc())
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return dnn_inference_pb2.ProfilingResponse(
                success=False,
                message=error_msg,
                updated_split_config=""
            )
    
    def _profile_server_model(self, model_name: str, client_profile: dnn_inference_pb2.ClientProfile) -> Optional[List[float]]:
        """Profile the model on server hardware and return execution times
        
        Args:
            model_name: Name of the model to profile
            client_profile: Client profile containing input size information
            
        Returns:
            List of server-side execution times (in ms) for each layer, or None if failed
        """
        try:
            if model_name not in self.models:
                logging.error(f"Model {model_name} not found on server")
                return None
            
            model = self.models[model_name]
            
            # Ensure model is on the correct device (should already be, but make sure)
            model = model.to(self.device).eval()
            
            # Create dummy input tensor with same shape as client used
            input_shape = client_profile.input_size
            # Convert protobuf RepeatedScalarContainer to tuple
            input_shape = tuple(input_shape)
            dummy_input = torch.randn(input_shape).to(self.device)
            
            logging.info(f"Profiling {model_name} on server (device: {self.device})")
            logging.info(f"Input shape: {input_shape}")
            
            # Get model layers using the same approach as client
            dnn_surgery = self.dnn_surgery_instances[model_name]
            layers = dnn_surgery.splitter.layers
            
            server_execution_times = []
            current_tensor = dummy_input
            
            # Profile each layer on server (using same approach as client)
            for idx, layer in enumerate(layers):
                layer_name = layer.__class__.__name__
                
                # Handle tensor shape transitions (same as client)
                profile_input = current_tensor
                needs_flatten = dnn_surgery._needs_flattening_for_layer(layer, current_tensor)
                
                if needs_flatten:
                    profile_input = torch.flatten(current_tensor, 1)
                
                # Profile execution time on server (multiple runs for accuracy)
                times = []
                for _ in range(5):  # Multiple measurements for accuracy
                    cuda_sync()
                    start_time = time.perf_counter()
                    
                    with torch.no_grad():
                        output = layer(profile_input)
                    
                    cuda_sync()
                    end_time = time.perf_counter()
                    
                    times.append((end_time - start_time) * 1000)  # Convert to ms
                
                # Use median time to avoid outliers
                execution_time = sorted(times)[len(times)//2]
                server_execution_times.append(execution_time)
                
                logging.info(f"Server Layer {idx} ({layer_name}): {execution_time:.2f}ms, "
                           f"Input shape: {current_tensor.shape}, Output shape: {output.shape}")
                
                # Update current tensor for next layer (same logic as client)
                with torch.no_grad():
                    if needs_flatten:
                        current_tensor = torch.flatten(current_tensor, 1)
                    current_tensor = layer(current_tensor)
            
            logging.info(f"Server profiling completed for {model_name}")
            return server_execution_times
            
        except Exception as e:
            logging.error(f"Failed to profile model {model_name} on server: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return None

    def _create_server_response(self, server_times: List[float]) -> str:
        """Create a compact representation of server profiling data
        
        Args:
            server_times: List of server execution times
            
        Returns:
            Compact string representation of server profile
        """
        # Create a simple comma-separated string of execution times
        return ','.join(f"{t:.2f}" for t in server_times)

    def set_split_point(self, request: dnn_inference_pb2.SplitConfigRequest,
                       context: grpc.ServicerContext) -> dnn_inference_pb2.SplitConfigResponse:
        """Receive split point decision from client and create cloud model accordingly
        
        Args:
            request: Split configuration request containing split point decision
            context: gRPC context
            
        Returns:
            Response confirming cloud model creation
        """
        try:
            model_name = request.model_name
            split_point = request.split_point
            
            logging.info(f"Received split point decision: split_point={split_point} for model {model_name}")
            
            original_model = self.models.get(model_name)
            if original_model is None:
                error_msg = f"Model {model_name} not registered on server"
                logging.error(error_msg)
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(error_msg)
                return dnn_inference_pb2.SplitConfigResponse(
                    success=False,
                    message=error_msg
                )

            # Create a DNN Surgery instance for this model and split point
            splitter = ModelSplitter(original_model, model_name)
            splitter.set_split_point(split_point)
            # Apply quantization to cloud model if enabled
            cloud_model = splitter.get_cloud_model(
                quantize=self.enable_quantization,
                quantizer=self.quantizer
            )
            
            # Store the client's split decision
            self.client_split_points[model_name] = split_point
            
            if cloud_model is None:
                logging.info(f"Split point {split_point} results in full edge execution for {model_name}")
                # Remove any cached cloud model for this configuration
                cloud_key = f"{model_name}_split_{split_point}"
                self.cloud_models.pop(cloud_key, None)
                return dnn_inference_pb2.SplitConfigResponse(
                    success=True,
                    message=f"Split point {split_point} configured for full edge execution of {model_name}"
                )
            
            # Store the cloud model for later inference
            cloud_key = f"{model_name}_split_{split_point}"
            self.cloud_models[cloud_key] = cloud_model.to(self.device)

            # Remove stale cloud models for the same base model
            for key in list(self.cloud_models.keys()):
                if key.startswith(f"{model_name}_split_") and key != cloud_key:
                    logging.info(f"Removing stale cloud model entry '{key}'")
                    self.cloud_models.pop(key, None)
            
            logging.info(f"Stored cloud model with key: '{cloud_key}'")
            logging.info(f"All cloud model keys now: {list(self.cloud_models.keys())}")
            logging.info(f"Stored split point for '{model_name}': {split_point}")
            logging.info(f"All split points now: {self.client_split_points}")
            
            logging.info(f"Created and stored cloud model for {model_name} at split point {split_point}")
            return dnn_inference_pb2.SplitConfigResponse(
                success=True,
                message=f"Split point {split_point} configured successfully for {model_name}"
            )
            
        except Exception as e:
            error_msg = f"Error setting split point: {str(e)}"
            logging.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return dnn_inference_pb2.SplitConfigResponse(
                success=False,
                message=error_msg
            )
            
def serve(port: int = 50051, max_workers: int = 10, enable_quantization: bool = False) -> tuple[grpc.Server, DNNInferenceServicer]:
    """Start the gRPC server
    
    Args:
        port: Port number to listen on
        max_workers: Maximum number of worker threads
        enable_quantization: Whether to enable INT8 quantization for models
        
    Returns:
        Tuple of (server, servicer)
    """
    # Configure gRPC options for larger messages (individual tensors, not full batches)
    options = GRPC_SETTINGS.channel_options
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=options)
    servicer = DNNInferenceServicer(enable_quantization=enable_quantization)
    dnn_inference_pb2_grpc.add_DNNInferenceServicer_to_server(servicer, server)
    
    server_addr = f'0.0.0.0:{port}'
    server.add_insecure_port(server_addr)
    server.start()
    
    logging.info(
        f"DNN Inference Server started on port {port} with max message size: {GRPC_SETTINGS.max_message_mb}MB"
    )
    return server, servicer


if __name__ == "__main__":
    try:
        # Start server
        server, servicer = serve()
        logging.info("Server started successfully. Press Ctrl+C to exit.")
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Server shutting down...")
    except Exception as e:
        logging.error(f"Server error: {str(e)}")
        raise
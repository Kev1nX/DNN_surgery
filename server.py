import grpc
import logging
from concurrent import futures
import torch
import torch.nn as nn
import io
import time
from typing import Dict, Optional, List
import gRPC.protobuf.dnn_inference_pb2 as dnn_inference_pb2
import gRPC.protobuf.dnn_inference_pb2_grpc as dnn_inference_pb2_grpc
from dnn_surgery import DNNSurgery, ModelSplitter

def cuda_sync():
    """Helper function to synchronize CUDA operations if available"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dnn_server.log')
    ]
)

class DNNInferenceServicer(dnn_inference_pb2_grpc.DNNInferenceServicer):
    """Implements the DNNInference gRPC service for distributed DNN computation."""
    
    def __init__(self):
        self.models: Dict[str, torch.nn.Module] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_profiles: Dict[str, dnn_inference_pb2.ClientProfile] = {}
        
        # Additional attributes for handling split models
        self.dnn_surgery_instances: Dict[str, DNNSurgery] = {}
        self.cloud_models: Dict[str, torch.nn.Module] = {}
        self.client_split_points: Dict[str, int] = {}
        
        logging.info(f"Initializing DNNInferenceServicer with device: {self.device}")
        
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

    def tensor_to_proto(self, tensor: torch.Tensor) -> dnn_inference_pb2.Tensor:
        """Convert PyTorch tensor to protobuf message
        
        Args:
            tensor: PyTorch tensor to convert
            
        Returns:
            Protobuf tensor message
        """
        try:
            logging.debug(f"Converting tensor to proto - Shape: {tensor.shape}, Device: {tensor.device}, Dtype: {tensor.dtype}")
            
            # Move tensor to CPU for serialization
            tensor = tensor.cpu()
            
            # Serialize tensor to bytes
            buffer = io.BytesIO()
            torch.save(tensor, buffer)
            tensor_bytes = buffer.getvalue()
            
            # Create shape message
            shape = dnn_inference_pb2.TensorShape(
                dimensions=list(tensor.shape)
            )
            
            # Create tensor message
            proto = dnn_inference_pb2.Tensor(
                data=tensor_bytes,
                shape=shape,
                dtype=str(tensor.dtype),
                requires_grad=tensor.requires_grad
            )
            
            logging.debug(f"Successfully converted tensor to proto message of size {len(tensor_bytes)} bytes")
            return proto
            
        except Exception as e:
            logging.error(f"Failed to convert tensor to proto: {str(e)}")
            raise
        
    def proto_to_tensor(self, proto: dnn_inference_pb2.Tensor) -> torch.Tensor:
        """Convert protobuf message back to PyTorch tensor
        
        Args:
            proto: Protobuf tensor message
            
        Returns:
            PyTorch tensor
        """
        # Deserialize tensor
        buffer = io.BytesIO(proto.data)
        tensor = torch.load(buffer)
        
        # Move to correct device
        return tensor.to(self.device)
        
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
            # Get the model
            model = self.models.get(request.model_id)
            if model is None:
                error_msg = f'Model {request.model_id} not found'
                logging.error(error_msg)
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(error_msg)
                return dnn_inference_pb2.InferenceResponse(
                    success=False,
                    error_message=error_msg
                )
                
            # Convert proto to tensor
            input_tensor = self.proto_to_tensor(request.tensor)
            
            # Log input tensor stats
            logging.info("=== Cloud Model Processing ===")
            logging.info(f"Input tensor shape: {input_tensor.shape}")
            logging.info(f"Input tensor stats - Min: {input_tensor.min().item():.3f}, Max: {input_tensor.max().item():.3f}, Mean: {input_tensor.mean().item():.3f}")
            
            # Run inference with timing
            start_time = time.perf_counter()
            with torch.no_grad():
                try:
                    logging.info(f"Cloud model structure:\n{model}")
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
            
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000  # ms
            
            logging.info(f"Cloud execution time: {execution_time:.2f}ms")
            logging.info(f"Output tensor shape: {output_tensor.shape}")
            logging.info(f"Output tensor stats - Min: {output_tensor.min().item():.3f}, Max: {output_tensor.max().item():.3f}, Mean: {output_tensor.mean().item():.3f}")
            
            # For classification outputs
            if output_tensor.dim() == 2:
                probs = torch.softmax(output_tensor, dim=1)
                max_prob, pred = probs.max(1)
                logging.info(f"Prediction confidence: {max_prob.item():.3f}, Predicted class: {pred.item()}")
                
            # Convert result back to proto
            response_tensor = self.tensor_to_proto(output_tensor)
            
            return dnn_inference_pb2.InferenceResponse(
                tensor=response_tensor,
                success=True
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
        """Receive profiling data, run server-side profiling, and return optimal split configuration
        
        Args:
            request: Profiling request containing client profile data
            context: gRPC context
            
        Returns:
            Profiling response with optimal split configuration and server profiling data
        """
        try:
            client_id = request.client_id
            client_profile = request.profile
            model_name = client_profile.model_name
            
            # Store the client profile
            self.client_profiles[client_id] = client_profile
            
            logging.info(f"Received profiling data from client: {client_id}")
            logging.info(f"Model: {model_name}, Total layers: {client_profile.total_layers}")
            
            # Log client-side layer details
            logging.info("Client-side layer profiling:")
            for layer_metric in client_profile.layer_metrics:
                logging.info(f"  Layer {layer_metric.layer_idx} ({layer_metric.layer_name}): "
                           f"{layer_metric.execution_time:.2f}ms (client), "
                           f"Transfer size: {layer_metric.data_transfer_size} bytes")
            
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
            
            # Compare client vs server performance
            logging.info("=== Client vs Server Performance Comparison ===")
            for i, (client_layer, server_time) in enumerate(zip(client_profile.layer_metrics, server_times)):
                speedup = client_layer.execution_time / server_time if server_time > 0 else float('inf')
                logging.info(f"Layer {i} ({client_layer.layer_name}): "
                           f"Client={client_layer.execution_time:.2f}ms, "
                           f"Server={server_time:.2f}ms, "
                           f"Speedup={speedup:.2f}x")
            
            # Find optimal split point using both client and server timings
            optimal_split = self._find_optimal_split_server(client_profile, server_times)
            
            # Create cloud model for this client
            client_model_key = f"{client_id}_{model_name}"
            if model_name in self.dnn_surgery_instances:
                cloud_model = self.create_cloud_model(model_name, optimal_split)
                if cloud_model is not None:
                    self.cloud_models[client_model_key] = cloud_model
                    self.client_split_points[client_model_key] = optimal_split
                    
                    logging.info(f"Created cloud model for client {client_id} with optimal split point: {optimal_split}")
                else:
                    logging.warning(f"No cloud model needed for split point {optimal_split} - all computation on edge")
            else:
                logging.error(f"DNNSurgery instance not found for model {model_name}")
                logging.info(f"Available models: {list(self.dnn_surgery_instances.keys())}")
            
            # Create response with server profiling data
            server_profile_data = self._create_server_response(server_times)
            
            return dnn_inference_pb2.ProfilingResponse(
                success=True,
                message=f"Profiling completed. Optimal split point: {optimal_split}. Server profiling included.",
                updated_split_config=f"split_point:{optimal_split};server_profile:{server_profile_data}"
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
            
            # Create dummy input tensor with same shape as client used
            input_shape = client_profile.input_size
            dummy_input = torch.randn(input_shape).to(self.device)
            
            logging.info(f"Profiling {model_name} on server (device: {self.device})")
            logging.info(f"Input shape: {input_shape}")
            
            # Get model layers using the same approach as client
            dnn_surgery = self.dnn_surgery_instances[model_name]
            layers = dnn_surgery.splitter.layers
            
            server_execution_times = []
            current_tensor = dummy_input
            
            # Profile each layer on server
            for idx, layer in enumerate(layers):
                layer_name = layer.__class__.__name__
                
                # Handle tensor shape transitions (same as client)
                profile_input = current_tensor
                if isinstance(layer, nn.Linear) and current_tensor.dim() > 2:
                    profile_input = torch.flatten(current_tensor, 1)
                
                # Profile execution time on server
                cuda_sync()
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    output = layer(profile_input)
                
                cuda_sync()
                end_time = time.perf_counter()
                
                execution_time = (end_time - start_time) * 1000  # Convert to ms
                server_execution_times.append(execution_time)
                
                logging.info(f"Server Layer {idx} ({layer_name}): {execution_time:.2f}ms, "
                           f"Output shape: {output.shape}")
                
                # Update current tensor for next layer (same logic as client)
                with torch.no_grad():
                    if isinstance(layer, nn.Linear) and current_tensor.dim() > 2:
                        current_tensor = torch.flatten(current_tensor, 1)
                    current_tensor = layer(current_tensor)
            
            logging.info(f"Server profiling completed for {model_name}")
            return server_execution_times
            
        except Exception as e:
            logging.error(f"Failed to profile model {model_name} on server: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return None
    
    def _find_optimal_split_server(self, client_profile: dnn_inference_pb2.ClientProfile, 
                                           server_times: List[float]) -> int:
        """Find optimal split point using both client and server timing data
        
        Args:
            client_profile: Client profile with layer metrics
            server_times: List of server execution times
            
        Returns:
            Optimal split point
        """
        client_times = [layer.execution_time for layer in client_profile.layer_metrics]
        
        if len(client_times) != len(server_times):
            logging.warning(f"Client and server layer counts don't match: {len(client_times)} vs {len(server_times)}")
            # Fall back to simple split
            return len(client_times) // 2
        
        best_split = 0
        min_total_time = float('inf')
        
        logging.info("=== Server-side Split Analysis ===")
        logging.info(f"{'Split':<5} {'Client':<8} {'Server':<8} {'Total':<8} {'Description'}")
        logging.info("-" * 50)
        
        for split_point in range(len(client_times) + 1):
            # Client execution time (layers 0 to split_point-1)
            client_time = sum(client_times[:split_point])
            
            # Server execution time (layers split_point to end)
            server_time = sum(server_times[split_point:])
            
            # Simplified total time (ignoring transfer for server-side calculation)
            total_time = max(client_time, server_time)  # Assuming parallel execution
            
            # Create description
            if split_point == 0:
                desc = "All server"
            elif split_point >= len(client_times):
                desc = "All client"
            else:
                desc = f"Split at {split_point}"
            
            logging.info(f"{split_point:<5} {client_time:<8.1f} {server_time:<8.1f} "
                       f"{total_time:<8.1f} {desc}")
            
            if total_time < min_total_time:
                min_total_time = total_time
                best_split = split_point
        
        logging.info("-" * 50)
        logging.info(f"Server recommends split point: {best_split}")
        
        return best_split
    
    def _create_server_response(self, server_times: List[float]) -> str:
        """Create a compact representation of server profiling data
        
        Args:
            server_times: List of server execution times
            
        Returns:
            Compact string representation of server profile
        """
        # Create a simple comma-separated string of execution times
        return ','.join(f"{t:.2f}" for t in server_times)

    def create_cloud_model(self, model_name: str, split_point: int):
        """Create cloud-side model for distributed inference
        
        Args:
            model_name: Name of the model to split
            split_point: Layer index where to split
            
        Returns:
            Cloud-side model or None if creation fails
        """
        if model_name not in self.models:
            logging.error(f"Model {model_name} not registered")
            return None
        
        try:
            # Create a DNN Surgery instance for this model and split point
            original_model = self.models[model_name]
            splitter = ModelSplitter(original_model, model_name)
            splitter.set_split_point(split_point)
            cloud_model = splitter.get_cloud_model()
            
            if cloud_model is None:
                logging.warning(f"Cloud model is None for split point {split_point} - model runs entirely on edge")
                return None
            
            logging.info(f"Created cloud model for {model_name} at split point {split_point}")
            return cloud_model
            
        except Exception as e:
            logging.error(f"Failed to create cloud model: {str(e)}")
            return None
            
def serve(port: int = 50051, max_workers: int = 10) -> tuple[grpc.Server, DNNInferenceServicer]:
    """Start the gRPC server
    
    Args:
        port: Port number to listen on
        max_workers: Maximum number of worker threads
        
    Returns:
        Tuple of (server, servicer)
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = DNNInferenceServicer()
    dnn_inference_pb2_grpc.add_DNNInferenceServicer_to_server(servicer, server)
    
    server_addr = f'0.0.0.0:{port}'
    server.add_insecure_port(server_addr)
    server.start()
    
    logging.info(f"DNN Inference Server started on port {port}")
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
import grpc
import logging
from concurrent import futures
import torch
import torch.nn as nn
import io
import time
from typing import Dict, Optional
import gRPC.protobuf.dnn_inference_pb2 as dnn_inference_pb2
import gRPC.protobuf.dnn_inference_pb2_grpc as dnn_inference_pb2_grpc
from dnn_surgery import DNNSurgery, LayerProfiler

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
        self.profiler = LayerProfiler()
        self.client_profiles: Dict[str, dnn_inference_pb2.ClientProfile] = {}
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
        """Receive profiling data and determine optimal split point
        
        Args:
            request: Profiling request containing client profile data
            context: gRPC context
            
        Returns:
            Profiling response with optimal split configuration
        """
        try:
            client_id = request.client_id
            profile = request.profile
            model_name = profile.model_name
            
            # Store the profile
            self.client_profiles[client_id] = profile
            
            logging.info(f"Received profiling data from client: {client_id}")
            logging.info(f"Model: {model_name}, Total layers: {profile.total_layers}")
            
            # Log layer details
            for layer_metric in profile.layer_metrics:
                logging.info(f"Layer {layer_metric.layer_idx} ({layer_metric.layer_name}): "
                           f"{layer_metric.execution_time:.2f}ms, "
                           f"Transfer size: {layer_metric.data_transfer_size} bytes")
            
            # Find optimal split point using NeuroSurgeon approach
            optimal_split = self._find_optimal_split_from_profile(profile)
            
            # Create cloud model for this client
            client_model_key = f"{client_id}_{model_name}"
            if model_name in self.dnn_surgery_instances:
                cloud_model = self.create_cloud_model(model_name, optimal_split)
                if cloud_model is not None:
                    self.cloud_models[client_model_key] = cloud_model
                    self.client_split_points[client_model_key] = optimal_split
                    
                    logging.info(f"Created cloud model for client {client_id} with optimal split point: {optimal_split}")
                else:
                    logging.warning(f"Failed to create cloud model for split point {optimal_split}")
            
            return dnn_inference_pb2.ProfilingResponse(
                success=True,
                message=f"Profiling data processed. Optimal split point: {optimal_split}",
                updated_split_config=f"split_point:{optimal_split}"
            )
            
        except Exception as e:
            error_msg = f"Error processing profiling data: {str(e)}"
            logging.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return dnn_inference_pb2.ProfilingResponse(
                success=False,
                message=error_msg,
                updated_split_config=""
            )
    
    def _find_optimal_split_from_profile(self, profile: dnn_inference_pb2.ClientProfile) -> int:
        """Find optimal split point from client profiling data using simplified NeuroSurgeon approach
        
        Args:
            profile: Client profile with layer metrics
            
        Returns:
            Optimal split point
        """
        # For server-side optimization, we use a simplified approach since we don't have
        # the actual network measurements here. The client should ideally send this.
        # We'll optimize based on computation time distribution.
        
        layer_times = [layer.execution_time for layer in profile.layer_metrics]
        total_time = sum(layer_times)
        
        if total_time == 0:
            return 0
        
        # Find the split point that balances computation
        # This is a simplified heuristic - ideally the client would send network metrics too
        min_imbalance = float('inf')
        optimal_split = 0
        
        for split_point in range(len(layer_times) + 1):
            client_time = sum(layer_times[:split_point])
            server_time = sum(layer_times[split_point:])
            
            # Simple heuristic: minimize the difference between client and server times
            # In a real implementation, this would include transfer times
            imbalance = abs(client_time - server_time)
            
            if imbalance < min_imbalance:
                min_imbalance = imbalance
                optimal_split = split_point
        
        logging.info(f"Server-side optimal split calculation: split_point={optimal_split}, "
                   f"client_time={sum(layer_times[:optimal_split]):.2f}ms, "
                   f"server_time={sum(layer_times[optimal_split:]):.2f}ms")
        
        return optimal_split
            
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
    
    server_addr = f'[::]:{port}'
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
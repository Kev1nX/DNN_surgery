import grpc
import logging
from concurrent import futures
import torch
import torch.nn as nn
import io
from typing import Dict, Optional
import gRPC.protobuf.dnn_inference_pb2 as dnn_inference_pb2
import gRPC.protobuf.dnn_inference_pb2_grpc as dnn_inference_pb2_grpc

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
        """Process a tensor using the specified model
        
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
            
            # Run inference
            with torch.no_grad():
                try:
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
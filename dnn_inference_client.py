import grpc
import logging
from concurrent import futures
import torch
import io
import time
from typing import Dict, Optional
import gRPC.protobuf.dnn_inference_pb2 as dnn_inference_pb2
import gRPC.protobuf.dnn_inference_pb2_grpc as dnn_inference_pb2_grpc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DNNInferenceClient:
    """Client for the DNNInference service with edge computing capability."""
    
    def __init__(self, server_address: str = 'localhost:50051', edge_model: Optional[torch.nn.Module] = None):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = dnn_inference_pb2_grpc.DNNInferenceStub(self.channel)
        self.edge_model = edge_model
        
        if self.edge_model is not None:
            self.edge_model.eval()  # Ensure model is in evaluation mode
            logging.info(f"Initialized edge model: {type(edge_model).__name__}")
        
    def tensor_to_proto(self, tensor: torch.Tensor) -> dnn_inference_pb2.Tensor:
        """Convert PyTorch tensor to protobuf message"""
        try:
            buffer = io.BytesIO()
            torch.save(tensor, buffer)
            tensor_bytes = buffer.getvalue()
            
            shape = dnn_inference_pb2.TensorShape(
                dimensions=list(tensor.shape)
            )
            
            logging.debug(f"Converting tensor of shape {tensor.shape} to proto message")
            return dnn_inference_pb2.Tensor(
                data=tensor_bytes,
                shape=shape,
                dtype=str(tensor.dtype),
                requires_grad=tensor.requires_grad
            )
        except Exception as e:
            logging.error(f"Failed to convert tensor to proto: {str(e)}")
            raise
        
    def proto_to_tensor(self, proto: dnn_inference_pb2.Tensor) -> torch.Tensor:
        """Convert protobuf message back to PyTorch tensor"""
        try:
            buffer = io.BytesIO(proto.data)
            tensor = torch.load(buffer)
            
            # Verify received tensor
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Received data is not a tensor: {type(tensor)}")
            if torch.isnan(tensor).any():
                raise ValueError("Received tensor contains NaN values")
            if torch.isinf(tensor).any():
                raise ValueError("Received tensor contains infinite values")
                
            logging.debug(f"Converted proto message back to tensor of shape {tensor.shape}")
            logging.debug(f"Tensor stats - Min: {tensor.min().item():.3f}, Max: {tensor.max().item():.3f}, Mean: {tensor.mean().item():.3f}")
            return tensor
        except Exception as e:
            logging.error(f"Failed to convert proto back to tensor: {str(e)}")
            raise
        
    def process_tensor(self, tensor: torch.Tensor, model_id: str) -> torch.Tensor:
        """Process tensor using edge model if available, then send to server for further processing"""
        logging.info(f"Processing tensor of shape {tensor.shape} with model {model_id}")
        try:
            # Log input tensor stats
            logging.info(f"Input tensor stats - Min: {tensor.min().item():.3f}, Max: {tensor.max().item():.3f}, Mean: {tensor.mean().item():.3f}")
            
            # Run edge inference if available
            if self.edge_model is not None:
                logging.info("=== Edge Model Processing ===")
                logging.info(f"Edge model structure:\n{self.edge_model}")
                start_time = time.perf_counter()
                with torch.no_grad():
                    tensor = self.edge_model(tensor)
                edge_time = (time.perf_counter() - start_time) * 1000  # ms
                logging.info(f"Edge inference completed in {edge_time:.2f}ms")
                logging.info(f"Intermediate tensor shape: {tensor.shape}")
                logging.info(f"Intermediate tensor stats - Min: {tensor.min().item():.3f}, Max: {tensor.max().item():.3f}, Mean: {tensor.mean().item():.3f}")
                if torch.isnan(tensor).any():
                    logging.error("Edge model output contains NaN values!")
                if torch.isinf(tensor).any():
                    logging.error("Edge model output contains Inf values!")
            
            request = dnn_inference_pb2.InferenceRequest(
                tensor=self.tensor_to_proto(tensor),
                model_id=model_id
            )
            
            response = self.stub.ProcessTensor(request)
            
            if not response.success:
                logging.error(f"Server processing failed: {response.error_message}")
                raise RuntimeError(f"Server error: {response.error_message}")
            
            result = self.proto_to_tensor(response.tensor)
            
            # Log output tensor stats
            logging.info(f"Output tensor stats - Shape: {result.shape}")
            if result.dim() == 2:  # For classification outputs
                probs = torch.softmax(result, dim=1)
                max_prob, pred = probs.max(1)
                logging.info(f"Prediction confidence: {max_prob.item():.3f}, Predicted class: {pred.item()}")
                
            logging.info(f"Successfully processed tensor with model {model_id}")
            return result
        except grpc.RpcError as rpc_error:
            logging.error(f"gRPC error: {rpc_error.code()}: {rpc_error.details()}")
            raise
        except Exception as e:
            logging.error(f"Error processing tensor: {str(e)}")
            raise

def run_edge_inference(model_id: str, input_tensor: torch.Tensor, server_address: str = 'localhost:50051') -> torch.Tensor:
    """Convenience function to run inference on edge device"""
    client = DNNInferenceClient(server_address)
    return client.process_tensor(input_tensor, model_id)

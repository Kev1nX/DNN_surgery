import grpc
import logging
from concurrent import futures
import torch
import io
import time
import uuid
from typing import Dict, Optional, Tuple
import gRPC.protobuf.dnn_inference_pb2 as dnn_inference_pb2
import gRPC.protobuf.dnn_inference_pb2_grpc as dnn_inference_pb2_grpc
from dnn_surgery import DNNSurgery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DNNInferenceClient:
    """Client for the DNNInference service with edge computing capability and profiling."""
    
    def __init__(self, server_address: str = 'localhost:50051', edge_model: Optional[torch.nn.Module] = None):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = dnn_inference_pb2_grpc.DNNInferenceStub(self.channel)
        self.edge_model = edge_model
        self.client_id = str(uuid.uuid4())
        self.transfer_times = []
        self.edge_times = []
        self.cloud_times = []
        
        if self.edge_model is not None:
            self.edge_model.eval()  # Ensure model is in evaluation mode
            logging.info(f"Initialized edge model: {type(edge_model).__name__}")
        
        logging.info(f"Client ID: {self.client_id}")
        
    def measure_transfer_time(self, tensor: torch.Tensor, model_id: str) -> float:
        """Measure round-trip transfer time for a tensor
        
        Args:
            tensor: Tensor to measure transfer time for
            model_id: Model ID for the request
            
        Returns:
            Transfer time in milliseconds
        """
        request = dnn_inference_pb2.InferenceRequest(
            tensor=self.tensor_to_proto(tensor),
            model_id=model_id
        )
        
        start_time = time.perf_counter()
        response = self.stub.ProcessTensor(request)
        end_time = time.perf_counter()
        
        if not response.success:
            raise RuntimeError(f"Transfer measurement failed: {response.error_message}")
        
        transfer_time = (end_time - start_time) * 1000  # ms
        return transfer_time
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
        
    def process_tensor(self, tensor: torch.Tensor, model_id: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Process tensor using edge model if available, then send to server for further processing
        
        Args:
            tensor: Input tensor to process
            model_id: Model identifier
            
        Returns:
            Tuple of (result tensor, timing dictionary)
        """
        logging.info(f"Processing tensor of shape {tensor.shape} with model {model_id}")
        timings = {'edge_time': 0.0, 'transfer_time': 0.0, 'cloud_time': 0.0}
        
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
                end_time = time.perf_counter()
                
                edge_time = (end_time - start_time) * 1000  # ms
                timings['edge_time'] = edge_time
                self.edge_times.append(edge_time)
                
                logging.info(f"Edge inference completed in {edge_time:.2f}ms")
                logging.info(f"Intermediate tensor shape: {tensor.shape}")
                logging.info(f"Intermediate tensor stats - Min: {tensor.min().item():.3f}, Max: {tensor.max().item():.3f}, Mean: {tensor.mean().item():.3f}")
                
                if torch.isnan(tensor).any():
                    logging.error("Edge model output contains NaN values!")
                if torch.isinf(tensor).any():
                    logging.error("Edge model output contains Inf values!")
            
            # Measure cloud processing with transfer time
            request = dnn_inference_pb2.InferenceRequest(
                tensor=self.tensor_to_proto(tensor),
                model_id=model_id
            )
            
            # Measure total round-trip time
            start_time = time.perf_counter()
            response = self.stub.ProcessTensor(request)
            end_time = time.perf_counter()
            
            total_time = (end_time - start_time) * 1000  # ms
            
            if not response.success:
                logging.error(f"Server processing failed: {response.error_message}")
                raise RuntimeError(f"Server error: {response.error_message}")
            
            result = self.proto_to_tensor(response.tensor)
            
            # Estimate transfer time (simplified: assume cloud processing is fast)
            # In a real implementation, you would separate this more carefully
            transfer_time = total_time * 0.2  # Rough estimate: 20% transfer, 80% compute
            cloud_time = total_time - transfer_time
            
            timings['transfer_time'] = transfer_time
            timings['cloud_time'] = cloud_time
            
            self.transfer_times.append(transfer_time)
            self.cloud_times.append(cloud_time)
            
            logging.info(f"Cloud processing time: {cloud_time:.2f}ms")
            logging.info(f"Transfer time: {transfer_time:.2f}ms")
            logging.info(f"Total time: {total_time:.2f}ms")
            
            # Log output tensor stats
            logging.info(f"Output tensor stats - Shape: {result.shape}")
            if result.dim() == 2:  # For classification outputs
                probs = torch.softmax(result, dim=1)
                max_prob, pred = probs.max(1)
                logging.info(f"Prediction confidence: {max_prob.item():.3f}, Predicted class: {pred.item()}")
                
            logging.info(f"Successfully processed tensor with model {model_id}")
            return result, timings
            
        except grpc.RpcError as rpc_error:
            logging.error(f"gRPC error: {rpc_error.code()}: {rpc_error.details()}")
            raise
        except Exception as e:
            logging.error(f"Error processing tensor: {str(e)}")
            raise
    
    def send_profiling_data(self, dnn_surgery: DNNSurgery, input_tensor: torch.Tensor) -> bool:
        """Send profiling data to server
        
        Args:
            dnn_surgery: DNNSurgery instance with profiling data
            input_tensor: Input tensor used for profiling
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create client profile
            client_profile = dnn_surgery.create_client_profile(input_tensor)
            
            # Create profiling request
            request = dnn_inference_pb2.ProfilingRequest(
                profile=client_profile,
                client_id=self.client_id
            )
            
            # Send to server
            response = self.stub.SendProfilingData(request)
            
            if response.success:
                logging.info(f"Profiling data sent successfully: {response.message}")
                return True
            else:
                logging.error(f"Failed to send profiling data: {response.message}")
                return False
                
        except Exception as e:
            logging.error(f"Error sending profiling data: {str(e)}")
            return False
    
    def get_timing_summary(self) -> Dict[str, float]:
        """Get summary of timing measurements
        
        Returns:
            Dictionary with timing statistics
        """
        summary = {}
        
        if self.edge_times:
            summary['avg_edge_time'] = sum(self.edge_times) / len(self.edge_times)
            summary['min_edge_time'] = min(self.edge_times)
            summary['max_edge_time'] = max(self.edge_times)
        
        if self.transfer_times:
            summary['avg_transfer_time'] = sum(self.transfer_times) / len(self.transfer_times)
            summary['min_transfer_time'] = min(self.transfer_times)
            summary['max_transfer_time'] = max(self.transfer_times)
            
        if self.cloud_times:
            summary['avg_cloud_time'] = sum(self.cloud_times) / len(self.cloud_times)
            summary['min_cloud_time'] = min(self.cloud_times)
            summary['max_cloud_time'] = max(self.cloud_times)
            
        return summary

def run_distributed_inference_with_profiling(model_id: str, input_tensor: torch.Tensor, 
                                            dnn_surgery: DNNSurgery, split_point: int = 0,
                                            server_address: str = 'localhost:50051') -> Tuple[torch.Tensor, Dict]:
    """Run distributed inference with profiling
    
    Args:
        model_id: Model identifier
        input_tensor: Input tensor
        dnn_surgery: DNNSurgery instance for model splitting
        split_point: Where to split the model (0 = all cloud, max = all edge)
        server_address: Server address
        
    Returns:
        Tuple of (result tensor, timing dictionary)
    """
    # Set split point and get edge model
    dnn_surgery.splitter.set_split_point(split_point)
    edge_model = dnn_surgery.splitter.get_edge_model()
    
    # Create client with edge model
    client = DNNInferenceClient(server_address, edge_model)
    
    # Run inference
    result, timings = client.process_tensor(input_tensor, model_id)
    
    # Send profiling data
    client.send_profiling_data(dnn_surgery, input_tensor)
    
    # Get timing summary
    timing_summary = client.get_timing_summary()
    timings.update(timing_summary)
    
    return result, timings

def run_edge_inference(model_id: str, input_tensor: torch.Tensor, server_address: str = 'localhost:50051') -> torch.Tensor:
    """Convenience function to run inference on edge device"""
    client = DNNInferenceClient(server_address)
    result, _ = client.process_tensor(input_tensor, model_id)
    return result

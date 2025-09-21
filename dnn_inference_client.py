import grpc
import logging
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
        
    def process_tensor(self, tensor: torch.Tensor, model_id: str, requires_cloud_processing: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Process tensor using edge model if available, then send to server for further processing
        
        Args:
            tensor: Input tensor to process
            model_id: Model identifier
            requires_cloud_processing: Whether cloud processing is needed (False for all-edge inference)
            
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
            
            # Check if cloud processing is required
            if not requires_cloud_processing:
                # All inference is done on client side - return edge model result
                logging.info("All inference completed on client side - no server communication needed")
                result = tensor
                
                # Log output tensor stats
                logging.info(f"Output tensor stats - Shape: {result.shape}")
                if result.dim() == 2:  # For classification outputs
                    probs = torch.softmax(result, dim=1)
                    max_prob, pred = probs.max(1)
                    logging.info(f"Prediction confidence: {max_prob.item():.3f}, Predicted class: {pred.item()}")
                    
                logging.info(f"Successfully processed tensor with model {model_id} (client-only)")
                return result, timings
            
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

def run_distributed_inference(model_id: str, input_tensor: torch.Tensor, 
                                            dnn_surgery: DNNSurgery, split_point: int = None,
                                            server_address: str = 'localhost:50051') -> Tuple[torch.Tensor, Dict]:
    """Run distributed inference with NeuroSurgeon optimization
    
    Args:
        model_id: Model identifier
        input_tensor: Input tensor
        dnn_surgery: DNNSurgery instance for model splitting
        split_point: Optional manual split point (if None, will use NeuroSurgeon optimization)
        server_address: Server address
        
    Returns:
        Tuple of (result tensor, timing dictionary)
    """
    try:
        # Use NeuroSurgeon approach if no manual split point provided
        if split_point is None:
            optimal_split, analysis = dnn_surgery.find_optimal_split(input_tensor, server_address)
            split_point = optimal_split
            logging.info(f"NeuroSurgeon optimal split point: {split_point}")
            logging.info(f"Predicted total time: {analysis['min_total_time']:.2f}ms")
        
        # Set split point and get edge model
        dnn_surgery.splitter.set_split_point(split_point)
        edge_model = dnn_surgery.splitter.get_edge_model()
        
        # Check if cloud processing is needed
        cloud_model = dnn_surgery.splitter.get_cloud_model()
        requires_cloud_processing = cloud_model is not None
        
        if not requires_cloud_processing:
            logging.info(f"Split point {split_point} means all inference on client side - no server communication needed")
        
        # Create client with edge model
        client = DNNInferenceClient(server_address, edge_model)
        
        # Run inference
        result, timings = client.process_tensor(input_tensor, model_id, requires_cloud_processing)
        
        # Send profiling data to server for future optimizations (only if server communication is available)
        if requires_cloud_processing:
            client.send_profiling_data(dnn_surgery, input_tensor)
        
        # Get timing summary
        timing_summary = client.get_timing_summary()
        timings.update(timing_summary)
        
        # Add split point info
        timings['split_point'] = split_point
        
        return result, timings
        
    except Exception as e:
        logging.error(f"Distributed inference failed: {str(e)}")
        raise

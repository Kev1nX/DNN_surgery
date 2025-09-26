import grpc
import logging
import torch
import time
import uuid
from typing import Dict, Optional, Tuple
import gRPC.protobuf.dnn_inference_pb2 as dnn_inference_pb2
import gRPC.protobuf.dnn_inference_pb2_grpc as dnn_inference_pb2_grpc
from config import GRPC_SETTINGS
from dnn_surgery import DNNSurgery
from grpc_utils import proto_to_tensor, tensor_to_proto

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DNNInferenceClient:
    """Client for the DNNInference service with edge computing capability and profiling."""
    
    def __init__(self, server_address: str = 'localhost:50051', edge_model: Optional[torch.nn.Module] = None):
        # Configure gRPC options for larger messages (individual tensors)
        options = GRPC_SETTINGS.channel_options
        
        self.channel = grpc.insecure_channel(server_address, options=options)
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
        logging.info(f"Configured max gRPC message size: {GRPC_SETTINGS.max_message_mb}MB")
        
    def process_tensor(self, tensor: torch.Tensor, model_id: str, requires_cloud_processing: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Process tensor using edge model if available, then send to server for further processing
        
        Automatically handles batched tensors by separating them into individual samples
        to avoid gRPC message size limits.
        
        Args:
            tensor: Input tensor to process (can be batched)
            model_id: Model identifier
            requires_cloud_processing: Whether cloud processing is needed (False for all-edge inference)
            
        Returns:
            Tuple of (result tensor, timing dictionary)
        """
        batch_size = tensor.shape[0]
        
        if batch_size == 1:
            # Single tensor - process normally
            return self._process_single_tensor(tensor, model_id, requires_cloud_processing)
        else:
            # Batched tensor - separate into individual samples
            logging.info(f"Processing batched tensor of shape {tensor.shape} (batch_size={batch_size}) with model {model_id}")
            return self._process_batched_tensor(tensor, model_id, requires_cloud_processing)
    
    def _process_single_tensor(self, tensor: torch.Tensor, model_id: str, requires_cloud_processing: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Process a single tensor (batch_size=1)
        
        Args:
            tensor: Input tensor to process (batch_size=1)
            model_id: Model identifier
            requires_cloud_processing: Whether cloud processing is needed (False for all-edge inference)
            
        Returns:
            Tuple of (result tensor, timing dictionary)
        """
        logging.info(f"Processing single tensor of shape {tensor.shape} with model {model_id}")
        timings = {'edge_time': 0.0, 'transfer_time': 0.0, 'cloud_time': 0.0}
        
        try:
            # Log input tensor stats
            logging.info(f"Input tensor stats - Min: {tensor.min().item():.3f}, Max: {tensor.max().item():.3f}, Mean: {tensor.mean().item():.3f}")
            
            # Run edge inference if available
            if self.edge_model is not None:
                logging.info("=== Edge Model Processing ===")
                # logging.info(f"Edge model structure:\n{self.edge_model}")
                
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
                tensor=tensor_to_proto(tensor, ensure_cpu=True),
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
            
            result = proto_to_tensor(response.tensor)
            if torch.isnan(result).any():
                raise ValueError("Received tensor contains NaN values")
            if torch.isinf(result).any():
                raise ValueError("Received tensor contains infinite values")
            
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
        
    def _process_batched_tensor(self, tensor: torch.Tensor, model_id: str, requires_cloud_processing: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Process a batched tensor by separating into individual samples
        
        Args:
            tensor: Input batched tensor to process
            model_id: Model identifier
            requires_cloud_processing: Whether cloud processing is needed
            
        Returns:
            Tuple of (batched result tensor, aggregated timing dictionary)
        """
        batch_size = tensor.shape[0]
        logging.info(f"Separating batch of {batch_size} samples for individual processing")
        
        # Initialize aggregated timings
        aggregated_timings = {'edge_time': 0.0, 'transfer_time': 0.0, 'cloud_time': 0.0}
        individual_results = []
        
        # Process each sample individually
        for i in range(batch_size):
            sample_tensor = tensor[i:i+1]  # Keep batch dimension but size 1
            logging.info(f"Processing sample {i+1}/{batch_size} with shape {sample_tensor.shape}")
            
            result, timings = self._process_single_tensor(sample_tensor, model_id, requires_cloud_processing)
            individual_results.append(result)
            
            # Aggregate timings
            for key in aggregated_timings:
                aggregated_timings[key] += timings.get(key, 0.0)
        
        # Concatenate results back into batch format
        batched_result = torch.cat(individual_results, dim=0)
        
        # Calculate average timings per sample
        avg_timings = {key: value / batch_size for key, value in aggregated_timings.items()}
        avg_timings['total_batch_processing_time'] = sum(aggregated_timings.values())
        avg_timings['batch_size'] = batch_size
        
        logging.info(f"Batch processing completed. Total time: {avg_timings['total_batch_processing_time']:.2f}ms, "
                    f"Average per sample: {avg_timings['total_batch_processing_time']/batch_size:.2f}ms")
        
        return batched_result, avg_timings
    
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
        else:
            # Manual split point provided - send it to server
            logging.info(f"Using manual split point: {split_point}")
            success = dnn_surgery._send_split_decision_to_server(split_point, server_address)
            if not success:
                logging.error("Failed to send manual split decision to server")
        
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
        
        # Get timing summary
        timing_summary = client.get_timing_summary()
        timings.update(timing_summary)
        
        # Add split point info
        timings['split_point'] = split_point
        
        return result, timings
        
    except Exception as e:
        logging.error(f"Distributed inference failed: {str(e)}")
        raise

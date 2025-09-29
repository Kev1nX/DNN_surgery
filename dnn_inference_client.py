import grpc
import logging
import torch
import time
import uuid
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
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
    
    def __init__(
        self,
        server_address: str = 'localhost:50051',
        edge_model: Optional[torch.nn.Module] = None,
        max_inflight_requests: Optional[int] = None,
    ):
        # Configure gRPC options for larger messages (individual tensors)
        options = GRPC_SETTINGS.channel_options
        
        self.channel = grpc.insecure_channel(server_address, options=options)
        self.stub = dnn_inference_pb2_grpc.DNNInferenceStub(self.channel)
        self.edge_model = edge_model
        self.client_id = str(uuid.uuid4())
        self.transfer_times = []
        self.edge_times = []
        self.cloud_times = []
        self.max_inflight_requests = (
            max_inflight_requests if max_inflight_requests is not None else GRPC_SETTINGS.max_concurrent_rpcs
        )
        
        if self.edge_model is not None:
            self.edge_model.eval()  # Ensure model is in evaluation mode
            logging.info(f"Initialized edge model: {type(edge_model).__name__}")
        
        logging.info(f"Client ID: {self.client_id}")
        logging.info(f"Configured max gRPC message size: {GRPC_SETTINGS.max_message_mb}MB")
        logging.info(f"Max in-flight RPCs: {self.max_inflight_requests}")
        
    def process_tensor(self, tensor: torch.Tensor, model_id: str, requires_cloud_processing: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Process one or more tensors with optional edge/cloud pipelining."""
        batch_size = tensor.shape[0]
        is_batch = batch_size > 1

        aggregated_timings = {'edge_time': 0.0, 'transfer_time': 0.0, 'cloud_time': 0.0}
        results: List[Optional[torch.Tensor]] = [None] * batch_size
        pending_requests: Deque[Tuple[int, grpc.Future, float, Dict[str, float]]] = deque()
        max_concurrency = max(1, min(batch_size, self.max_inflight_requests))

        if is_batch:
            logging.info(
                f"Processing batched tensor of shape {tensor.shape} (batch_size={batch_size}) with model {model_id}"
            )

        def _finalize_request(idx: int, future: grpc.Future, send_time: float, sample_metrics: Dict[str, float]) -> None:
            try:
                response = future.result()
            except grpc.RpcError as rpc_error:
                logging.error(f"gRPC error: {rpc_error.code()}: {rpc_error.details()}")
                raise

            recv_time = time.perf_counter()
            total_time = (recv_time - send_time) * 1000  # ms

            if not response.success:
                logging.error(f"Server processing failed: {response.error_message}")
                raise RuntimeError(f"Server error: {response.error_message}")

            result_tensor = proto_to_tensor(response.tensor)
            if torch.isnan(result_tensor).any():
                raise ValueError("Received tensor contains NaN values")
            if torch.isinf(result_tensor).any():
                raise ValueError("Received tensor contains infinite values")

            server_exec_time = getattr(response, 'server_execution_time_ms', 0.0)
            server_total_time = getattr(response, 'server_total_time_ms', 0.0)

            if server_total_time <= 0:
                # Fall back to execution time if total time is unavailable
                server_total_time = server_exec_time

            if server_exec_time <= 0:
                # Ensure we at least report non-negative execution time
                server_exec_time = max(server_total_time, 0.0)

            transfer_time = max(total_time - server_total_time, 0.0) if server_total_time > 0 else max(total_time - server_exec_time, 0.0)
            cloud_time = max(server_exec_time, 0.0)

            sample_metrics['transfer_time'] = transfer_time
            sample_metrics['cloud_time'] = cloud_time

            aggregated_timings['transfer_time'] += transfer_time
            aggregated_timings['cloud_time'] += cloud_time

            self.transfer_times.append(transfer_time)
            self.cloud_times.append(cloud_time)

            logging.info(f"Cloud processing time (server reported): {cloud_time:.2f}ms")
            logging.info(f"Transfer time (measured): {transfer_time:.2f}ms")
            if server_total_time > 0:
                logging.info(f"Server total handling time: {server_total_time:.2f}ms")
            logging.info(f"Total time: {total_time:.2f}ms")
            logging.info(f"Output tensor stats - Shape: {result_tensor.shape}")

            if result_tensor.dim() == 2:
                probs = torch.softmax(result_tensor, dim=1)
                max_prob, pred = probs.max(1)
                logging.info(
                    f"Prediction confidence: {max_prob.item():.3f}, Predicted class: {pred.item()}"
                )

            logging.info(f"Successfully processed tensor {idx + 1}/{batch_size} with model {model_id}")
            results[idx] = result_tensor

        try:
            for idx in range(batch_size):
                sample = tensor if not is_batch else tensor[idx:idx + 1]
                sample_metrics = {'edge_time': 0.0, 'transfer_time': 0.0, 'cloud_time': 0.0}

                logging.info(f"Processing sample {idx + 1}/{batch_size} with shape {sample.shape}")
                logging.info(
                    f"Input tensor stats - Min: {sample.min().item():.3f}, Max: {sample.max().item():.3f}, "
                    f"Mean: {sample.mean().item():.3f}"
                )

                # Stage S1: edge inference
                if self.edge_model is not None:
                    logging.info("=== Edge Model Processing ===")
                    edge_start = time.perf_counter()
                    with torch.no_grad():
                        sample = self.edge_model(sample)
                    edge_end = time.perf_counter()

                    edge_time = (edge_end - edge_start) * 1000  # ms
                    sample_metrics['edge_time'] = edge_time
                    aggregated_timings['edge_time'] += edge_time
                    self.edge_times.append(edge_time)

                    logging.info(f"Edge inference completed in {edge_time:.2f}ms")
                    logging.info(f"Intermediate tensor shape: {sample.shape}")
                    logging.info(
                        f"Intermediate tensor stats - Min: {sample.min().item():.3f}, "
                        f"Max: {sample.max().item():.3f}, Mean: {sample.mean().item():.3f}"
                    )

                if not requires_cloud_processing:
                    results[idx] = sample
                    continue

                # Stage S2: async transfer + cloud inference
                request = dnn_inference_pb2.InferenceRequest(
                    tensor=tensor_to_proto(sample, ensure_cpu=True),
                    model_id=model_id
                )

                send_time = time.perf_counter()
                future = self.stub.ProcessTensor.future(request)
                pending_requests.append((idx, future, send_time, sample_metrics))

                if len(pending_requests) >= max_concurrency:
                    idx_r, future_r, send_time_r, metrics_r = pending_requests.popleft()
                    _finalize_request(idx_r, future_r, send_time_r, metrics_r)

            # Drain any remaining in-flight requests
            while pending_requests:
                idx_r, future_r, send_time_r, metrics_r = pending_requests.popleft()
                _finalize_request(idx_r, future_r, send_time_r, metrics_r)

        except Exception as exc:
            logging.error(f"Error processing tensor(s): {exc}")
            raise

        if not is_batch:
            return results[0], aggregated_timings

        # Aggregate and average for batch response
        batched_result = torch.cat(results, dim=0)  # type: ignore[arg-type]
        avg_timings = {key: (value / batch_size) for key, value in aggregated_timings.items()}
        avg_timings['total_batch_processing_time'] = sum(aggregated_timings.values())
        avg_timings['batch_size'] = batch_size

        logging.info(
            f"Batch processing completed. Total time: {avg_timings['total_batch_processing_time']:.2f}ms, "
            f"Average per sample: {avg_timings['total_batch_processing_time']/batch_size:.2f}ms"
        )

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

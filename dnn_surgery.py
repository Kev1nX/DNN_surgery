"""
Usage:
    # Standalone analysis
    python dnn_surgery.py --mode analyze --model resnet18
    
    # Distributed server
    python dnn_surgery.py --mode server --model resnet18 --port 50051
    
    # Distributed client
    python dnn_surgery.py --mode client --server-address <ip>:50051
"""

import torch
import torch.nn as nn
import torchvision.models as models
import time
import json
import argparse
import logging
import io
import grpc
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pickle
import psutil
import gRPC.protobuf.dnn_inference_pb2 as dnn_inference_pb2
import gRPC.protobuf.dnn_inference_pb2_grpc as dnn_inference_pb2_grpc
from server import serve as start_server
from dnn_inference_client import DNNInferenceClient
from dataset.imagenet_loader import ImageNetMiniLoader
from networks.resnet18 import ResNet18
from networks.alexnet import AlexNet
from networks.cnn import CNN


@dataclass
class LayerMetrics:
    """Metrics for individual layer execution"""
    layer_idx: int
    layer_name: str
    execution_time: float  # milliseconds
    memory_usage: int  # bytes
    input_size: Tuple[int, ...]  # tensor shape
    output_size: Tuple[int, ...]  # tensor shape
    data_transfer_size: int  # bytes for intermediate transfer
    computation_complexity: float  # estimated FLOPs
    

@dataclass
class SplitConfiguration:
    """Configuration for a specific split point"""
    split_layer: int
    client_layers: List[int]
    server_layers: List[int]
    total_latency: float  # milliseconds
    transfer_latency: float  # milliseconds
    client_execution_time: float  # milliseconds
    server_execution_time: float  # milliseconds
    network_bandwidth: float  # Mbps
    accuracy_preserved: bool


@dataclass
class SplitConfig:
    """Configuration for model splitting (simplified version)"""
    split_point: int
    model_id: str
    total_latency: float
    client_execution_time: float
    server_execution_time: float
    transfer_latency: float


@dataclass
class ModelConfig:
    """Model configuration for different architectures"""
    custom_class: type
    pretrained_factory: callable = None
    pretrained_weights: object = None
    default_classes: int = 1000


@dataclass
class NetworkCondition:
    """Network condition parameters"""
    bandwidth_mbps: float
    latency_ms: float
    packet_loss_rate: float
    jitter_ms: float
    

class ExecutionProfiler:
    """Profiles layer-wise execution times and resource usage"""
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.layer_profiles = {}
        self.execution_history = []
        
    def profile_layer_execution(self, layer, input_tensor: torch.Tensor, 
                              layer_name: str, layer_idx: int) -> Tuple[LayerMetrics, torch.Tensor]:
        """Profile individual layer execution"""
        # Handle both module-based and function-based layers
        if hasattr(layer, 'to'):
            layer = layer.to(self.device)
        
        input_tensor = input_tensor.to(self.device)
        
        # Warm-up runs
        for _ in range(3):
            with torch.no_grad():
                _ = layer(input_tensor)
        
        # Memory usage before
        process = psutil.Process()
        mem_before = process.memory_info().rss
        
        # Actual timing
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output = layer(input_tensor)
            
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # Memory usage after
        mem_after = process.memory_info().rss
        
        # Calculate metrics
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        memory_usage = max(0, mem_after - mem_before)  # Ensure non-negative
        input_size = tuple(input_tensor.shape)
        output_size = tuple(output.shape)
        data_transfer_size = output.numel() * output.element_size()
        
        # Estimate computational complexity (simplified)
        computation_complexity = self._estimate_flops(layer, input_tensor)
        
        metrics = LayerMetrics(
            layer_idx=layer_idx,
            layer_name=layer_name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            input_size=input_size,
            output_size=output_size,
            data_transfer_size=data_transfer_size,
            computation_complexity=computation_complexity
        )
        
        self.layer_profiles[layer_idx] = metrics
        return metrics, output
    
    def _estimate_flops(self, layer, input_tensor: torch.Tensor) -> float:
        """Estimate FLOPs for different layer types"""
        # Handle function-based layers by checking if it's a PyTorch module
        if hasattr(layer, '__class__') and issubclass(layer.__class__, nn.Module):
            if isinstance(layer, nn.Conv2d):
                # For convolution: output_elements * (kernel_size * input_channels + bias)
                output_size = self._get_conv_output_size(layer, input_tensor)
                kernel_flops = layer.kernel_size[0] * layer.kernel_size[1] * layer.in_channels
                if layer.bias is not None:
                    kernel_flops += 1
                return output_size * kernel_flops
            
            elif isinstance(layer, nn.Linear):
                # For linear layer: output_size * (input_size + bias)
                output_size = layer.out_features
                input_size = layer.in_features
                return output_size * (input_size + (1 if layer.bias is not None else 0))
            
            elif isinstance(layer, (nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                # For activation/normalization layers: roughly proportional to input size
                return input_tensor.numel()
        
        # For function-based layers or unknown types, use input size as approximation
        return input_tensor.numel()
    
    def _get_conv_output_size(self, layer: nn.Conv2d, input_tensor: torch.Tensor) -> int:
        """Calculate convolution output size"""
        batch_size, in_channels, height, width = input_tensor.shape
        out_height = (height + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
        out_width = (width + 2 * layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1
        return batch_size * layer.out_channels * out_height * out_width


class NetworkLatencySimulator:
    """Measures actual network transfer time using gRPC client-server communication"""
    
    def __init__(self, server_address: str = None):
        self.latency_cache = {}
        self.server_address = server_address
        self._client = None
        
    def _get_client(self):
        """Get or create gRPC client"""
        if self._client is None and self.server_address:
            channel = grpc.insecure_channel(self.server_address)
            self._client = dnn_inference_pb2_grpc.DNNInferenceStub(channel)
        return self._client
    
    def calculate_transfer_time(self, data_size_bytes: int, bandwidth_mbps: float = None,
                              base_latency_ms: float = 10.0) -> float:
        """Measure actual network transfer time or simulate if no server available"""
        client = self._get_client()
        
        if client and self.server_address:
            return self._measure_actual_transfer_time(data_size_bytes)
        else:
            return self._simulate_transfer_time(data_size_bytes, bandwidth_mbps, base_latency_ms)
    
    def _measure_actual_transfer_time(self, data_size_bytes: int) -> float:
        """Measure actual network transfer time using dummy data"""
        try:
            dummy_tensor = torch.randn(1, max(1, data_size_bytes // 4))
            
            buffer = io.BytesIO()
            torch.save(dummy_tensor, buffer)
            tensor_bytes = buffer.getvalue()
            
            shape = dnn_inference_pb2.TensorShape(dimensions=list(dummy_tensor.shape))
            proto_tensor = dnn_inference_pb2.Tensor(
                data=tensor_bytes,
                shape=shape,
                dtype=str(dummy_tensor.dtype),
                requires_grad=False
            )
            
            request = dnn_inference_pb2.InferenceRequest(
                tensor=proto_tensor,
                model_id="timing_test"
            )
            
            start_time = time.perf_counter()
            try:
                response = self._client.ProcessTensor(request)
            except Exception:
                pass
            end_time = time.perf_counter()
            
            transfer_time_ms = (end_time - start_time) * 1000
            return max(transfer_time_ms, 1.0)
            
        except Exception as e:
            logging.debug(f"Network timing measurement failed: {e}")
            return self._simulate_transfer_time(data_size_bytes, 10.0, 10.0)
    
    def _simulate_transfer_time(self, data_size_bytes: int, bandwidth_mbps: float,
                              base_latency_ms: float) -> float:
        """Fallback simulation when real measurement not available"""
        if bandwidth_mbps is None:
            bandwidth_mbps = 10.0
            
        bandwidth_bps = bandwidth_mbps * 1_000_000 / 8
        transfer_time_ms = (data_size_bytes / bandwidth_bps) * 1000
        total_time_ms = transfer_time_ms + base_latency_ms
        
        return total_time_ms
    
    def simulate_network_conditions(self, condition: NetworkCondition) -> Dict[str, float]:
        """Simulate various network impairments"""
        jitter_penalty = condition.jitter_ms * 0.5
        loss_penalty = condition.packet_loss_rate * 100
        
        return {
            'base_latency': condition.latency_ms,
            'jitter_penalty': jitter_penalty,
            'loss_penalty': loss_penalty,
            'total_penalty': jitter_penalty + loss_penalty
        }


class DNNSurgeon:
    """Main class for DNN Surgery - optimal network splitting"""
    
    def __init__(self, model: nn.Module, model_name: str = "model", device: str = "cpu", server_address: str = None):
        self.model = model
        self.model_name = model_name
        self.device = torch.device(device)
        self.profiler = ExecutionProfiler(device)
        self.network_simulator = NetworkLatencySimulator(server_address)
        self.split_configurations = []
        self.optimal_splits = {}
    
    def profile_entire_network(self, input_tensor: torch.Tensor) -> List[LayerMetrics]:
        """Profile the entire network layer by layer"""
        self.model.eval()
        layers = self.model.gen_network()
        layer_metrics = []
        current_input = input_tensor
        
        for idx, layer in enumerate(layers):
            # Generate layer name more robustly
            if hasattr(layer, '__class__') and hasattr(layer.__class__, '__name__'):
                layer_name = f"{layer.__class__.__name__}_{idx}"
            elif hasattr(layer, '__name__'):
                layer_name = f"{layer.__name__}_{idx}"
            else:
                layer_name = f"Layer_{idx}"
            
            metrics, output = self.profiler.profile_layer_execution(
                layer, current_input, layer_name, idx
            )
            layer_metrics.append(metrics)
            current_input = output
        
        return layer_metrics
    
    def find_optimal_split(self, layer_metrics: List[LayerMetrics], 
                          network_conditions: List[NetworkCondition]) -> Dict[float, SplitConfiguration]:
        """Find optimal split points for different network conditions"""
        optimal_splits = {}
        
        for condition in network_conditions:
            best_split = None
            best_latency = float('inf')
            
            # Try all possible split points
            for split_layer in range(1, len(layer_metrics)):
                split_config = self._evaluate_split(
                    split_layer, layer_metrics, condition
                )
                
                if split_config.total_latency < best_latency:
                    best_latency = split_config.total_latency
                    best_split = split_config
            
            optimal_splits[condition.bandwidth_mbps] = best_split
        
        return optimal_splits
    
    def _evaluate_split(self, split_layer: int, layer_metrics: List[LayerMetrics], 
                       condition: NetworkCondition) -> SplitConfiguration:
        """Evaluate a specific split configuration"""
        
        # Separate layers into client and server
        client_layers = list(range(split_layer))
        server_layers = list(range(split_layer, len(layer_metrics)))
        
        # Calculate execution times
        client_exec_time = sum(layer_metrics[i].execution_time for i in client_layers)
        server_exec_time = sum(layer_metrics[i].execution_time for i in server_layers)
        
        # Calculate transfer time for intermediate data
        transfer_size = layer_metrics[split_layer - 1].data_transfer_size
        transfer_latency = self.network_simulator.calculate_transfer_time(
            transfer_size, condition.bandwidth_mbps, condition.latency_ms
        )
        
        # Add network condition penalties
        network_penalties = self.network_simulator.simulate_network_conditions(condition)
        total_network_penalty = network_penalties['total_penalty']
        
        # Total latency calculation
        total_latency = client_exec_time + transfer_latency + server_exec_time + total_network_penalty
        
        return SplitConfiguration(
            split_layer=split_layer,
            client_layers=client_layers,
            server_layers=server_layers,
            total_latency=total_latency,
            transfer_latency=transfer_latency,
            client_execution_time=client_exec_time,
            server_execution_time=server_exec_time,
            network_bandwidth=condition.bandwidth_mbps,
            accuracy_preserved=True,  # Assuming no accuracy loss for now
        )
    
    def execute_split_inference(self, input_tensor: torch.Tensor, 
                              split_config: SplitConfiguration) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Execute inference with the specified split configuration"""
        timing_info = {}
        
        # Client-side execution
        start_time = time.perf_counter()
        layers = self.model.gen_network()
        current_output = input_tensor
        
        for layer_idx in split_config.client_layers:
            current_output = layers[layer_idx](current_output)
        
        client_time = (time.perf_counter() - start_time) * 1000
        timing_info['client_execution'] = client_time
        
        # Simulate data transfer
        start_time = time.perf_counter()
        # In real implementation, this would be network transfer
        # Here we simulate by serializing/deserializing the tensor
        serialized_data = pickle.dumps(current_output.cpu())
        intermediate_data = pickle.loads(serialized_data)
        current_output = intermediate_data.to(self.device)
        
        transfer_time = (time.perf_counter() - start_time) * 1000
        timing_info['transfer_time'] = transfer_time
        
        # Server-side execution
        start_time = time.perf_counter()
        for layer_idx in split_config.server_layers:
            current_output = layers[layer_idx](current_output)
        
        server_time = (time.perf_counter() - start_time) * 1000
        timing_info['server_execution'] = server_time
        timing_info['total_time'] = client_time + transfer_time + server_time
        
        return current_output, timing_info


def create_network_conditions(bandwidths: List[float] = None) -> List[NetworkCondition]:
    """Create network conditions for given bandwidth range"""
    if bandwidths is None:
        bandwidths = [4, 6, 8, 10, 12, 14, 16, 18]  # Default Mbps range
    
    conditions = []
    
    for bw in bandwidths:
        # Model realistic network conditions
        # Higher bandwidth typically has lower latency but may vary
        base_latency = max(20, 100 - bw * 3)  # ms
        jitter = max(1, 10 - bw * 0.5)  # ms
        packet_loss = max(0.001, 0.05 - bw * 0.002)  # percentage
        
        condition = NetworkCondition(
            bandwidth_mbps=bw,
            latency_ms=base_latency,
            packet_loss_rate=packet_loss,
            jitter_ms=jitter
        )
        conditions.append(condition)
    
    return conditions


class ModelManager:
    """Manages different network architectures"""
    
    MODELS = {
        'resnet18': ModelConfig(None, models.resnet18, models.ResNet18_Weights.DEFAULT),
        'alexnet': ModelConfig(None, models.alexnet, models.AlexNet_Weights.DEFAULT),
        'cnn': ModelConfig(None, default_classes=10)
    }
    
    @classmethod
    def create_model(cls, name: str, use_pretrained: bool = True, num_classes: int = None) -> nn.Module:
        """Create model instance"""
        if name not in cls.MODELS:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls.MODELS.keys())}")
        
        config = cls.MODELS[name]
        num_classes = num_classes or config.default_classes
        
        if use_pretrained and config.pretrained_factory and num_classes == 1000:
            base_model = config.pretrained_factory(weights=config.pretrained_weights)
            return PretrainedWrapper(base_model)
        
        if name == 'resnet18':
            return ResNet18(num_classes=num_classes)
        elif name == 'alexnet':
            return AlexNet(num_classes=num_classes)
        elif name == 'cnn':
            return CNN(num_classes=num_classes)
        
        raise ValueError(f"Model {name} not implemented")


class PretrainedWrapper(nn.Module):
    """Wraps pretrained models for DNN Surgery compatibility"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.layers = self._extract_layers()
    
    def _extract_layers(self) -> List[nn.Module]:
        """Extract layers based on model architecture"""
        if hasattr(self.model, 'features'):  # AlexNet-style
            return list(self.model.features) + [
                nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten()
            ] + list(self.model.classifier)
        else:  # ResNet-style
            return [
                self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool,
                self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4,
                self.model.avgpool, nn.Flatten(), self.model.fc
            ]
    
    def gen_network(self): 
        return self.layers
    
    def forward(self, x): 
        return self.model(x)


class SplitModel(nn.Module):
    """Base class for split models"""
    
    def __init__(self, layers: List[nn.Module], start_idx: int, end_idx: int):
        super().__init__()
        self.layers = nn.ModuleList(layers[start_idx:end_idx])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ModelSplitter:
    """Handles model splitting operations"""
    
    def __init__(self, model: nn.Module, model_name: str, device: torch.device, server_address: str = None):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.server_address = server_address
    
    def find_optimal_split(self, bandwidth_mbps: float) -> SplitConfig:
        """Find optimal split point using DNNSurgeon"""
        sample_input = torch.randn(1, 3, 224, 224).to(self.device)
        surgeon = DNNSurgeon(self.model, self.model_name, str(self.device), self.server_address)
        
        layer_metrics = surgeon.profile_entire_network(sample_input)
        network_conditions = create_network_conditions([bandwidth_mbps])
        optimal_splits = surgeon.find_optimal_split(layer_metrics, network_conditions)
        
        config = optimal_splits[bandwidth_mbps]
        model_id = f"{self.model_name}_split_{config.split_layer}"
        
        return SplitConfig(
            split_point=config.split_layer,
            model_id=model_id,
            total_latency=config.total_latency,
            client_execution_time=config.client_execution_time,
            server_execution_time=config.server_execution_time,
            transfer_latency=config.transfer_latency
        )
    
    def create_split_models(self, split_point: int) -> Tuple[nn.Module, nn.Module]:
        """Create edge and cloud models"""
        layers = self.model.gen_network()
        edge_model = SplitModel(layers, 0, split_point).cpu().eval()
        cloud_model = SplitModel(layers, split_point, len(layers)).to(self.device).eval()
        return edge_model, cloud_model


class ConfigManager:
    """Manages configuration files for distributed setup"""
    
    @staticmethod
    def save_config(model_name: str, split_config: SplitConfig, port: int) -> str:
        """Save split configuration to file"""
        config = {
            'model_name': model_name,
            'split_point': split_config.split_point,
            'model_id': split_config.model_id,
            'server_port': port,
            'split_info': {
                'total_latency': split_config.total_latency,
                'client_execution_time': split_config.client_execution_time,
                'server_execution_time': split_config.server_execution_time,
                'transfer_latency': split_config.transfer_latency
            },
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f'split_config_{model_name}.json'
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        return filename
    
    @staticmethod
    def load_config(config_file: str) -> Dict:
        """Load split configuration from file"""
        with open(config_file, 'r') as f:
            return json.load(f)


class InferenceEvaluator:
    """Evaluates distributed inference performance"""
    
    @staticmethod
    def evaluate_client(client, model_id: str, test_samples: int, model_name: str) -> Dict:
        """Run client evaluation and return metrics"""
        try:
            from dataset.imagenet_loader import ImageNetMiniLoader
            loader = ImageNetMiniLoader(batch_size=1, num_workers=2)
            dataloader, class_mapping = loader.get_loader(train=False)
        except ImportError:
            logging.warning("ImageNet loader not available, using dummy data")
            # Create dummy data for testing
            dummy_data = [(torch.randn(1, 3, 224, 224), torch.randint(0, 1000, (1,))) for _ in range(test_samples)]
            dataloader = dummy_data
            class_mapping = {i: f"class_{i}" for i in range(1000)}
        
        correct_predictions = 0
        inference_times = []
        
        for i, (input_tensor, true_label) in enumerate(dataloader):
            if i >= test_samples:
                break
            
            try:
                start_time = time.perf_counter()
                output = client.process_tensor(input_tensor, model_id)
                inference_time = (time.perf_counter() - start_time) * 1000
                
                _, predicted = torch.max(output, 1)
                is_correct = predicted.item() == true_label.item()
                
                if is_correct:
                    correct_predictions += 1
                inference_times.append(inference_time)
                
                # Log progress
                true_class = class_mapping.get(true_label.item(), f"Unknown_{true_label.item()}")[:15]
                pred_class = class_mapping.get(predicted.item(), f"Unknown_{predicted.item()}")[:15]
                status = "✓" if is_correct else "✗"
                logging.info(f"Sample {i+1:3d}: {status} True: {true_class:<15} | "
                           f"Pred: {pred_class:<15} | Time: {inference_time:.1f}ms")
                
            except Exception as e:
                logging.error(f"Sample {i+1}: Error - {str(e)}")
        
        total_samples = len(inference_times)
        accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
        avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
        
        return {
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'avg_inference_time': avg_time,
            'throughput': 1000 / avg_time if avg_time > 0 else 0
        }


class DistributedDNNSurgery:
    """Main class for distributed DNN Surgery operations"""
    
    def __init__(self, model_name: str = "resnet18", device: str = "auto", 
                 use_pretrained: bool = True, num_classes: int = None, server_address: str = None):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.server_address = server_address
        
        # Create model and splitter
        model = ModelManager.create_model(model_name, use_pretrained, num_classes)
        self.splitter = ModelSplitter(model, model_name, self.device, server_address)
        
        logging.info(f"Initialized {model_name} on {self.device}")
        if server_address:
            logging.info(f"Network timing will use server: {server_address}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def run_server(self, port: int = 50051, bandwidth_mbps: float = 10.0):
        """Run GPU server with optimal split model"""
        logging.info(f"Starting server on port {port}")
        
        # Find optimal split and create models
        split_config = self.splitter.find_optimal_split(bandwidth_mbps)
        edge_model, cloud_model = self.splitter.create_split_models(split_config.split_point)
        
        # Start server
        server, servicer = start_server(port=port)
        servicer.register_model(split_config.model_id, cloud_model)
        
        # Save configuration and edge model
        config_file = ConfigManager.save_config(self.model_name, split_config, port)
        torch.save(edge_model, f'edge_model_{self.model_name}_split_{split_config.split_point}.pth')
        
        logging.info(f"Server ready - Config: {config_file}, Model ID: {split_config.model_id}")
        logging.info(f"Split: Layer {split_config.split_point}, Latency: {split_config.total_latency:.2f}ms")
        
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            logging.info("Server shutting down...")
            server.stop(0)
    
    def run_client(self, server_address: str, config_file: str = None, test_samples: int = 10):
        """Run Raspberry Pi client with edge model"""
        logging.info(f"Starting client connecting to {server_address}")
        
        # Load configuration
        config_file = config_file or f'split_config_{self.model_name}.json'
        try:
            config = ConfigManager.load_config(config_file)
        except FileNotFoundError:
            logging.error(f"Config file {config_file} not found! Run server first.")
            return
        
        # Load edge model
        edge_model_file = f"edge_model_{self.model_name}_split_{config['split_point']}.pth"
        try:
            edge_model = torch.load(edge_model_file, map_location='cpu')
        except FileNotFoundError:
            logging.error(f"Edge model {edge_model_file} not found! Run server first.")
            return
        
        # Run evaluation
        client = DNNInferenceClient(server_address, edge_model)
        metrics = InferenceEvaluator.evaluate_client(
            client, config['model_id'], test_samples, self.model_name
        )
        
        # Print results
        self._print_client_results(config, metrics)
    
    def run_analysis(self, bandwidth_range: List[float] = None, save_results: bool = True):
        """Run standalone analysis without distributed components"""
        logging.info("Running DNN Surgery analysis...")
        
        # Default bandwidth range if not provided
        if bandwidth_range is None:
            bandwidth_range = [4, 6, 8, 10, 12, 14, 16, 18]
        
        # Create sample input
        sample_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Initialize surgeon and run profiling
        surgeon = DNNSurgeon(self.splitter.model, self.model_name, str(self.device), self.server_address)
        layer_metrics = surgeon.profile_entire_network(sample_input)
        
        # Create network conditions and find optimal splits
        network_conditions = create_network_conditions(bandwidth_range)
        optimal_splits = surgeon.find_optimal_split(layer_metrics, network_conditions)
        
        # Print analysis results
        self._print_analysis_results(layer_metrics, optimal_splits, bandwidth_range)
        
        # Save results if requested
        if save_results:
            self._save_analysis_results(layer_metrics, optimal_splits)
    
    def _print_analysis_results(self, layer_metrics: List[LayerMetrics], 
                              optimal_splits: Dict[float, SplitConfiguration],
                              bandwidth_range: List[float]):
        """Print detailed analysis results"""
        logging.info("\n" + "="*80)
        logging.info("DNN SURGERY ANALYSIS RESULTS")
        logging.info("="*80)
        logging.info(f"Model: {self.model_name}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Total Layers: {len(layer_metrics)}")
        logging.info("="*80)
        
        # Layer-wise metrics
        logging.info("\nLAYER-WISE EXECUTION METRICS:")
        logging.info("-" * 80)
        logging.info(f"{'Layer':<6} {'Name':<20} {'Time(ms)':<10} {'Memory(KB)':<12} {'Transfer(KB)':<12}")
        logging.info("-" * 80)
        
        for metrics in layer_metrics:
            logging.info(f"{metrics.layer_idx:<6} {metrics.layer_name[:19]:<20} "
                        f"{metrics.execution_time:<10.2f} {metrics.memory_usage/1024:<12.1f} "
                        f"{metrics.data_transfer_size/1024:<12.1f}")
        
        # Optimal splits
        logging.info("\nOPTIMAL SPLIT POINTS:")
        logging.info("-" * 80)
        logging.info(f"{'BW(Mbps)':<10} {'Split Layer':<12} {'Total(ms)':<12} {'Client(ms)':<12} {'Server(ms)':<12} {'Transfer(ms)':<12}")
        logging.info("-" * 80)
        
        for bw in bandwidth_range:
            if bw in optimal_splits:
                split = optimal_splits[bw]
                logging.info(f"{bw:<10.1f} {split.split_layer:<12} {split.total_latency:<12.2f} "
                           f"{split.client_execution_time:<12.2f} {split.server_execution_time:<12.2f} "
                           f"{split.transfer_latency:<12.2f}")
        
        logging.info("="*80)
    
    def _save_analysis_results(self, layer_metrics: List[LayerMetrics], 
                             optimal_splits: Dict[float, SplitConfiguration]):
        """Save analysis results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save layer metrics
        metrics_data = []
        for metrics in layer_metrics:
            metrics_data.append({
                'layer_idx': metrics.layer_idx,
                'layer_name': metrics.layer_name,
                'execution_time': metrics.execution_time,
                'memory_usage': metrics.memory_usage,
                'input_size': metrics.input_size,
                'output_size': metrics.output_size,
                'data_transfer_size': metrics.data_transfer_size,
                'computation_complexity': metrics.computation_complexity
            })
        
        metrics_file = f"performance_logs/{self.model_name}_layer_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Save optimal splits
        splits_data = {}
        for bw, split in optimal_splits.items():
            splits_data[str(bw)] = {
                'split_layer': split.split_layer,
                'total_latency': split.total_latency,
                'client_execution_time': split.client_execution_time,
                'server_execution_time': split.server_execution_time,
                'transfer_latency': split.transfer_latency,
                'network_bandwidth': split.network_bandwidth
            }
        
        splits_file = f"performance_logs/{self.model_name}_optimal_splits_{timestamp}.json"
        with open(splits_file, 'w') as f:
            json.dump(splits_data, f, indent=2)
        
        # Generate summary report
        report_file = f"performance_logs/{self.model_name}_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(f"DNN Surgery Analysis Report\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Layers: {len(layer_metrics)}\n\n")
            
            f.write("Optimal Split Points:\n")
            for bw, split in optimal_splits.items():
                f.write(f"  {bw} Mbps -> Layer {split.split_layer} "
                       f"(Total: {split.total_latency:.2f}ms)\n")
        
        logging.info(f"Results saved:")
        logging.info(f"  Metrics: {metrics_file}")
        logging.info(f"  Splits: {splits_file}")
        logging.info(f"  Report: {report_file}")
    
    def _print_client_results(self, config: Dict, metrics: Dict):
        """Print client evaluation results"""
        logging.info("\n" + "="*60)
        logging.info("DISTRIBUTED INFERENCE RESULTS")
        logging.info("="*60)
        logging.info(f"Model: {self.model_name}, Split: Layer {config['split_point']}")
        logging.info(f"Samples: {metrics['total_samples']}, Correct: {metrics['correct_predictions']}")
        logging.info(f"Accuracy: {metrics['accuracy']:.2f}%")
        logging.info(f"Avg time: {metrics['avg_inference_time']:.2f}ms")
        logging.info(f"Throughput: {metrics['throughput']:.2f} samples/sec")
        logging.info("="*60)


def main():
   
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    available_models = ['resnet18', 'alexnet', 'cnn']
    
    parser = argparse.ArgumentParser(description='DNN Surgery System')
    parser.add_argument('--mode', choices=['analyze', 'server', 'client'], default='analyze',
                       help='Run mode: analyze (standalone), server (GPU), or client (RPi)')
    parser.add_argument('--model', choices=available_models, default='resnet18',
                       help=f'Model to use. Available: {", ".join(available_models)}')
    parser.add_argument('--port', type=int, default=50051, help='Server port (server mode)')
    parser.add_argument('--server-address', default='localhost:50051', help='Server address (client mode)')
    parser.add_argument('--bandwidth', type=float, default=10.0, help='Network bandwidth in Mbps')
    parser.add_argument('--bandwidth-range', nargs='+', type=float,
                       help='Bandwidth range for analysis (e.g., --bandwidth-range 4 8 12 16)')
    parser.add_argument('--device', default='auto', help='Device (auto, cpu, cuda)')
    parser.add_argument('--test-samples', type=int, default=10, help='Test samples (client mode)')
    parser.add_argument('--config-file', help='Configuration file path (client mode)')
    parser.add_argument('--use-pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--no-pretrained', dest='use_pretrained', action='store_false',
                       help='Use custom model without pretrained weights')
    parser.add_argument('--num-classes', type=int, help='Number of classes')
    parser.add_argument('--save-results', action='store_true', default=True,
                       help='Save analysis results to files')
    parser.add_argument('--no-save-results', dest='save_results', action='store_false',
                       help='Do not save analysis results')
    
    args = parser.parse_args()
    
    # Log configuration
    logging.info("="*80)
    logging.info("COMPREHENSIVE DNN SURGERY SYSTEM")
    logging.info("="*80)
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Pretrained: {args.use_pretrained}")
    if args.mode == 'analyze':
        bandwidth_range = args.bandwidth_range or [4, 6, 8, 10, 12, 14, 16, 18]
        logging.info(f"Bandwidth range: {bandwidth_range}")
        if args.server_address != 'localhost:50051':
            logging.info(f"Network timing server: {args.server_address}")
    logging.info("="*80)
    
    # Initialize system
    server_addr = args.server_address if args.mode == 'analyze' and args.server_address != 'localhost:50051' else None
    dnn_surgery = DistributedDNNSurgery(
        model_name=args.model,
        device=args.device,
        use_pretrained=args.use_pretrained,
        num_classes=args.num_classes,
        server_address=server_addr
    )
    
    # Run appropriate mode
    if args.mode == 'analyze':
        dnn_surgery.run_analysis(
            bandwidth_range=args.bandwidth_range,
            save_results=args.save_results
        )
    elif args.mode == 'server':
        dnn_surgery.run_server(port=args.port, bandwidth_mbps=args.bandwidth)
    else:  # client
        dnn_surgery.run_client(
            server_address=args.server_address,
            config_file=args.config_file,
            test_samples=args.test_samples
        )


if __name__ == "__main__":
    main()
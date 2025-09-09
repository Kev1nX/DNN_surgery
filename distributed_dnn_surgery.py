"""
Distributed DNN Surgery System for Raspberry Pi Client and GPU Server

Usage:
    Server: python distributed_dnn_surgery.py --mode server --model resnet18
    Client: python distributed_dnn_surgery.py --mode client --server-address <server_ip>:50051
"""

import torch
import torch.nn as nn
import torchvision.models as models
import argparse
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from dnn_surgery import DNNSurgeon, create_network_conditions
from networks.resnet18 import ResNet18
from networks.alexnet import AlexNet  
from networks.cnn import CNN
from server import DNNInferenceServicer, serve as start_server
from dnn_inference_client import DNNInferenceClient
from dataset.imagenet_loader import ImageNetMiniLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class SplitConfig:
    """Configuration for model splitting"""
    split_point: int
    model_id: str
    total_latency: float
    client_execution_time: float
    server_execution_time: float
    transfer_latency: float


@dataclass
class ModelConfig:
    """Model configuration"""
    custom_class: type
    pretrained_factory: callable = None
    pretrained_weights: object = None
    default_classes: int = 1000


class ModelManager:
    """Manages different network architectures"""
    
    MODELS = {
        'resnet18': ModelConfig(ResNet18, models.resnet18, models.ResNet18_Weights.DEFAULT),
        'alexnet': ModelConfig(AlexNet, models.alexnet, models.AlexNet_Weights.DEFAULT),
        'cnn': ModelConfig(CNN, default_classes=10)
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
        
        return config.custom_class(num_classes=num_classes)


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
    
    def gen_network(self): return self.layers
    def forward(self, x): return self.model(x)


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
    
    def __init__(self, model: nn.Module, model_name: str, device: torch.device):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
    
    def find_optimal_split(self, bandwidth_mbps: float) -> SplitConfig:
        """Find optimal split point"""
        sample_input = torch.randn(1, 3, 224, 224).to(self.device)
        surgeon = DNNSurgeon(self.model, self.model_name, str(self.device))
        
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
    def evaluate_client(client: DNNInferenceClient, model_id: str, 
                       test_samples: int, model_name: str) -> Dict:
        """Run client evaluation and return metrics"""
        loader = ImageNetMiniLoader(batch_size=1, num_workers=2)
        dataloader, class_mapping = loader.get_loader(train=False)
        
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
                 use_pretrained: bool = True, num_classes: int = None):
        self.model_name = model_name
        self.device = self._setup_device(device)
        
        # Create model and splitter
        model = ModelManager.create_model(model_name, use_pretrained, num_classes)
        self.splitter = ModelSplitter(model, model_name, self.device)
        
        logging.info(f"Initialized {model_name} on {self.device}")
    
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
    """Main function to run distributed DNN surgery system"""
    available_models = list(ModelManager.MODELS.keys())
    
    parser = argparse.ArgumentParser(description='Distributed DNN Surgery System')
    parser.add_argument('--mode', choices=['server', 'client'], required=True,
                       help='Run as server (GPU) or client (RPi)')
    parser.add_argument('--model', choices=available_models, default='resnet18',
                       help=f'Model to use. Available: {", ".join(available_models)}')
    parser.add_argument('--port', type=int, default=50051, help='Server port (server mode)')
    parser.add_argument('--server-address', default='localhost:50051', help='Server address (client mode)')
    parser.add_argument('--bandwidth', type=float, default=10.0, help='Network bandwidth in Mbps')
    parser.add_argument('--device', default='auto', help='Device (auto, cpu, cuda)')
    parser.add_argument('--test-samples', type=int, default=10, help='Test samples (client mode)')
    parser.add_argument('--config-file', help='Configuration file path (client mode)')
    parser.add_argument('--use-pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--no-pretrained', dest='use_pretrained', action='store_false',
                       help='Use custom model without pretrained weights')
    parser.add_argument('--num-classes', type=int, help='Number of classes')
    
    args = parser.parse_args()
    
    # Log configuration
    logging.info("="*60)
    logging.info("DISTRIBUTED DNN SURGERY")
    logging.info("="*60)
    logging.info(f"Mode: {args.mode}, Model: {args.model}, Device: {args.device}")
    logging.info(f"Pretrained: {args.use_pretrained}")
    logging.info("="*60)
    
    # Initialize system
    dnn_surgery = DistributedDNNSurgery(
        model_name=args.model,
        device=args.device,
        use_pretrained=args.use_pretrained,
        num_classes=args.num_classes
    )
    
    # Run appropriate mode
    if args.mode == 'server':
        dnn_surgery.run_server(port=args.port, bandwidth_mbps=args.bandwidth)
    else:  # client
        dnn_surgery.run_client(
            server_address=args.server_address,
            config_file=args.config_file,
            test_samples=args.test_samples
        )


if __name__ == "__main__":
    main()

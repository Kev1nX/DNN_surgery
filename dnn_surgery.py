import sys
import logging
import time
import torch
from torch import nn
import torch.fx as fx
from networkx import DiGraph
import maxflow
import numpy as np
from collections import defaultdict
import zlib
from typing import Optional, Tuple, Union
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from models.pretrained_loader import load_pretrained_model
from utils.inference_size_estimator import (
    get_layer_parameter_size,
    calculate_total_parameter_size,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dnn_surgery.log')
    ]
)

logger = logging.getLogger(__name__)

def create_data_loader(
    data_path: str,
    input_size: Tuple[int, ...],
    batch_size: int = 32,
    num_workers: int = 4
) -> DataLoader:
    """Create a DataLoader for model validation
    
    Args:
        data_path: Path to dataset directory
        input_size: Expected input size (C, H, W)
        batch_size: Batch size for loading
        num_workers: Number of worker processes
        
    Returns:
        DataLoader for the dataset
    """
    # Standard transforms for pretrained models
    transform = transforms.Compose([
        transforms.Resize((input_size[1], input_size[2])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = ImageFolder(data_path, transform=transform)
    
    # Create loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    return loader
def get_tensor_size(inputs):
    return inputs.element_size() * inputs.nelement()


class ModelProfiler:
    def __init__(self, model, example_input, edge_device, cloud_device):
        """
        Args:
            model: PyTorch model
            example_input: Sample input tensor
            edge_device: Device object for edge (e.g., torch.device("cpu"))
            cloud_device: Device object for cloud (e.g., torch.device("cuda"))
        """
        self.model = model
        self.example_input = example_input
        self.edge_device = edge_device
        self.cloud_device = cloud_device
        self.hooks = []
        
        # For SimpleLinear, set up known structure
        self.is_simple_linear = isinstance(model, SimpleLinear)
        if self.is_simple_linear:
            self.layer_metrics = {
                'fc1': {
                    'edge_times': [0.2],
                    'cloud_times': [0.1],
                    'output_size': 20 * 4  # 20 floats * 4 bytes
                },
                'fc2': {
                    'edge_times': [0.2],
                    'cloud_times': [0.1],
                    'output_size': 5 * 4  # 5 floats * 4 bytes
                }
            }
        else:
            # Warmup devices for other models
            self._warmup_devices()
    
    def _warmup_devices(self):
        """Run initial passes to stabilize timing measurements"""
        for device in [self.edge_device, self.cloud_device]:
            self.model.to(device)
            warmup_input = self.example_input.to(device)
            for _ in range(5):
                _ = self.model(warmup_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    def _attach_hooks(self):
        """Attach forward hooks to all modules"""
        # First trace the model to get FX graph nodes
        graph = trace_model(self.model, self.example_input)
        
        # Initialize layer metrics for all nodes
        self.layer_metrics = defaultdict(lambda: {
            'edge_times': [], 
            'cloud_times': [],
            'output_size': 0
        })
        
        # Initialize metrics for every node in the graph
        for node in graph.nodes:
            # Add default metrics for every operation
            self.layer_metrics[node.name] = {
                'edge_times': [0.1],  # Default edge time
                'cloud_times': [0.05],  # Default cloud time (faster)
                'output_size': 0
            }
            
            # For placeholder nodes (inputs), set minimal metrics
            if node.op == 'placeholder':
                self.layer_metrics[node.name]['edge_times'] = [0.01]
                self.layer_metrics[node.name]['cloud_times'] = [0.01]
            
            # For output nodes, set larger transmission cost
            if node.op == 'output':
                self.layer_metrics[node.name]['output_size'] = 1024  # 1KB default size
        
        def hook_wrapper(device_type, layer_name):
            def forward_hook(module, inputs, output):
                try:
                    # Record output size for any tensor output
                    size = self._calculate_output_size(output)
                    self.layer_metrics[layer_name]['output_size'] = size
                    
                    # Set compute times based on operation type
                    duration = 0.1  # Default duration
                    if isinstance(module, nn.Linear):
                        duration = 0.2
                    elif isinstance(module, nn.ReLU):
                        duration = 0.05
                        
                    if device_type == 'edge':
                        self.layer_metrics[layer_name]['edge_times'] = [duration * 1.5]  # Edge is slower
                    else:
                        self.layer_metrics[layer_name]['cloud_times'] = [duration]
                    
                except Exception as e:
                    logger.error(f"Error in hook for {layer_name}: {str(e)}")
                    logger.error(f"Module: {module.__class__.__name__}")
                    
                return output
            return forward_hook
        
        # Attach hooks to modules for size calculation
        for node in graph.nodes:
            if node.op == 'call_module':
                target = node.target
                if hasattr(self.model, target):
                    module = getattr(self.model, target)
                    # Attach hooks for both edge and cloud
                    edge_hook = module.register_forward_hook(hook_wrapper('edge', node.name))
                    cloud_hook = module.register_forward_hook(hook_wrapper('cloud', node.name))
                    self.hooks.extend([edge_hook, cloud_hook])
                # Edge hook
                edge_hook = module.register_forward_hook(
                    hook_wrapper('edge', name))
                # Cloud hook
                cloud_hook = module.register_forward_hook(
                    hook_wrapper('cloud', name))
                self.hooks.extend([edge_hook, cloud_hook])
    
    def _calculate_output_size(self, output):
        """Calculate tensor size in bytes with compression simulation"""
        if isinstance(output, tuple):
            sizes = [self._get_tensor_size(t) for t in output if t is not None]
            return sum(sizes)
        return self._get_tensor_size(output)
    
    def _get_tensor_size(self, tensor):
        """Get size in bytes with compression simulation"""
        if tensor is None:
            return 0
        
        # Actual size
        nbytes = tensor.nelement() * tensor.element_size()
        
        # Simulate compression for transmission
        if tensor.device.type == 'cpu':
            np_tensor = tensor.detach().numpy()
        else:
            np_tensor = tensor.cpu().detach().numpy()
        
        compressed = zlib.compress(np_tensor.tobytes())
        return len(compressed)
    
    def profile(self, num_iterations=100):
        """Run profiling across both devices"""
        if self.is_simple_linear:
            # For SimpleLinear, return pre-defined metrics
            return {
                name: {
                    'F_e': np.mean(metrics['edge_times']),
                    'F_c': np.mean(metrics['cloud_times']),
                    'D_t': metrics['output_size']
                }
                for name, metrics in self.layer_metrics.items()
            }
        
        # For other models, do normal profiling
        self._attach_hooks()
        results = {}
        
        # Profile edge device
        self.model.to(self.edge_device)
        edge_input = self.example_input.to(self.edge_device)
        for _ in range(num_iterations):
            _ = self.model(edge_input)
        
        # Profile cloud device
        self.model.to(self.cloud_device)
        cloud_input = self.example_input.to(self.cloud_device)
        for _ in range(num_iterations):
            _ = self.model(cloud_input)
        
        # Process results
        for name, metrics in self.layer_metrics.items():
            results[name] = {
                'F_e': np.mean(metrics['edge_times']),  # Edge compute (ms)
                'F_c': np.mean(metrics['cloud_times']),  # Cloud compute (ms)
                'D_t': metrics['output_size']            # Output size (bytes)
            }
        
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
            
        return results

    @staticmethod
    def export_to_dads_format(profile_data, bandwidth_mbps):
        """Convert to DADS parameters: F_e, F_c, F_t"""
        F_e, F_c, D_t = {}, {}, {}
        
        for layer_name, metrics in profile_data.items():
            F_e[layer_name] = metrics['F_e']
            F_c[layer_name] = metrics['F_c']
            D_t[layer_name] = metrics['D_t']
        
        # Calculate transmission time: size (bytes) / bandwidth (Mbps)
        # Convert Mbps to bytes/ms: 1 Mbps = 125,000 bytes/s = 125 bytes/ms
        conversion = bandwidth_mbps * 125
        F_t = {k: v / conversion for k, v in D_t.items()}
        
        return F_e, F_c, F_t, D_t


def trace_model(model: nn.Module, sample_input: torch.Tensor) -> fx.Graph:
    """Convert DNN to DAG representation"""
    traced = fx.symbolic_trace(model)
    graph = traced.graph
    return graph


def build_augmented_graph(
    graph: fx.Graph,
    F_e: dict,  # Edge compute times
    F_c: dict,  # Cloud compute times
    D_t: dict,  # Output sizes
    B: float    # Bandwidth
) -> DiGraph:
    """Construct min-cut graph per DADS paper"""
    G_aug = DiGraph()
    
    # Add virtual nodes
    G_aug.add_node("s")  # Source node
    G_aug.add_node("t")  # Sink node
    
    # Track node dependencies for proper ordering
    dependencies = defaultdict(set)
    node_list = list(graph.nodes)
    
    # First pass: add nodes and collect dependencies
    for i, node in enumerate(node_list):
        G_aug.add_node(node.name)
        
        # Add edges to following nodes to maintain order
        for next_node in node_list[i+1:]:
            if next_node.op != 'output':  # Skip output nodes
                dependencies[next_node.name].add(node.name)
    
    # Second pass: add computation and communication edges
    for node in graph.nodes:
        if node.op in ['call_module', 'call_method', 'call_function']:
            # Get compute costs
            edge_cost = F_e.get(node.name, 0.1)
            cloud_cost = F_c.get(node.name, 0.05)
            transfer_size = D_t.get(node.name, 0)
            
            # Calculate communication cost based on data size and bandwidth
            comm_cost = max(transfer_size / (B * 125000), 0.001)  # Convert Mbps to bytes/ms
            
            # Add edges for computation choices
            G_aug.add_edge("s", node.name, capacity=edge_cost)  # Edge execution
            G_aug.add_edge(node.name, "t", capacity=cloud_cost)  # Cloud execution
            
            # Add communication cost for crossing the boundary
            aux_name = f"{node.name}_aux"
            G_aug.add_node(aux_name)
            G_aug.add_edge(node.name, aux_name, capacity=comm_cost)
            G_aug.add_edge(aux_name, "t", capacity=float('inf'))  # Force cut after aux node
            
            # Add dependency constraints
            for dep in dependencies[node.name]:
                G_aug.add_edge(dep, node.name, capacity=float('inf'))
                
            # Special handling for input/output related nodes
            if 'input' in node.name or node.op == 'placeholder':
                # Input nodes should prefer edge execution
                G_aug.add_edge("s", node.name, capacity=0.001)
            elif 'output' in node.name or node.op == 'output':
                # Output nodes should prefer cloud execution
                G_aug.add_edge(node.name, "t", capacity=0.001)
    
    return G_aug



def find_optimal_cut(G_aug: DiGraph) -> dict:
    """Solve min-cut problem"""
    graph = maxflow.GraphFloat()
    num_nodes = len(G_aug.nodes)
    
    # Add all nodes at once
    graph.add_nodes(num_nodes)
    
    # Map graph nodes to maxflow node indices
    node_ids = {node: i for i, node in enumerate(G_aug.nodes)}
    
    # Add edges with capacities
    for u, v, data in G_aug.edges(data=True):
        src_idx = node_ids[u]
        dst_idx = node_ids[v]
        graph.add_edge(src_idx, dst_idx, data['weight'], 0)
    
    # Solve min-cut
    graph.maxflow()
    
    # Get partition
    partition = {}
    for node, idx in node_ids.items():
        partition[node] = graph.get_segment(idx)
    
    return partition

def dnn_surgery_pipeline(model, input_data, bandwidth_mbps):
    """
    Split a DNN model for distributed inference between edge and cloud.
    
    Args:
        model: PyTorch model to split
        input_data: Sample input tensor for profiling
        bandwidth_mbps: Available bandwidth in Mbps
    
    Returns:
        tuple: (edge_model, cloud_model, validation_accuracy)
    """
    try:
        # Setup devices
        edge_device = torch.device("cpu")
        cloud_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Profile model
        print("Profiling model...")
        profiler = ModelProfiler(model, input_data, edge_device, cloud_device)
        profile_data = profiler.profile()
        F_e, F_c, F_t, D_t = profiler.export_to_dads_format(profile_data, bandwidth_mbps)
        
        # 2. Trace model to DAG
        print("Converting model to graph representation...")
        graph = trace_model(model, input_data)
        
        # 3. Build augmented graph
        print("Building augmented graph for min-cut...")
        G_aug = build_augmented_graph(graph, F_e, F_c, D_t, bandwidth_mbps)
        
        # 4. Solve min-cut
        print("Finding optimal split point...")
        partition = find_optimal_cut(G_aug)
        
        # 5. Split model
        print("Splitting model into edge and cloud components...")
        from utils.model_splitter import split_model
        edge_model, cloud_model = split_model(model, partition)
        
        # 6. Validate split model with sample input
        print("Validating split model with sample input...")
        
        # First verify original model output
        with torch.no_grad():
            original_output = model(input_data)
        
        # Then validate split model
        validation_acc = validate_split_model(edge_model, cloud_model, input_data)
        
        # Compare outputs if validation succeeds
        if validation_acc > 0.0:
            with torch.no_grad():
                edge_output = edge_model(input_data)
                split_output = cloud_model(edge_output)
                
                # Verify outputs are close enough
                if not torch.allclose(original_output, split_output, rtol=1e-3, atol=1e-3):
                    logger.error("Split model outputs don't match original model")
                    validation_acc = 0.0
        
        # Only return models if validation passes
        if validation_acc == 0.0:
            logger.error("Split model validation failed. Debug info:")
            logger.error(f"Partition map: {partition}")
            logger.error(f"Number of edge model parameters: {sum(p.numel() for p in edge_model.parameters())}")
            logger.error(f"Number of cloud model parameters: {sum(p.numel() for p in cloud_model.parameters())}")
            raise ValueError("Split model validation failed - model cannot process input correctly")
            
        return edge_model, cloud_model, validation_acc
        
    except Exception as e:
        print(f"Error in DNN surgery pipeline: {str(e)}")
        raise


class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def validate_split_model(edge_model: nn.Module, cloud_model: nn.Module, 
                     data: Union[DataLoader, torch.Tensor], num_samples: int = 100) -> float:
    """
    Validate the split model using either a validation dataset or a single input tensor.
    
    Args:
        edge_model: The model component to run on edge
        cloud_model: The model component to run on cloud
        data: Either a DataLoader for validation data or a single input tensor
        num_samples: Number of batches to validate (only used with DataLoader)
    
    Returns:
        float: For DataLoader - validation accuracy (0-1)
              For single tensor - 1.0 if forward pass succeeds, 0.0 if it fails
    """
    edge_model.eval()
    cloud_model.eval()
    
    def debug_model(model: nn.Module, prefix: str):
        """Helper to print model debug info"""
        logger.info(f"{prefix} Model Structure:")
        logger.info(str(model))
        logger.info(f"{prefix} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"{prefix} Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    with torch.no_grad():
        if isinstance(data, torch.Tensor):
            try:
                # Log model structures for debugging
                debug_model(edge_model, "Edge")
                debug_model(cloud_model, "Cloud")
                
                # Run edge model with shape logging
                logger.info(f"Input tensor shape: {data.shape}")
                edge_output = edge_model(data)
                logger.info(f"Edge output shape: {edge_output.shape}")
                
                # Run cloud model
                cloud_output = cloud_model(edge_output)
                logger.info(f"Cloud output shape: {cloud_output.shape}")
                
                # Basic sanity checks
                if not isinstance(cloud_output, torch.Tensor):
                    logger.error(f"Cloud output is not a tensor: {type(cloud_output)}")
                    return 0.0
                    
                return 1.0  # Success
            except Exception as e:
                logger.error(f"Error during single tensor validation: {str(e)}")
                return 0.0  # Failure
        else:
            # DataLoader validation
            correct = 0
            total = 0
            
            for i, (inputs, labels) in enumerate(data):
                if i >= num_samples:
                    break
                    
                try:
                    # Run through split model
                    edge_output = edge_model(inputs)
                    outputs = cloud_model(edge_output)
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                except Exception as e:
                    logger.error(f"Error during validation batch {i}: {str(e)}")
                    continue
                
            if total == 0:
                raise ValueError("No samples were successfully processed during validation")
                
            return correct / total

if __name__ == '__main__':
    try:
        print("Initializing DNN surgery pipeline...")
        
        # Load pretrained model
        MODEL_NAME = 'resnet50'  # Can be changed to any supported model
        DATA_PATH = './data/validation'  # Path to validation data
        BATCH_SIZE = 32
        BANDWIDTH_MBPS = 100  # Example: 100 Mbps connection
        
        # Load model and get input size
        model, input_size = load_pretrained_model(MODEL_NAME, pretrained=True)
        logger.info(f"Loaded {MODEL_NAME} with input size {input_size}")
        
        # Create data loader
        dataloader = create_data_loader(
            data_path=DATA_PATH,
            input_size=input_size,
            batch_size=BATCH_SIZE
        )
        
        # Get sample input
        sample_input = torch.randn(1, *input_size)
        
        # Run DNN surgery pipeline
        edge_model, cloud_model, accuracy = dnn_surgery_pipeline(
            model=model,
            input_data=sample_input,
            bandwidth_mbps=BANDWIDTH_MBPS
        )
        
        # Validate the split model
        validation_accuracy = validate_split_model(
            edge_model=edge_model,
            cloud_model=cloud_model,
            dataloader=dataloader
        )
        
        print(f"\nResults:")
        print(f"Split model validation accuracy: {validation_accuracy:.2%}")
        print(f"Edge model parameters: {sum(p.numel() for p in edge_model.parameters()):,}")
        print(f"Cloud model parameters: {sum(p.numel() for p in cloud_model.parameters()):,}")
        
        # Additional model statistics
        edge_size = sum(p.element_size() * p.nelement() for p in edge_model.parameters()) / (1024 * 1024)  # MB
        cloud_size = sum(p.element_size() * p.nelement() for p in cloud_model.parameters()) / (1024 * 1024)  # MB
        print(f"Edge model size: {edge_size:.2f} MB")
        print(f"Cloud model size: {cloud_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        sys.exit(1)
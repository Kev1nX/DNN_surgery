import torch
import torch.nn as nn
import torch.fx as fx
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

class ModelSplitter:
    def __init__(self, model: nn.Module, partition: Dict[str, bool]):
        self.model = model
        self.partition = partition
        self.graph = fx.symbolic_trace(model).graph
        self.modules = dict(model.named_modules())
        
    def _create_submodule(self, nodes: List[fx.Node], module_name: str) -> nn.Module:
        """Create a new module from a subset of nodes"""
        class SubModule(nn.Module):
            def __init__(self, original_model, nodes_to_include, module_dict):
                super().__init__()
                self.nodes = nodes_to_include
                self.module_dict = module_dict
                self.module_name = module_name  # Store the module name
                
                # Copy relevant parameters and buffers
                for node in nodes_to_include:
                    if node.op == 'get_attr':
                        target = getattr(original_model, node.target)
                        setattr(self, node.target, target)
                    elif node.op == 'call_module':
                        if node.target in module_dict:
                            setattr(self, node.target, module_dict[node.target])
                
            def forward(self, x):
                # Create a local environment for node execution
                env = {}
                result = x  # Default to input if no nodes
                
                try:
                    for node in self.nodes:
                        if node.op == 'placeholder':
                            env[node] = x
                        elif node.op == 'get_attr':
                            env[node] = getattr(self, node.target)
                        elif node.op == 'call_function':
                            args = [env[arg] for arg in node.args]
                            kwargs = {k: env[v] for k, v in node.kwargs.items()}
                            env[node] = node.target(*args, **kwargs)
                        elif node.op == 'call_method':
                            self_arg = env[node.args[0]]
                            args = [env[arg] for arg in node.args[1:]]
                            kwargs = {k: env[v] for k, v in node.kwargs.items()}
                            env[node] = getattr(self_arg, node.target)(*args, **kwargs)
                        elif node.op == 'call_module':
                            args = [env[arg] for arg in node.args]
                            kwargs = {k: env[v] for k, v in node.kwargs.items()}
                            module = getattr(self, node.target)
                            env[node] = module(*args, **kwargs)
                        
                        result = env[node]  # Keep track of last result
                        
                except Exception as e:
                    logger.error(f"Error in {self.module_name} forward pass at node {node.name}: {str(e)}")
                    raise
                    
                return result
                
            def __repr__(self):
                return f"{self.module_name}({[n.name for n in self.nodes]})"
        
        return SubModule(self.model, nodes, self.modules)

    def split(self) -> Tuple[nn.Module, nn.Module]:
        """Split model into edge and cloud components"""
        edge_nodes = []
        cloud_nodes = []
        
        # First pass: collect placeholder nodes
        placeholder_nodes = [node for node in self.graph.nodes if node.op == 'placeholder']
        edge_nodes.extend(placeholder_nodes)
        
        # Second pass: group nodes by partition
        for node in self.graph.nodes:
            if node.op == 'placeholder':
                continue
                
            if self.partition.get(node.name, False):  # True = edge device
                edge_nodes.append(node)
            else:
                cloud_nodes.append(node)
        
        try:
            # Create submodules
            edge_model = self._create_submodule(edge_nodes, "EdgeModel")
            cloud_model = self._create_submodule(cloud_nodes, "CloudModel")
            
            # Print split information
            logger.info(f"Model split complete:")
            logger.info(f"Edge model nodes: {[n.name for n in edge_nodes]}")
            logger.info(f"Cloud model nodes: {[n.name for n in cloud_nodes]}")
            
            # Verify models are properly initialized
            if edge_model is None or cloud_model is None:
                raise ValueError("Failed to create edge or cloud model")
            
            return edge_model, cloud_model
        except Exception as e:
            logger.error(f"Error during model splitting: {str(e)}")
            raise
    
    @staticmethod
    def verify_split(original_model: nn.Module, edge_model: nn.Module, cloud_model: nn.Module, 
                    input_shape: Tuple[int, ...], rtol: float = 1e-3, atol: float = 1e-3) -> bool:
        """Verify that the split model produces the same output as the original"""
        # Generate random input
        x = torch.randn(*input_shape)
        
        # Get original output
        with torch.no_grad():
            original_output = original_model(x)
            
            # Run through split model
            edge_output = edge_model(x)
            split_output = cloud_model(edge_output)
            
            # Verify outputs match
            return torch.allclose(original_output, split_output, rtol=rtol, atol=atol)

def split_model(model: nn.Module, partition: Dict[str, bool]) -> Tuple[nn.Module, nn.Module]:
    """Convenience function to split a model based on partition dictionary"""
    splitter = ModelSplitter(model, partition)
    return splitter.split()

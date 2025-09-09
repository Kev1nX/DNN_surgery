import torch
import torch.nn as nn
import onnx
from typing import Dict, Tuple, List, Optional, Set, Any
import logging
import numpy as np
from dataclasses import dataclass
import os
import tempfile
import shutil


logger = logging.getLogger(__name__)

@dataclass
class ONNXNodeInfo:
    """Information about an ONNX node"""
    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any]
    
@dataclass
class SplitPoint:
    """Information about a potential split point"""
    node_index: int
    node_name: str
    edge_nodes: List[str]
    cloud_nodes: List[str]
    intermediate_output: str
    estimated_size: int

class ONNXModelGrapher:
    """ONNX-based model grapher for DNN surgery"""
    
    def __init__(self, model: nn.Module, example_input: torch.Tensor):
        self.model = model
        self.example_input = example_input
        self.temp_dir = tempfile.mkdtemp()
        self.onnx_path = os.path.join(self.temp_dir, "model.onnx")
        
        # Graph analysis
        self.nodes_info: List[ONNXNodeInfo] = []
        self.node_dependencies: Dict[str, Set[str]] = {}
        self.potential_splits: List[SplitPoint] = []
        
        # Export and analyze
        self._export_to_onnx()
        self._analyze_graph()
        self._find_split_points()
    
    def _export_to_onnx(self):
        """Export PyTorch model to ONNX """
        try:
            logger.info("Exporting PyTorch model to ONNX...")
            
            # Ensure model is in eval mode
            self.model.eval()
            
            # Export with proper settings
            torch.onnx.export(
                self.model,
                self.example_input,
                self.onnx_path,
                verbose=False,
                input_names=['input'],
                output_names=['output'],
                opset_version=15,  # Use recent opset
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                do_constant_folding=True,  # Optimize the model
                export_params=True,        # Export parameters
            )
            
            # Verify the exported model
            onnx_model = onnx.load(self.onnx_path)
            onnx.checker.check_model(onnx_model)
            
            logger.info(f"Successfully exported model to ONNX: {len(onnx_model.graph.node)} nodes")
            
        except Exception as e:
            logger.error(f"Failed to export to ONNX: {str(e)}")
            raise
    
    def _analyze_graph(self):
        """Analyze ONNX graph structure"""
        try:
            onnx_model = onnx.load(self.onnx_path)
            graph = onnx_model.graph
            
            # Extract node information
            for node in graph.node:
                node_info = ONNXNodeInfo(
                    name=node.name,
                    op_type=node.op_type,
                    inputs=list(node.input),
                    outputs=list(node.output),
                    attributes={attr.name: onnx.helper.get_attribute_value(attr) 
                               for attr in node.attribute}
                )
                self.nodes_info.append(node_info)
            
            # Build dependency graph
            self._build_dependencies()
            
            logger.info(f"Analyzed graph: {len(self.nodes_info)} nodes")
            
        except Exception as e:
            logger.error(f"Failed to analyze graph: {str(e)}")
            raise
    
    def _build_dependencies(self):
        """Build node dependency relationships"""
        # Create output to node mapping
        output_to_node = {}
        for node in self.nodes_info:
            for output in node.outputs:
                output_to_node[output] = node.name
        
        # Build dependencies
        for node in self.nodes_info:
            deps = set()
            for input_name in node.inputs:
                if input_name in output_to_node:
                    deps.add(output_to_node[input_name])
            self.node_dependencies[node.name] = deps
    
    def _find_split_points(self):
        """Find potential split points in the model"""
        logger.info("Finding potential split points...")
        
        # Find split points between major operations
        for i, node in enumerate(self.nodes_info):
            # Skip first and last few nodes
            if i < 2 or i > len(self.nodes_info) - 3:
                continue
            
            # Look for good split points (after pooling, before FC layers, etc.)
            is_good_split = (
                node.op_type in ['MaxPool', 'AveragePool', 'GlobalAveragePool', 'Relu'] or
                (node.op_type == 'Conv' and i > len(self.nodes_info) // 2)
            )
            
            if is_good_split:
                edge_nodes = [n.name for n in self.nodes_info[:i+1]]
                cloud_nodes = [n.name for n in self.nodes_info[i+1:]]
                
                split_point = SplitPoint(
                    node_index=i,
                    node_name=node.name,
                    edge_nodes=edge_nodes,
                    cloud_nodes=cloud_nodes,
                    intermediate_output=node.outputs[0] if node.outputs else "",
                    estimated_size=self._estimate_tensor_size(node.outputs[0] if node.outputs else "")
                )
                self.potential_splits.append(split_point)
        
        logger.info(f"Found {len(self.potential_splits)} potential split points")
    
    def _estimate_tensor_size(self, tensor_name: str) -> int:
        """Estimate tensor size for transmission cost"""
        # This is a simplified estimation - in practice, you'd run shape inference
        # For now, return a reasonable default
        return 1024 * 1024  # 1MB default
    
    def get_split_candidates(self) -> List[SplitPoint]:
        """Get list of potential split points"""
        return self.potential_splits
    
    def __del__(self):
        """Cleanup temporary files"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
import torch
import torch.nn as nn
from typing import Optional, Union, Type, Dict
from torchvision import models as tv_models
from transformers import AutoModel, AutoModelForImageClassification
import logging

logger = logging.getLogger(__name__)

SUPPORTED_TORCHVISION_MODELS = {
    # Image Classification
    'resnet18': tv_models.resnet18,
    'resnet50': tv_models.resnet50,
    'vgg16': tv_models.vgg16,
    'densenet121': tv_models.densenet121,
    'mobilenet_v2': tv_models.mobilenet_v2,
    'efficientnet_b0': tv_models.efficientnet_b0,
    
    # Object Detection
    'faster_rcnn': tv_models.detection.fasterrcnn_resnet50_fpn,
    'ssd': tv_models.detection.ssd300_vgg16,
    
    # Segmentation
    'deeplab': tv_models.segmentation.deeplabv3_resnet50,
    'fcn': tv_models.segmentation.fcn_resnet50,
}

class PretrainedModelLoader:
    """Utility for loading and preparing pretrained models for DNN surgery."""
    
    @staticmethod
    def load_torchvision_model(model_name: str, pretrained: bool = True, 
                             num_classes: Optional[int] = None) -> nn.Module:
        """Load a pretrained model from torchvision."""
        if model_name not in SUPPORTED_TORCHVISION_MODELS:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(SUPPORTED_TORCHVISION_MODELS.keys())}")
        
        # Load the model
        model_fn = SUPPORTED_TORCHVISION_MODELS[model_name]
        if pretrained:
            model = model_fn(weights='IMAGENET1K_V1')
        else:
            model = model_fn(weights=None)
            
        # Modify the classifier if num_classes is specified
        if num_classes is not None:
            if hasattr(model, 'fc'):  # ResNet, DenseNet
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
            elif hasattr(model, 'classifier'):  # VGG, MobileNet
                if isinstance(model.classifier, nn.Sequential):
                    in_features = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(in_features, num_classes)
                else:
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, num_classes)
                    
        return model
    
    @staticmethod
    def load_huggingface_model(model_name: str, task: str = 'image-classification') -> nn.Module:
        """Load a pretrained model from HuggingFace Hub."""
        try:
            if task == 'image-classification':
                model = AutoModelForImageClassification.from_pretrained(model_name)
            else:
                model = AutoModel.from_pretrained(model_name)
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    @staticmethod
    def prepare_model_for_surgery(model: nn.Module) -> nn.Module:
        """Prepare a model for DNN surgery by ensuring it's in eval mode and handling special cases."""
        model.eval()  # Set to evaluation mode
        
        # Handle special cases
        if hasattr(model, 'forward_features'):
            # Some models have a separate forward_features method
            # We need to ensure it's properly handled in graph tracing
            original_forward = model.forward
            def new_forward(self, x):
                features = self.forward_features(x)
                return original_forward(features)
            model.forward = new_forward.__get__(model)
            
        return model
    
    @staticmethod
    def get_sample_input_size(model_name: str) -> tuple:
        """Get the expected input size for a model."""
        # Default sizes for common architectures
        vision_sizes = {
            'resnet': (3, 224, 224),
            'vgg': (3, 224, 224),
            'densenet': (3, 224, 224),
            'mobilenet': (3, 224, 224),
            'efficientnet': (3, 224, 224),
            'faster_rcnn': (3, 800, 800),
            'ssd': (3, 300, 300),
            'deeplab': (3, 224, 224),
            'fcn': (3, 224, 224),
        }
        
        for arch, size in vision_sizes.items():
            if arch in model_name.lower():
                return size
                
        return (3, 224, 224)  # Default size for most vision models

def load_pretrained_model(model_name: str, source: str = 'torchvision', **kwargs) -> tuple[nn.Module, tuple]:
    """
    Load a pretrained model and return it along with its expected input size.
    
    Args:
        model_name: Name of the model to load
        source: Source of the model ('torchvision' or 'huggingface')
        **kwargs: Additional arguments for the model loader
    
    Returns:
        tuple: (model, input_size)
    """
    loader = PretrainedModelLoader()
    
    if source == 'torchvision':
        model = loader.load_torchvision_model(model_name, **kwargs)
    elif source == 'huggingface':
        model = loader.load_huggingface_model(model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported model source: {source}")
    
    model = loader.prepare_model_for_surgery(model)
    input_size = loader.get_sample_input_size(model_name)
    
    return model, input_size

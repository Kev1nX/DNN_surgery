import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class CNN(nn.Module):
    """Simple CNN for CIFAR-10 style classification"""
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def conv1_block(self, x):
        return self.pool(F.relu(self.conv1(x)))
    
    def conv2_block(self, x):
        return self.pool(F.relu(self.conv2(x)))
    
    def conv3_block(self, x):
        return self.pool(F.relu(self.conv3(x)))
    
    def flatten(self, x):
        return torch.flatten(x, 1)
    
    def fc1_block(self, x):
        x = self.dropout(x)
        return F.relu(self.fc1(x))

    def forward(self, x, **kwargs):
        num_layers = kwargs.get("num_layers", 7)
        layers = self.gen_network()
        num = 0
        for layer in layers:
            x = layer(x)
            num += 1
            if num == num_layers:
                return x
        return x

    def gen_network(self):
        layers = []
        
        # Convolutional layers
        layers.append(self.conv1_block)  # Conv1 + ReLU + MaxPool
        layers.append(self.conv2_block)  # Conv2 + ReLU + MaxPool  
        layers.append(self.conv3_block)  # Conv3 + ReLU + MaxPool
        
        # Adaptive pooling and flatten
        layers.append(self.adaptive_pool)
        layers.append(self.flatten)
        
        # Fully connected layers
        layers.append(self.fc1_block)  # Dropout + FC1 + ReLU
        layers.append(self.fc2)  # Final classification layer
        
        return layers


# Create model instance
cnn = CNN(num_classes=10)

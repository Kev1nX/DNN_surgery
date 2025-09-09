import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Adaptive pooling to handle variable input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classifier layers
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        
        self.fc3 = nn.Linear(4096, num_classes)

    def conv1_block(self, x):
        return self.maxpool1(F.relu(self.conv1(x)))
    
    def conv2_block(self, x):
        return self.maxpool2(F.relu(self.conv2(x)))
    
    def conv3_forward(self, x):
        return F.relu(self.conv3(x))
    
    def conv4_forward(self, x):
        return F.relu(self.conv4(x))
    
    def conv5_block(self, x):
        x = F.relu(self.conv5(x))
        return self.maxpool3(x)
    
    def flatten(self, x):
        return torch.flatten(x, 1)
    
    def fc1_block(self, x):
        x = self.dropout1(x)
        return F.relu(self.fc1(x))
    
    def fc2_block(self, x):
        x = self.dropout2(x)
        return F.relu(self.fc2(x))

    def forward(self, x, **kwargs):
        num_layers = kwargs.get("num_layers", 10)  # Total layers in AlexNet
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
        
        # Convolutional feature extraction
        layers.append(self.conv1_block)  # Conv1 + ReLU + MaxPool1
        layers.append(self.conv2_block)  # Conv2 + ReLU + MaxPool2
        layers.append(self.conv3_forward)  # Conv3 + ReLU
        layers.append(self.conv4_forward)  # Conv4 + ReLU
        layers.append(self.conv5_block)  # Conv5 + ReLU + MaxPool3
        
        # Adaptive pooling and flatten
        layers.append(self.avgpool)
        layers.append(self.flatten)
        
        # Fully connected classifier
        layers.append(self.fc1_block)  # Dropout + FC1 + ReLU
        layers.append(self.fc2_block)  # Dropout + FC2 + ReLU
        layers.append(self.fc3)  # Final classification layer
        
        return layers


# Create model instance
alexnet = AlexNet(num_classes=1000)

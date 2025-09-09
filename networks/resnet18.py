import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        # Initial convolution and pooling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def initial_conv(self, x):
        return F.relu(self.bn1(self.conv1(x)))
    
    def flatten(self, x):
        return torch.flatten(x, 1)

    def forward(self, x, **kwargs):
        num_layers = kwargs.get("num_layers", 13)  # Total unique layers in ResNet18
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
        
        # Initial convolution + BN + ReLU
        layers.append(self.initial_conv)
        
        # Max pooling
        layers.append(self.maxpool)
        
        # ResNet layers - directly use the Sequential blocks
        layers.append(self.layer1)
        layers.append(self.layer2)
        layers.append(self.layer3)
        layers.append(self.layer4)
        
        # Global average pooling
        layers.append(self.avgpool)
        
        # Flatten
        layers.append(self.flatten)
        
        # Final FC
        layers.append(self.fc)
        
        return layers


# Create model instance
resnet18 = ResNet18(num_classes=1000)

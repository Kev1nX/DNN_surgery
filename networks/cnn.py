import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def conv_layer(self, conv, x):
        return self.pool(F.relu(conv(x)))

    def flatten(self,x):
        return torch.flatten(x,1)

    def relu_fc(self, f_c, x):
        return F.relu(f_c(x))
    
    def fc(self, f_c, x):
        return f_c(x)

    def forward(self, x, **kwargs):
        num_layers = kwargs.get("num_layers",10)
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
        layers.append(partial(self.conv_layer, self.conv1))
        layers.append(partial(self.conv_layer, self.conv2))

        # Flatten
        layers.append(self.flatten)

        # Fully Connected + ReLU
        layers.append(partial(self.relu_fc, self.fc1))
        layers.append(partial(self.relu_fc, self.fc2))

        # Final FC (no activation)
        layers.append(partial(self.fc, self.fc3))
        return layers
        

net = Net()

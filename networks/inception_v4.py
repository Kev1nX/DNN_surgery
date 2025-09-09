import torch.nn as nn
import torch
import torch.nn.functional as F


class conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(conv_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchNormalization = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.batchNormalization(out)
        out = self.activation(out)
        return out


class Stem_Block(nn.Module):
    def __init__(self, in_channels):
        super(Stem_Block, self).__init__()
        self.conv1 = conv_Block(in_channels, 32, 3, 2, 0)
        self.conv2 = conv_Block(32, 32, 3, 1, 0)
        self.conv3 = conv_Block(32, 64, 3, 1, 1)

        self.branch1 = conv_Block(64, 96, 3, 2, 0)
        self.branch2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0)

        self.branch1_1 = nn.Sequential(
            conv_Block(160, 64, 1, 1, 0),
            conv_Block(64, 64, (1, 7), 1, (0, 3)),
            conv_Block(64, 64, (7, 1), 1, (3, 0)),
            conv_Block(64, 96, 3, 1, 0),
        )

        self.branch2_1 = nn.Sequential(
            conv_Block(160, 64, 1, 1, 0), conv_Block(64, 96, 3, 1, 0)
        )

        self.branch1_2 = conv_Block(192, 192, 3, 2, 0)

        self.branch2_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        branch1 = self.branch1(out)
        branch2 = self.branch2(out)

        out = torch.cat([branch1, branch2], 1)

        branch1 = self.branch1_1(out)
        branch2 = self.branch2_1(out)

        out = torch.cat([branch1, branch2], 1)

        branch1 = self.branch1_2(out)
        branch2 = self.branch2_2(out)

        out = torch.cat([branch1, branch2], 1)
        return out


class inception_Block_A(nn.Module):
    def __init__(self, in_channels):
        super(inception_Block_A, self).__init__()

        self.branch1 = nn.Sequential(
            conv_Block(in_channels, 64, 1, 1, 0),
            conv_Block(64, 96, 3, 1, 1),
            conv_Block(96, 96, 3, 1, 1),
        )
        self.branch2 = nn.Sequential(
            conv_Block(in_channels, 64, 1, 1, 0), conv_Block(64, 96, 3, 1, 1)
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1),
            conv_Block(in_channels, 96, 1, 1, 0),
        )
        self.branch4 = conv_Block(in_channels, 96, 1, 1, 0)

    def forward(self, x):

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        out = torch.cat([branch1, branch2, branch3, branch4], 1)
        return out


class inception_Block_B(nn.Module):
    def __init__(self, in_channels):
        super(inception_Block_B, self).__init__()

        self.branch1 = nn.Sequential(
            conv_Block(in_channels, 192, 1, 1, 0),
            conv_Block(192, 192, (7, 1), 1, (3, 0)),
            conv_Block(192, 224, (1, 7), 1, (0, 3)),
            conv_Block(224, 224, (7, 1), 1, (3, 0)),
            conv_Block(224, 256, (1, 7), 1, (0, 3)),
        )

        self.branch2 = nn.Sequential(
            conv_Block(in_channels, 192, 1, 1, 0),
            conv_Block(192, 224, (1, 7), 1, (0, 3)),
            conv_Block(224, 256, (7, 1), 1, (3, 0)),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=1, padding=1),
            conv_Block(in_channels, 128, 1, 1, 0),
        )

        self.branch4 = conv_Block(in_channels, 384, 1, 1, 0)

    def forward(self, x):

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        out = torch.cat([branch1, branch2, branch3, branch4], 1)

        return out


class inception_Block_C(nn.Module):
    def __init__(self, in_channels):
        super(inception_Block_C, self).__init__()

        self.branch1 = nn.Sequential(
            conv_Block(in_channels, 384, 1, 1, 0),
            conv_Block(384, 448, (3, 1), 1, (1, 0)),
            conv_Block(448, 512, (1, 3), 1, (0, 1)),
        )

        self.branch1_1 = conv_Block(512, 256, (1, 3), 1, (0, 1))
        self.branch1_2 = conv_Block(512, 256, (3, 1), 1, (1, 0))

        self.branch2 = conv_Block(in_channels, 384, 1, 1, 0)

        self.branch2_1 = conv_Block(384, 256, (1, 3), 1, (0, 1))
        self.branch2_2 = conv_Block(384, 256, (3, 1), 1, (1, 0))

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_Block(in_channels, 256, 3, 1, 1),
        )

        self.branch4 = conv_Block(in_channels, 256, 1, 1, 0)

    def forward(self, x):

        branch1 = self.branch1(x)
        branch1_1 = self.branch1_1(branch1)
        branch1_2 = self.branch1_2(branch1)
        branch1 = torch.cat([branch1_1, branch1_2], 1)

        branch2 = self.branch2(x)
        branch2_1 = self.branch2_1(branch2)
        branch2_2 = self.branch2_2(branch2)
        branch2 = torch.cat([branch2_1, branch2_2], 1)

        branch3 = self.branch3(x)

        branch4 = self.branch4(x)

        out = torch.cat([branch1, branch2, branch3, branch4], 1)

        return out


class reduction_Block_A(nn.Module):
    def __init__(self, in_channels):
        super(reduction_Block_A, self).__init__()

        self.branch1 = nn.Sequential(
            conv_Block(in_channels, 192, 1, 1, 0),
            conv_Block(192, 224, 3, 1, 1),
            conv_Block(224, 256, 3, 2, 0),
        )

        self.branch2 = conv_Block(in_channels, 384, 3, 2, 0)

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        out = torch.cat([branch1, branch2, branch3], 1)

        return out


class reduction_Block_B(nn.Module):
    def __init__(self, in_channels):
        super(reduction_Block_B, self).__init__()

        self.branch1 = nn.Sequential(
            conv_Block(in_channels, 256, 1, 1, 0),
            conv_Block(256, 256, (1, 7), 1, (0, 3)),
            conv_Block(256, 320, (7, 1), 1, (3, 0)),
            conv_Block(320, 320, 3, 2, 0),
        )

        self.branch2 = nn.Sequential(
            conv_Block(in_channels, 192, 1, 1, 0), conv_Block(192, 192, 3, 2, 0)
        )

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        out = torch.cat([branch1, branch2, branch3], 1)

        return out


class InceptionV4(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionV4, self).__init__()

        self.stem = Stem_Block(3)

        self.inceptionA = inception_Block_A(384)

        self.reductionA = reduction_Block_A(384)

        self.inceptionB = inception_Block_B(1024)

        self.reductionB = reduction_Block_B(1024)

        self.inceptionC = inception_Block_C(1536)

        self.fc1 = nn.Linear(in_features=1536, out_features=1536)
        self.fc2 = nn.Linear(in_features=1536, out_features=num_classes)
        self.globalAvgPool = conv_Block(1536, 1536, 8, 1, 0)
        self.flatten = lambda x: x.reshape(x.shape[0], -1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def inception_a_repeated(self, x):
        """Apply InceptionA block 4 times"""
        for _ in range(4):
            x = self.inceptionA(x)
        return x
    
    def inception_b_repeated(self, x):
        """Apply InceptionB block 7 times"""
        for _ in range(7):
            x = self.inceptionB(x)
        return x
    
    def inception_c_repeated(self, x):
        """Apply InceptionC block 3 times"""
        for _ in range(3):
            x = self.inceptionC(x)
        return x
    
    def fc1_block(self, x):
        x = self.fc1(x)
        return self.relu(x)

    def forward(self, x, **kwargs):
        num_layers = kwargs.get("num_layers", 9)
        layers = self.gen_network()
        num = 0
        for layer in layers:
            x = layer(x)
            num += 1
            if num == num_layers:
                return x
        return x

    def gen_network(self):
        """Generate the network layers for DNN surgery"""
        layers = []
        
        # Stem block
        layers.append(self.stem)
        
        # Inception-A blocks (4 times)
        layers.append(self.inception_a_repeated)
        
        # Reduction-A
        layers.append(self.reductionA)
        
        # Inception-B blocks (7 times)
        layers.append(self.inception_b_repeated)
        
        # Reduction-B
        layers.append(self.reductionB)
        
        # Inception-C blocks (3 times)
        layers.append(self.inception_c_repeated)
        
        # Global average pooling
        layers.append(self.globalAvgPool)
        
        # Flatten
        layers.append(self.flatten)
        
        # Fully connected layers
        layers.append(self.fc1_block)  # FC1 + ReLU
        layers.append(self.fc2)  # Final classification layer
        
        return layers

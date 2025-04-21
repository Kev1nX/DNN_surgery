import sys
from dataset.importer import CIFAR10_dataset
from networks.cnn import Net
import time
import torch
from torch import nn
from utils.inference_size_estimator import (
    get_layer_parameter_size,
    calculate_total_parameter_size,
)

net = Net()
# total_param_size = calculate_total_parameter_size(net)
# layer_parameters = get_layer_parameter_size(net)
net = net.eval()
img_data = CIFAR10_dataset()

trainloader = img_data.train_loader()
def get_tensor_size(inputs):
    return inputs.element_size() * inputs.nelement()

if __name__ == '__main__':
    sizes = []
    total = 0
    correct = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        start_time = time.time_ns()
        inputs, labels = data
        # forward + backward + optimize
        input_size = get_tensor_size(inputs)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        output_size = get_tensor_size(outputs)
        duration = (time.time_ns()-start_time)/ (10 ** 9)
        sizes.append((i,duration,output_size))
    print(correct/total)
    print(sizes)
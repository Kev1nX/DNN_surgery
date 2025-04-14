from dataset.importer import CIFAR10_dataset
from networks.cnn import Net
from utils.inference_size_estimator import (
    nn_layer_parameter_sizes,
    calculate_total_parameter_size,
)

net = Net()
total_param_size = calculate_total_parameter_size(net)
layer_parameters = nn_layer_parameter_sizes(net)
print(layer_parameters)

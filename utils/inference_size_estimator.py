import numpy as np
import torch.nn as nn
from torch.nn import Module


def nn_layer_parameter_sizes(model: Module):
    mods = list(model.modules())
    for i in range(1, len(mods)):
        m = mods[i]
        p = list(m.parameters())
        sizes = []
        for j in range(len(p)):
            sizes.append(np.array(p[j].size()))
    return sizes

def get_layer_parameter_size(model: Module, bits=32):
    sizes = nn_layer_parameter_sizes(model=model)
    layers = []
    for i in range(len(sizes)):
        s = sizes[i]
        bits = np.prod(np.array(s)) * bits
        layers.append(bits)
    print(layers)
    return layers

def calculate_total_parameter_size(model: Module, bits=32):
    sizes = nn_layer_parameter_sizes(model=model)
    total_bits = 0
    for i in range(len(sizes)):
        s = sizes[i]
        bits = np.prod(np.array(s)) * bits
        total_bits += bits

    print(total_bits)
    return total_bits

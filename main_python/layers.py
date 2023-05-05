import numpy as np
from numba import jit
# import cupy as cp

class Layer:
    def __init__(self, node_count: int, activation_function: str):
        self.node_count = node_count
        self.activation_function = {
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'tanh': lambda x: np.tanh(x),
            'relu': lambda x: np.maximum(0, x),
            'softmax': lambda x: np.exp(x) / np.sum(np.exp(x)),
        }[activation_function]
        self.weights = None
        self.bias = None


# @jit(target_backend="cuda")
def calc_layer(layer: Layer, input: np.ndarray):
    Z = input * layer.weights + layer.bias
    A= layer.activation_function(Z)
    return [Z, A]


# test_layer = Layer(10, 'sigmoid')
# test_layer.weights = np.random.rand(10, 10)
# test_layer.bias = np.random.rand(10)
# print(calc_layer(test_layer, np.ones(10)))
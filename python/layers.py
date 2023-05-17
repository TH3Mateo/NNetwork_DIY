import numpy as np


class Layer:
    def __init__(self, node_count: int=None, activation_function: str = None, weights=None, bias=None):
        self.activation_str = activation_function
        self.node_count = node_count
        self.activation_function = {
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'tanh': lambda x: np.tanh(x),
            'relu': lambda x: np.maximum(x, 0.05*x),
            'softmax': lambda x: np.exp(x) / sum(np.exp(x)),
            'output': lambda x: x
        }[activation_function]

        self.activation_deriv = {
            'sigmoid': lambda x: x * (1 - x),
            'tanh': lambda x: 1 - x ** 2,
            'relu': lambda x: np.where(x > 0, 1, 0.05),
            'softmax': lambda x: x * (1 - x),
            'output': lambda x: 1

        }[activation_function]
        self.weights = weights
        self.bias = bias

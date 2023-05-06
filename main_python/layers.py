import numpy as np
# from numba import jit
import cupy as cp

class Layer:
    def __init__(self, node_count: int, activation_function: str):
        self.node_count = node_count
        self.activation_function = {
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'tanh': lambda x: np.tanh(x),
            'relu': lambda x: np.maximum(0, x),
            'softmax': lambda x: np.exp(x) / np.sum(np.exp(x)),
        }[activation_function]

        self.activation_deriv = {
            'sigmoid': lambda x: x * (1 - x),
            'tanh': lambda x: 1 - x ** 2,
            'relu': lambda x: 1. * (x > 0),
            'softmax': lambda x: x * (1 - x),

        }[activation_function]
        self.weights = None
        self.bias = None


def calc_layer(layer: Layer, input: cp.array):
    # print('Layers: ')
    # print(input.shape)
    # print(layer.weights.shape)
    # print(layer.bias.shape)
    Z = cp.dot(input,cp.asarray(layer.weights)) + cp.asarray(layer.bias)
    A= layer.activation_function(Z)
    print("LAYERO OUTPUT SHAPE:")
    print(A.shape)
    return Z, A


# test_layer = Layer(10, 'sigmoid')
# test_layer.weights = np.random.rand(10, 10)
# test_layer.bias = np.random.rand(10)
# print(calc_layer(test_layer, np.ones(10)))
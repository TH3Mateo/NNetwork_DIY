import layers
from layers import Layer
import numpy as np
import random
from numba import jit


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add_layer(self, layer: Layer):
        self.layers.append(layer)


    def mass_predict(self, X):
        for layer in self.layers:
             = layers.calc_layer(layer, X)[1]

    @jit(target_backend="cuda")
    def train(self, X, Y, iterations, learning_rate, batch_size):
        for index in range(len(self.layers) - 1):
            self.layers[index].weights = np.random.rand(self.layers[index].node_count, self.layers[index+1].node_count)
            self.layers[index].bias = np.random.rand(self.layers[index].node_count)

        for iteration in range(iterations):
            pass






def output_binarator(output):
    max = np.argmax(output)
    output = np.zeros(len(output))
    output[max] = 1
    return output





import layers
from layers import Layer
import numpy as np
import cupy as cp
import random


# from numba import jit


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def mass_predict(self, X):
        out = []
        A = cp.array(X)
        for layer in self.layers:
            Z, A = layers.calc_layer(layer, A)
            out.append([np.array(Z), np.array(A)])
        return out

    def back_prop(self, X, Y, data, learning_rate):
        # predictions = self.mass_predict(X)
        # dz, dw, db = [], [], []
        # for index in range(len(self.layers) - 1, 0, -1):
        #     dz[index] = self.layers[index + 1].weights * dz[index + 1] * self.layers[index].activation_deriv(
        #         predictions[index][0])
        #     dw[index] = (dz[index] * self.layers[index - 1]
        dZ_prev = None
        for i in reversed(range(len(self.layers))):
            if i == len(self.layers) - 1:
                dZ = data[i][1] - Y_binarator(Y)
            else:
                dZ = self.layers[i + 1].weights.transpose() * dZ_prev * self.layers[i].activation_deriv(data[i][0])

            if i == 0:
                dW = (dZ * X.transpose()) / len(X)
            else:
                dW = (dZ * data[i - 1][1].transpose()) / len(X)

            dB = (np.sum(dZ, axis=1)) / len(X)
            dZ_prev = dZ
            self.layers[i].weights = self.layers[i].weights - learning_rate * dW
            self.layers[i].bias = self.layers[i].bias - learning_rate * dB

    # def tester(self):
    #     for i in reversed(range(len(self.layers))):
    #         print(i)

    def train(self):
        for index in range(len(self.layers) - 1):
            self.layers[index].weights = np.random.rand(self.layers[index].node_count,
                                                        self.layers[index + 1].node_count) - 0.5
            self.layers[index].bias = np.random.rand(self.layers[index+1].node_count) - 0.5

        # for iteration in range(iterations):
        #     pass


def output_binarator(output):  # takes output in form of probability of each class
    max = np.argmax(output)
    output = np.zeros(len(output))
    output[max] = 1
    return output


def Y_binarator(Y):
    pass

#z softmax na końcowe dane
def output_numerator(output):
    pass

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
        out=[]
        A=cp.array(X)
        for layer in self.layers:
            Z,A = layers.calc_layer(layer, A)
            out.append([np.array(Z),np.array(A)])


    def train(self, X, Y, iterations, learning_rate, batch_size):
        for index in range(len(self.layers) - 1):
            self.layers[index].weights = np.random.rand(self.layers[index].node_count, self.layers[index+1].node_count)-0.5
            self.layers[index].bias = np.random.rand(self.layers[index].node_count)-0.5

        for iteration in range(iterations):

            pass






def output_binarator(output):  #takes output in form of probability of each class
    max = np.argmax(output)
    output = np.zeros(len(output))
    output[max] = 1
    return output





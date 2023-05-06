import sys
import math
import layers
from layers import Layer
import numpy as np
import cupy as cp
import random
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import utilities as u

# from numba import jit


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

########################################
    #  NIE ROBIE TRANSPOZYCJI DANYCH, TRZEBA GDZIES ZROBIC
########################################


    def mass_predict(self, X: np.array):
        out = []
        A = cp.array(X)
        for layer in self.layers:
            print(layer)
            Z, A = layers.calc_layer(layer, A)
            out.append([cp.asnumpy(Z), cp.asnumpy(A)])

        del Z,A
        cp._default_memory_pool.free_all_blocks()
        return out

    def back_prop(self, X, Y, data, learning_rate):

        dZ_prev = None
        for i in reversed(range(len(self.layers))):
            if i == len(self.layers) - 1:
                dZ = cp.asarray(data[i][1] - Y_binarator(Y))
            else:
                dZ = cp.asarray(self.layers[i + 1].weights).transpose() * dZ_prev * self.layers[i].activation_deriv(cp.asarray(data[i][0]))

            if i == 0:
                dW = (dZ * cp.asarray(X).transpose()) / len(X)
            else:
                dW = (dZ * cp.asarray(data[i - 1][1]).transpose()) / len(X)

            dB = (np.sum(dZ, axis=1)) / len(X)
            dZ_prev = dZ
            self.layers[i].weights = self.layers[i].weights - cp.asnumpy(learning_rate * dW)
            self.layers[i].bias = self.layers[i].bias - cp.asnumpy(learning_rate * dB)

    # def tester(self):
    #     for i in reversed(range(len(self.layers))):
    #         print(i)

    def train(self, full_X, full_Y, iterations, learning_rate, batch_size):

        for index in range(len(self.layers)):
            self.layers[index].weights = np.random.rand(self.layers[index].node_count, (1 if index == len(self.layers)-1 else self.layers[index+1].node_count)) - 0.5
            self.layers[index].bias = np.random.rand((0 if index == len(self.layers)-1 else self.layers[index+1].node_count)) - 0.5
        rounds = 0
        loss = 420 # just a big number to initiate loss for the first iteration
        batch_count = math.ceil(len(full_X)/batch_size)
        while iterations>0 and loss > 0.01:

            dataset =np.append(full_X,full_Y,axis=1)
            np.random.shuffle(dataset)
            for b in range(batch_count):

                batch = dataset[b*batch_size:(b+1)*batch_size]# tu trzeba bedzie zabezpieczyc w razie jakby nie dzieliło sie idealnie rowno
                X_batch = batch[:,:-1]
                Y_batch = batch[:,[-1]]
                data = self.mass_predict(X_batch)
                self.back_prop(X_batch, Y_batch, data, learning_rate)
                loss = calc_loss(Y_binarator(Y_batch), output_binarator(data[-1][1]))
            iterations -= 1

    def save_model(self, file_name):
        with open(u.create_path("models\\"+file_name), 'wb') as f:
            pickle.dump(self, f,pickle.HIGHEST_PROTOCOL)



def calc_loss(predicted,expected):
    return np.sum((predicted-expected)**2)/len(predicted)



def output_binarator(output):  # takes output in form of probability of each class
    max = np.argmax(output)
    output = np.zeros(len(output))
    output[max] = 1
    return output

#z numerka klasyfikacji na array 0 i 1
def Y_binarator(Y):
    pass

#z softmax na końcowe dane
def output_numerator(output):
    pass

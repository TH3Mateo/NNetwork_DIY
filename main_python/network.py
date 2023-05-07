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


    def mass_predict(self, X: np.array):
        out = []
        A = cp.array(X)
        for layer in self.layers:
            # print(layer)
            Z, A = layers.calc_layer(layer, A)
            # print("A:")
            # print(A.shape)
            # print(A[0:5])
            out.append([cp.asnumpy(Z), cp.asnumpy(A)])

        del Z, A
        cp._default_memory_pool.free_all_blocks()
        return out

    def back_prop(self, X, Y, data, learning_rate):



        dZ_prev = None
        for i in reversed(range(len(self.layers)-1)):
            if i == len(self.layers) - 2:
                # print("Train dz: ")
                # print(data[i][1].shape)
                # print(Y_binarator(Y).shape)
                dZ = cp.asarray(data[i][1] - Y_binarator(Y).transpose())
            else:
                # print("i kin calculating dZ: ",i)
                # print(cp.asarray(self.layers[i+1].weights).shape)
                # print(dZ_prev.shape)
                # print(self.layers[i].activation_deriv(cp.asarray(data[i][0])).shape)



                dZ = cp.dot(dZ_prev,cp.asarray(self.layers[i+1].weights).transpose())*self.layers[i].activation_deriv(cp.asarray(data[i][0]))
                # print(dZ.shape)

            if i == 0:
                # print("Train dw1: ")
                # print(dZ.shape)
                # print(cp.asarray(X).transpose().shape)

                dW = cp.dot(cp.asarray(X).transpose(),dZ) / len(X)
            else:
                # print("Train dw: ")
                # print(dZ.shape)
                # print(cp.asarray(data[i - 1][1]).transpose().shape)


                dW = cp.dot(cp.asarray(data[i - 1][1]).transpose(),(dZ)) / len(X)


            dB = (np.sum(dZ, axis=0)) / len(X)
            dZ_prev = dZ
            self.layers[i].weights = self.layers[i].weights - cp.asnumpy(learning_rate * dW)
            # print("----------------")
            # print(self.layers[i].weights.shape)
            # print("----------------------")
            self.layers[i].bias = self.layers[i].bias - cp.asnumpy(learning_rate * dB)

    # def tester(self):
    #     for i in reversed(range(len(self.layers))):
    #         print(i)

    def train(self, full_X, full_Y, iterations, learning_rate, batch_size):

        for index in range(len(self.layers)):
            self.layers[index].weights = np.random.rand(self.layers[index].node_count, (
                1 if index == len(self.layers) - 1 else self.layers[index + 1].node_count)) - 0.5
            self.layers[index].bias = np.random.rand(
                (0 if index == len(self.layers) - 1 else self.layers[index + 1].node_count)) - 0.5
        rounds = 0
        loss = 420  # just a big number to initiate loss for the first iteration
        batch_count = math.ceil(len(full_X) / batch_size)
        while iterations > 0 and loss > 0.01:
            dataset = np.append(full_X, np.reshape(full_Y,(-1,1)), axis=1)
            # print(dataset)
            np.random.shuffle(dataset)
            for b in range(batch_count):
                batch = dataset[b * batch_size:(b + 1) * batch_size]  # tu trzeba bedzie zabezpieczyc w razie jakby nie dzieliło sie idealnie rowno
                X_batch = batch[:, :-1]
                Y_batch = batch[:, [-1]]
                data = self.mass_predict(X_batch)
                self.back_prop(X_batch, Y_batch, data, learning_rate)

                loss = calc_loss(Y_binarator(Y_batch), output_binarator(data[-2][1]))
                print(loss)
            iterations -= 1

    def save_model(self, file_name):
        with open(u.create_path("models\\" + file_name), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def calc_loss(predicted, expected):
    return np.sum((predicted - expected) ** 2) / len(predicted)


def output_binarator(output):
    # print(output.shape)
    out = np.zeros(tuple(reversed(output.shape)))
    for i in range(len(output)):
        out[np.argmax(output[i])][i]= 1
    return out


# z numerka klasyfikacji na array 0 i 1
def Y_binarator(Y):
    array = np.zeros((10, Y.size), dtype=int)
    for i in range(Y.size):
        array[Y[i][0]][i] = 1
    return array



# z softmax na końcowe dane
def output_numerator(output):
    max=[]
    output = np.array(output)
    for i in range(len(output)):
        max.append(np.argmax(output[i]))
    return max


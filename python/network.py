import sys
import os
import math
import data_tools as dt
from layers import Layer
import numpy as np
import cupy as cp

try:
    import Cpickle as pickle
except:
    import pickle


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.accuracy = 0

    def add_layer(self, new_layer):
        self.layers.append(new_layer)

    def init_params_modular(self):
        # for layer in self.layers[:-1]:
        # print(layer.activation_str)

        for i in range(len(self.layers) - 1):
            self.layers[i].weights = cp.asarray(
                np.random.rand(self.layers[i + 1].node_count, self.layers[i].node_count) - 0.5)
            self.layers[i].bias = cp.asarray(np.random.rand(self.layers[i + 1].node_count, 1) - 0.5)
            # print(self.layers[i].weights.shape)
            # print(self.layers[i].bias.shape)
            # print(self.layers[i].activation_str)

    def forward_prop_modular(self, X: cp.ndarray):
        Z, A = [], []
        for i in range(len(self.layers) - 1):
            if i == 0:
                # print(self.layers[i].weights.shape)
                # print(X.shape)
                Z.append(self.layers[i].weights.dot(X) + self.layers[i].bias)
                A.append(self.layers[i].activation_function(Z[i]))
            else:
                Z.append(self.layers[i].weights.dot(A[i - 1]) + self.layers[i].bias)
                A.append(self.layers[i].activation_function(Z[i]))
        # sys.exit()
        return Z, A

    def back_prop_modular(self, Z, A, X, Y, alpha):
        m = X.shape[1]

        one_hot_Y = cp.asarray(one_hot(Y))
        dZ, dW, db = [], [], []
        for i in range(len(A)):
            dZ.append(0)
            dW.append(0)
            db.append(0)
        for i in reversed(range(len(A))):
            if i == len(A) - 1:
                dZ[i] = A[i] - one_hot_Y
                dW[i] = 1 / m * dZ[i].dot(A[i - 1].T)
                db[i] = 1 / m * np.sum(dZ[i])
            elif i != 0:
                dZ[i] = self.layers[i + 1].weights.T.dot(dZ[i + 1]) * self.layers[i].activation_deriv(Z[i])
                dW[i] = 1 / m * dZ[i].dot(A[i - 1].T)
                db[i] = 1 / m * np.sum(dZ[i])
            else:
                dZ[i] = self.layers[i + 1].weights.T.dot(dZ[i + 1]) * self.layers[i].activation_function(Z[i])
                dW[i] = 1 / m * dZ[i].dot(X.T)
                db[i] = 1 / m * np.sum(dZ[i])

        # migrated update params modular
        for i in range(len(self.layers) - 1):
            self.layers[i].weights = self.layers[i].weights - alpha * dW[i]
            self.layers[i].bias = self.layers[i].bias - alpha * db[i]

    def train(self, full_X, full_Y, alpha, iterations, batch_size):
        # print(full_X.shape)

        self.init_params_modular()
        batch_count = math.ceil(len(full_X) / batch_size)
        dataset = np.append(full_X, full_Y.reshape(1, len(full_Y)), axis=0)
        # print(dataset.shape)
        i = 0
        while i < iterations and self.accuracy < 0.95:
            dataset = dataset[:, np.random.permutation(dataset.shape[1])]
            for b in range(batch_count):
                batch = dataset[:, b * batch_size:(b + 1) * batch_size]
                X = cp.asarray(batch[:-1])
                Y = batch[-1].reshape(1, batch_size).astype(int)

                Z, A = self.forward_prop_modular(X)

                self.back_prop_modular(Z, A, X, Y, alpha)

            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = cp.asnumpy(get_predictions(A[-1]))
                self.accuracy = get_accuracy(predictions, Y)
                print(self.accuracy)
            i += 1

        print("final accuracy after training: ", self.accuracy)
        return

    def test_network(self, X: np.array, Y: np.array):
        Z, A = self.forward_prop_modular(cp.asarray(X))
        predictions = get_predictions(A[-1])

        accuracy = get_accuracy(cp.asnumpy(predictions), Y)
        print("Loaded model accuracy: " + str(accuracy))

    def single_predict(self, X):
        X = cp.asarray(X)
        Z, A = [], []
        for i in range(len(self.layers) - 1):
            if i == 0:
                # print(W[i].shape)
                # print(X.shape)
                Z.append(self.layers[i].weights.dot(X) + self.layers[i].bias)
                A.append(self.layers[i].activation_function(Z[i]))
            else:
                Z.append(self.layers[i].weights.dot(A[i - 1]) + self.layers[i].bias)
                A.append(self.layers[i].activation_function(Z[i]))
                # print(A[-1].shape)
        return get_predictions(A[-1])

    def save_model(self, filename):
        try:
            os.mkdir(dt.create_path("models\\" + filename))
        except:
            pass

        for i in range(len(self.layers)):
            with open(dt.create_path("models\\" + filename + "\\l" + str(i) + ".pkl"), 'wb') as output:
                layer_data = [self.layers[i].weights, self.layers[i].bias, self.layers[i].activation_str]
                pickle.dump(layer_data, output, pickle.HIGHEST_PROTOCOL)


def load_model(filename):
    N = Network()
    for i in range(len(os.listdir(dt.create_path("models\\" + filename)))):
        with open(dt.create_path("models\\" + filename + "\\l" + str(i) + ".pkl"), 'rb') as input:
            layer_data = pickle.load(input)

            N.add_layer(Layer(activation_function=layer_data[2], weights=layer_data[0], bias=layer_data[1]))
            # print(L.weights.shape)
            # print(L.bias.shape)
    # print("--------------------------------")
    # for i in range(len(N.layers)-1):
    #
    #     print(N.layers[i].weights.shape)
    #     print(N.layers[i].bias.shape)
    #     print(N.layers[i].activation_str)

    return N


def one_hot(Y):
    # print(Y)
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def get_predictions(A2):  # 0-9
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

import sys
import math

import yaml

import layers
# from layers import layer_activation
import numpy as np
import cupy as cp
import random
import data_tools as d
import layers
import utilities as uti
import data_tools as dt

import os

layers = []
connections = []
def layer(new_layer):
    layers.append(new_layer)
def connection(new_connection):
    connections.append(new_connection)

def init_params_modular():
    W, b = [], []
    print(connections)
    for i in range(len(connections)):
        W.append(np.random.rand(layers[i+1], layers[i]) - 0.5)
        b.append(np.random.rand(layers[i+1], 1) - 0.5)
    return W, b

# function_dic = {
#     'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
#     'tanh': lambda x: np.tanh(x),
#     'relu': lambda x: np.maximum(0.05 * x, x),
#     'softmax': lambda x: np.exp(x)/sum(np.exp(x)),
#     'output': lambda x: x
# }

def function_type(Z, type):
    if type == 'relu':
        return np.maximum(Z, 0)
    elif type == 'softmax':
        A = np.exp(Z)/sum(np.exp(Z))
        return A
    elif type == 'tanh':
        return np.tanh(Z)
    elif type == 'sigmoid':
        return 1 / (1 + np.exp(-Z))
    else:
        print("Funkcja poza zakresem")



def forward_prop_modular(W, b, X, type):
    Z, A = [], []
    for i in range(len(W)):
        if i == 0:
            # print(W[i].shape)
            # print(X.shape)
            Z.append(W[i].dot(X) + b[i])
            A.append(function_type(Z[i], type[i]))
        else:
            Z.append(W[i].dot(A[i-1]) + b[i])
            A.append(function_type(Z[i], type[i]))
    return Z, A

def softmax_deriv(X):
    softmax_matrix = np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
    return softmax_matrix[:, :, np.newaxis] * (
                np.expand_dims(np.eye(X.shape[1]), axis=0) - softmax_matrix[:, np.newaxis, :])

# deriv_dic = {
#     'sigmoid': lambda x: x * (1 - x),
#     'tanh': lambda x: 1 - x ** 2,
#     'relu': lambda x: x > 0,
#     'softmax': lambda x: softmax_deriv(x),
#     'output': lambda x: 1
# }

def deriv_type(Z,type):
    if type == 'relu':
        return Z > 0
    elif type == 'softmax':
        A = softmax_deriv(Z)
        return A
    elif type == 'tanh':
        return 1 - Z ** 2
    elif type == 'sigmoid':
        return Z * (1 - Z)
    else:
        print("Funkcja poza zakresem")


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


# dane wejściowe to tablice pełne danych które powinny otrzymać
def back_prop_modular(Z, A, W, X, Y):
    one_hot_Y = one_hot(Y)
    dZ, dW, db = [], [], []
    for i in range(len(A)):
        dZ.append(0)
        dW.append(0)
        db.append(0)
    for i in reversed(range(len(A))):
        if i == len(A)-1:
            dZ[i] = A[i] - one_hot_Y
            dW[i] = 1 / m * dZ[i].dot(A[i-1].T)
            db[i] = 1 / m * np.sum(dZ[i])
        elif i != 0:
            dZ[i] = W[i+1].T.dot(dZ[i+1]) * deriv_type(Z[i],connections[i])
            dW[i] = 1 / m * dZ[i].dot(A[i-1].T)
            db[i] = 1 / m * np.sum(dZ[i])
        else:
            dZ[i] = W[i+1].T.dot(dZ[i+1]) * deriv_type(Z[i],connections[i])
            dW[i] = 1 / m * dZ[i].dot(X.T)
            db[i] = 1 / m * np.sum(dZ[i])
    return dW, db



def update_params_modular(W, b, dW, db, alpha):
    for i in range(len(W)):
        W[i] = W[i] - alpha * dW[i]
        b[i] = b[i] - alpha * db[i]
    return W, b


def get_predictions(A2):  # 0-9
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    print(X.shape)
    print(Y.shape)
    print(m)
    temp = []
    # W1, b1, W2, b2 = init_params()
    W, b = init_params_modular()
    for i in range(iterations):

        Z, A = forward_prop_modular(W, b, X, connections)

        dW, db = back_prop_modular(Z, A, W, X, Y)

        W, b = update_params_modular(W, b, dW, db, alpha)

        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A[-1])
            accuracy = get_accuracy(predictions, Y)
            print(accuracy)
            if accuracy > 0.55:
                return W, b
    return W, b

def main(X_train,Y_train,alpha,iterations,number):
    global m
    m = number

    W,b = gradient_descent(X_train, Y_train, alpha, iterations)
    return W,b


def single_predict(W,b,connections, X):
    Z, A = [], []
    for i in range(len(W)):
        if i == 0:
            # print(W[i].shape)
            # print(X.shape)
            Z.append(W[i].dot(X) + b[i])
            A.append(function_type(Z[i], connections[i]))
        else:
            Z.append(W[i].dot(A[i-1]) + b[i])
            A.append(function_type(Z[i], connections[i]))
            print(A[-1].shape)
    return get_predictions(A[-1])

def save(W,b,connections,filename):
    # os.mkdir(d.create_path("models\\"+filename))
    for i in range(len(W)):
        np.save(d.create_path("models\\"+filename+"\\W"+str(i)+".npy"), W[i])
        np.save(d.create_path("models\\"+filename+"\\b"+str(i)+".npy"), b[i])
        np.save(d.create_path("models\\"+filename+"\\connections"+str(i)+".npy"), connections[i])

def load(filename):
    W,b,connections = [],[],[]
    for i in range(int(len(os.listdir(d.create_path("models\\"+filename)))/3)):
        W.append(np.load(d.create_path("models\\"+filename+"\\W"+str(i)+".npy")))
        b.append(np.load(d.create_path("models\\"+filename+"\\b"+str(i)+".npy")))
        connections.append(np.load(d.create_path("models\\"+filename+"\\connections"+str(i)+".npy")))
    return W,b,connections

def test_network(W,b,connections,X,Y):
    Z, A = forward_prop_modular(W, b, X, connections)
    predictions = get_predictions(A[-1])
    accuracy = get_accuracy(predictions, Y)
    print("Loaded model accuracy: " + str(accuracy))
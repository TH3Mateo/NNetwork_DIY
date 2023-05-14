import numpy as np
from data_tools import data_loader
from network import Network
from layers import Layer
#
X,Y = data_loader("mnist_digits_full_dataset.csv")
# print("loaded data")
# print("X: ")
# print(X.shape)
# print(X)
# print("Y: ")
# print(Y.shape)
# print(Y)
X=X/255
# print("NEW X: ")
# print(X)
N = Network()

N.add_layer(Layer(784, 'sigmoid'))
N.add_layer(Layer(512, 'relu'))
N.add_layer(Layer(128, 'tanh'))
# N.add_layer(Layer(10, 'softmax'))
N.add_layer(Layer(10, 'output'))



N.train(X,Y,50,0.1,5000)

N.save_model("first_working_gen")

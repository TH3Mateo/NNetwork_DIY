import numpy as np
from data_tools import data_loader
from network import Network
from layers import Layer

X,Y = data_loader("mnist_digits_full_dataset.csv")
# print("loaded data")
# N = Network()
# N.add_layer(Layer(784, 'sigmoid'))
# N.add_layer(Layer(512, 'relu'))
# N.add_layer(Layer(128, 'relu'))
# N.add_layer(Layer(10, 'softmax'))

# N.train(X,Y,100,0.01,100)


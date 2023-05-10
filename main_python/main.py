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

# layer(768)
# connection 'sigmoid'
# layer(512)
# connection 'relu'
# layer(128)
# connection 'relu'
# layer(64)
# connection 'softmax'
# layer(10)

N.add_layer(Layer(784, 'sigmoid'))
N.add_layer(Layer(512, 'relu'))
N.add_layer(Layer(128, 'tanh'))
N.add_layer(Layer(64, 'softmax'))
N.add_layer(Layer(10, 'output'))


# N.add_layer(Layer(10




# print(N.mass_predict(X)[0])





N.train(X,Y,50,0.1,5000)

N.save_model("first_working_gen")

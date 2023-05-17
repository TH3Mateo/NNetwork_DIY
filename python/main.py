import numpy as np
import matplotlib.pyplot as plt
from data_tools import data_loader,data_slicer
from network import Network,load_model
from layers import Layer
#
data = data_loader('mnist_digits_full_dataset.csv')
data = np.array(data)

np.random.shuffle(data)  # shuffle before splitting into dev and training sets

X_train,Y_train,X_dev,Y_dev = data_slicer(data,0.14)
#
# print("X_train shape: " + str(X_train.shape))
# print("Y_train shape: " + str(Y_train.shape))
#
# N = Network()
#
# N.add_layer(Layer(784, 'relu'))
# N.add_layer(Layer(128, 'relu'))
# N.add_layer(Layer(64, 'softmax'))
# N.add_layer(Layer(10, 'output'))
#
# N.train(X_train,Y_train,0.10,2000,5000)
# N.save_model("first_correct")

N = load_model("first_correct")
print("-----------------------------------")
exmp = X_dev[:,[0]]
print(np.array(exmp).shape)
print(N.single_predict(exmp))

l = int(np.sqrt(exmp.shape[0]))
plt.imshow(exmp.reshape(l, l), cmap="Blues_r")
N.test_network(X_dev,Y_dev)
plt.show()
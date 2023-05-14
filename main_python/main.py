import numpy as np
# from data_tools import data_loader

import network as N
import data_tools as dt
import matplotlib.pyplot as plt
from network import connections
data = dt.data_loader('mnist_digits_full_dataset.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[-1]
X_dev = data_dev[0:n-1]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[-1]
X_train = data_train[0:n-1]
X_train = X_train / 255.
_, m_train = X_train.shape
#
# N.layer(784)
# N.connection('relu')
# # N.layer(512)
# # N.connection('relu')
# N.layer(128)
# N.connection('relu')
# N.layer(64)
# N.connection('softmax')
# N.layer(10)
#
# W,b = N.main(X_train,Y_train,0.10,500,m)
# print(W,b) #zapisać dane do wykorzystania
# print(type(W))
# N.save(W,b,connections,"chaotic_working") #zapisać dane do wykorzystania


#stworzyć funkcjie wykorzystującą zapisane dane
W,b,connections = N.load("chaotic_working")

print("-----------------------------------")
exmp = X_dev[:,[0]]
print(np.array(exmp).shape)
print(N.single_predict(W,b,connections,exmp))
l = int(np.sqrt(exmp.shape[0]))
plt.imshow(exmp.reshape(l, l), cmap="Blues_r")
plt.show()


N.test_network(W,b,connections,X_dev,Y_dev)
#przesada w ilości warstw powoduje za wolne nauczanie lub przeuczanie
import tensorflow as tf
import keras
from keras.layers import Dense, Activation
import random
import numpy as np
import pandas as pd
from dask import dataframe as dk
import matplotlib.pyplot as plt
from keras.models import Sequential


data = dk.read_csv('mnist_digits_full_dataset.csv')
data = pd.DataFrame(data)

train = data.iloc[0:65000, :]
test = data.iloc[65000:70000, :]
X_train = train.iloc[:, 0:-1]/255
Y_train = train.iloc[:, -1]
X_test = test.iloc[:, 0:-1]/255
Y_test = test.iloc[:, -1]

# Network = Sequential([
#     Dense(784, input_shape=(784,),activation='sigmoid'),
#     Dense(128, activation='relu'),
#     Dense(10, activation='softmax')
# ])
# Network.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Network.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2)
# Network.evaluate(X_test, Y_test)
# Network.save('mnist_digits_model.h5')


Network = keras.models.load_model('mnist_digits_model.h5')
Network.evaluate(X_test, Y_test)
sample = X_test.iloc[[random.randint(0, 5000)]]
# print(sample)
print(np.argmax(Network.predict(sample)))

plt.imshow(np.array(sample).reshape(28, 28), cmap='Greys')
plt.show()

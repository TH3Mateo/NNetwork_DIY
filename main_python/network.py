import math
import layers
from layers import Layer
import numpy as np
import cupy as cp

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import utilities as u



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
        for layer in self.layers[:-1]:
            # print(layer)
            Z, A = layers.calc_layer(layer, A)
            # print("A:")
            # print(A.shape)
            # print(A[0:5])
            out.append([cp.asnumpy(Z), cp.asnumpy(A)])

        del Z, A
        cp._default_memory_pool.free_all_blocks()
        # print(self.layers)
        return out

    #
    # def back_prop_old(self, X, Y, data, learning_rate):
    #    exp_matrix = np.exp(matrix)

    # Compute the sum of each row
    # row_sum =
    #
    # # Divide each element of the matrix by the sum of its row
    # softmax_matrix =
    #
    # return softmax_matrix

    #
    #     for i in reversed(range(len(self.layers[:-1]))):
    #         if i == len(self.layers) - 2:
    #              dZ = cp.asarray(data[i][1] - Y_binarator(Y).transpose())
    #         else:
    #            dZ = cp.dot(dZ_prev,cp.asarray(self.layers[i+1].weights).transpose())*self.layers[i].activation_deriv(cp.asarray(data[i][0]))
    #         if i == len(self.layers)-1:
    #             dW = cp.dot(cp.asarray(X).transpose(),dZ) / len(X)
    #         else:
    #             dW = cp.dot(cp.asarray(data[i - 1][1]).transpose(),dZ) / len(X)
    #             print(dW.shape)
    #
    #         dB = (np.sum(dZ, axis=0)) / len(X)
    #         dZ_prev = dZ
    #         print("----------------")
    #         print(self.layers[i].weights.shape)
    #         print(dW.shape)
    #         print("----------------------")
    #         self.layers[i].weights = self.layers[i].weights - cp.asnumpy(learning_rate * dW)
    #
    #         self.layers[i].bias = self.layers[i].bias - cp.asnumpy(learning_rate * dB)



    def back_prop(self, X, Y, data, learning_rate):
        dZ_prev = None
        for i in reversed(range(len(self.layers[:-1]))):
            # print(len(self.layers[:-1]))
            # print(i)
            if i == len(self.layers[:-1])-1:
                dZ = cp.asarray(data[i][1] - Y_binarator(Y).transpose())
                # print(dZ.shape)

            else:
                dZ = cp.dot(dZ_prev,cp.asarray(self.layers[i+1].weights).transpose())*self.layers[i].activation_deriv(cp.asarray(data[i][0]))

            if i == 0:
                dW = cp.dot(cp.asarray(X).transpose(),dZ) / len(X)

            else:
                dW = cp.dot(cp.asarray(data[i - 1][1]).transpose(),dZ) / len(X)

            dB = (np.sum(dZ, axis=0)) / len(X)

            self.layers[i].weights = self.layers[i].weights - cp.asnumpy(learning_rate * dW)
            self.layers[i].bias = self.layers[i].bias - cp.asnumpy(learning_rate * dB)
            dZ_prev = dZ.copy()


        # sys.exit()

    def train(self, full_X, full_Y, iterations, learning_rate, batch_size):

        for index in range(len(self.layers)-1):
            print(index)
            self.layers[index].weights = np.random.rand(self.layers[index].node_count,self.layers[index + 1].node_count) - 0.5
            self.layers[index].bias = np.random.rand(self.layers[index + 1].node_count) - 0.5

        loss = 420  # just a big number to initiate loss for the first iteration
        batch_count = math.ceil(len(full_X) / batch_size)
        while iterations > 0 and self.loss > 0.01:
            dataset = np.append(full_X, np.reshape(full_Y,(-1,1)), axis=1)
            # print(dataset)
            np.random.shuffle(dataset)

            for b in range(batch_count):
                batch = dataset[b * batch_size:(b + 1) * batch_size]  # tu trzeba bedzie zabezpieczyc w razie jakby nie dzieliło sie idealnie rowno
                X_batch = batch[:, :-1]
                Y_batch = batch[:, [-1]]



                data = self.mass_predict(X_batch)
                self.back_prop(X_batch, Y_batch, data, learning_rate)

            self.loss = calc_loss(Y_binarator(Y_batch), output_binarator(data[-1][1]))
            # print("Iteration: ",iterations)
            print("ACUURACY: ",calc_accuracy(Y_binarator(Y_batch), output_binarator(data[-1][1])))
            # print("LOSS: ",loss)
            iterations -= 1

    def save_model(self, file_name):
        with open(u.create_path("models\\" + file_name), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def calc_loss(predicted, expected):

    return np.sum(np.logical_xor(predicted ,expected) / len(predicted) ** 2)

def calc_accuracy(predicted, expected):
    return np.sum(np.logical_and(predicted,expected)) / len(predicted[0])


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
        array[int(Y[i])][i] = 1
    # print(Y)
    # print(array)
    return array



# z softmax na końcowe dane
def output_numerator(output):
    max=[]
    output = np.array(output)
    for i in range(len(output)):
        max.append(np.argmax(output[i]))
    return max


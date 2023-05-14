import os
def create_path(filename):return "\\".join(os.path.dirname(__file__).split("\\")[0:-1])+"\\"+filename

# import numpy as np
# array = np.random.randint(1,50, size=(3,3))
# print(f"array:\n{array}\n")
# np.random.shuffle(array)
# print(f"shuffled array:\n{array}")
#
# print(array[:,:-1])
# # print(array[:,[-1]])
#
#     # Compute the exponential of each element of the matrix
#     exp_matrix = np.exp(matrix)
#
#     # Compute the sum of each row
#     row_sum = np.sum(exp_matrix, axis=1, keepdims=True)
#
#     # Compute the softmax matrix
#     softmax_matrix = exp_matrix / row_sum
#
#     # Compute the derivative matrix
#     diag_matrix = np.expand_dims(np.eye(matrix.shape[1]), axis=0)
#     derivative_matrix = softmax_matrix[:, :, np.newaxis] * (diag_matrix - softmax_matrix[:, np.newaxis, :])
#
#     return derivative_matrix

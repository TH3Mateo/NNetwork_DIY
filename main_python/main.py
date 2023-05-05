import numpy as np

import tensorflow
import keras
from numba import jit, cuda

print(tensorflow.__version__)
print(keras.__version__)



@jit(target_backend="cuda")
def func(a):
    for i in range(10000000):
        a[i] += 1
    return a




def main():
    print("start")
    func(np.ones(10000000))
    print("end")

main()

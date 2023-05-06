import tensorflow
import keras
from numba import jit, cuda
import numpy as np

print(tensorflow.__version__)
print(keras.__version__)



@jit(target_backend="cuda")
def func(a):
    for i in range(10000000):
        a += 1
    return a




def main():
    print("start")
    print(func(15))
    print("end")


main()

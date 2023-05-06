import os
def create_path(filename):return "\\".join(os.path.dirname(__file__).split("\\")[0:-1])+"\\"+filename

# import numpy as np
# array = np.random.randint(1,50, size=(3,3))
# print(f"array:\n{array}\n")
# np.random.shuffle(array)
# print(f"shuffled array:\n{array}")
#
# print(array[:,:-1])
# print(array[:,[-1]])
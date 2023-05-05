import os

def create_path(filename):
    path_str = os.path.dirname(__file__).split("\\")
    absolute_path = path_str[0:-1]
    return "\\".join(absolute_path)+"\\"+filename



print(create_path("scikit_learn_boston_dataset.csv"))
import os


def create_path(filename):return "\\".join(os.path.dirname(__file__).split("\\")[0:-1])+"\\"+filename



print(create_path("data\\scikit_learn_boston_dataset.csv"))
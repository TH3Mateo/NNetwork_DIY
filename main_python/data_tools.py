import matplotlib.pyplot as plt
import numpy as np

import utilities as u
import pandas as pd
from sklearn import datasets
from sklearn.datasets import fetch_openml
import random
import math
import os



def data_viewer():
    df = pd.read_csv(u.create_path("data\\mnist_digits_full_dataset.csv"))
    print(df.head())
    l = int(math.sqrt(len(df.iloc[0])))
    plt.imshow(df.iloc[random.randint(0,100)].values.reshape(l,l), cmap="Blues_r")
    plt.show()


def data_saver():
    db = fetch_openml('mnist_784',parser='auto')
    df = pd.DataFrame(db.data, columns=db.feature_names)
    df.to_csv(u.create_path("data\\mnist_digits_full_dataset.csv"), index=False)

def data_loader(filename):
    df = pd.read_csv(u.create_path("data\\"+filename))
    return  np.ndarray(df.drop("label", axis=1)),np.ndarray(df["label"])




data_viewer()
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
    db,y = fetch_openml('mnist_784',parser='auto',return_X_y=True)
    print(db.head())
    print(y.head())
    df = db.concat(pd.DataFrame(y,columns=["label"]),ignore_index=True)
    df.to_csv(u.create_path("data\\mnist_digits_full_dataset.csv"), index=False)

def data_loader(filename):
    print("Loading data...")
    df = pd.read_csv(u.create_path("data\\"+filename))
    print(df.head())
    return  np.ndarray(df.drop("label", axis=1)),np.ndarray(df["label"])

data_saver()
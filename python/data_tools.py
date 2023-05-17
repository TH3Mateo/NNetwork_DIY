import matplotlib.pyplot as plt
import dask.dataframe as d
from sklearn.datasets import fetch_openml
import pandas as pd
import random
import math
import os


def create_path(filename):return "\\".join(os.path.dirname(__file__).split("\\")[0:-1])+"\\"+filename

def data_viewer():
    data = pd.read_csv(create_path('data\\mnist_digits_full_dataset.csv')) #dalej naprawić dataviewer
    print(data.head())
    l = int(math.sqrt(len(data.iloc[0])))
    plt.imshow(data.iloc[random.randint(0,100)].values.reshape(l,l), cmap="Blues_r")
    plt.show()


def data_saver():
    db,y = fetch_openml('mnist_784',parser='auto',return_X_y=True)
    db = d.from_pandas(db,chunksize=1000)
    y = d.from_pandas(pd.DataFrame(y),chunksize=1000)
    df = db.merge(y)
    df.to_csv(create_path("data\\mnist_digits_full_dataset.csv"), index=False,single_file=True,mode="w")
    print(df.head())


def data_loader(filename):
    print("Loading data...")
    data = d.read_csv(create_path("data\\"+filename))
    # print(df.head())
    return data


def data_slicer(data, eval_ratio):
    m, n = data.shape
    eval_count = int(len(data) * eval_ratio)
    data_dev = data[0:eval_count].T
    Y_dev = data_dev[-1]
    X_dev = data_dev[0:n - 1]
    X_dev = X_dev / 255.

    data_train = data[eval_count:m].T
    Y_train = data_train[-1]
    X_train = data_train[0:n - 1]
    X_train = X_train / 255.
    _, m_train = X_train.shape
    return X_train, Y_train, X_dev, Y_dev
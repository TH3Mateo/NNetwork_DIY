import matplotlib.pyplot as plt
import numpy as np

import utilities as uti
import pandas as pd
# import dask.dataframe as d
# from sklearn.datasets import fetch_openml
import random
import math
import os
import os
def create_path(filename):return "\\".join(os.path.dirname(__file__).split("\\")[0:-1])+"\\"+filename



def data_viewer():
    data = pd.read_csv(create_path('data\\mnist_digits_full_dataset.csv')) #dalej naprawić dataviewer
    print(data.head())
    l = int(math.sqrt(len(data.iloc[0])))
    plt.imshow(data.iloc[random.randint(0,100)].values.reshape(l,l), cmap="Blues_r")
    plt.show()


# def data_saver():
#     db,y = fetch_openml('mnist_784',parser='auto',return_X_y=True)
#     db = d.from_pandas(db,chunksize=1000)
#     y = d.from_pandas(pd.DataFrame(y),chunksize=1000)
#     df = db.merge(y)
#     df.to_csv(u.create_path("data\\mnist_digits_full_dataset.csv"), index=False,single_file=True,mode="w")
#     print(df.head())

def data_loader(filename):
    print("Loading data...")
    data = pd.read_csv(create_path("data\\"+filename))
    # print(df.head())
    return data

# data_saver()
import matplotlib.pyplot as plt
import utilities as u
import numpy as np
import random
import pandas as pd
import os


df = pd.read_csv(u.create_path("data\\scikit_learn_boston_dataset.csv"))
print(df.head())
plt.imshow(df.iloc[random.randint(0,100)].values.reshape(8, 8), cmap="Blues_r")
plt.show()
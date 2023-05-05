from sklearn import datasets
import pandas as pd
db = datasets.load_digits()
df = pd.DataFrame(db.data, columns=db.feature_names)
df.to_csv("scikit_learn_boston_dataset.csv", index=False)
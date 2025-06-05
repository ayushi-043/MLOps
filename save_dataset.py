from sklearn.datasets import load_iris 
import pandas as pd
import os

#Load iris data
iris = load_iris(as_frame=True)
df = pd.concat([iris.data, pd.Series(iris.target, name="target")], axis=1)

#Create folder if not exists
os.makedirs("data", exist_ok=True)

#Sve to CSV
df.to_csv("data/iris.csv",  index=False)
print("iris.csv have been created successfully.")
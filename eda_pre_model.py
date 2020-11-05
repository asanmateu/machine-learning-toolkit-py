from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
type(iris)

print(iris.keys())
print(iris.target_names)

X = iris.data
Y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
print(df.head())

_ = pd.plotting.scatter_matrix(df, c=y, figsize=[8,8], s=150, marker=‘D’)

# Pandas methods to start EDA
df.head()
df.info()
df.describe()

# Choose Seaborn's plot that best suits the type of data -- in this a count plot
plt.figure()
sns.countplot(x=‘education’, hue=‘party’, data=df, palette=‘RdBu’)
plt.xticks([0,1], [’No’, ‘Yes’])
Plt.show()


import scipy.stats 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
# Machine Learning 

df = pd.read_csv("./technical_data/2_a.csv")


print(df)

x = df.loc[:, df.columns!='Class']
pca = PCA(n_components=2) # might change

x_pca = pca.fit_transform(x.values)

df['pca-one'] = x_pca[:,0]
df['pca-two'] = x_pca[:,1] 

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue='Class',
    data=df,
    legend="full",
    alpha=0.3
)
# Scatter Plot
print(x_pca)
print(x_pca.shape)
plt.show()

# Using a K-Means Algorithm 

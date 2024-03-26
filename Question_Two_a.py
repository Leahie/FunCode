import scipy.stats 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import seaborn as sns
# Machine Learning 

df = pd.read_csv("./technical_data/2_a.csv")


print(df)

x = df.loc[:, df.columns!='Class']
pca = PCA(n_components=2) # might change

x_pca = pca.fit_transform(x.values)

df['pca-one'] = x_pca[:,0]
df['pca-two'] = x_pca[:,1] 

fig, ax = plt.subplots(1,2)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue='Class',
    data=df,
    legend="full",
    alpha=0.3, 
    ax = ax[0]
) 
# Scatter Plot
# print(x_pca)
# print(x_pca.shape)
# plt.show()

# Using a K-Means Algorithm 
X_train_norm = preprocessing.normalize(df[['pca-one', 'pca-two']])

kmeans = KMeans(n_clusters = 2, random_state = 0, n_init='auto')
kmeans.fit(X_train_norm)

sns.scatterplot(data=df[['pca-one', 'pca-two']], x='pca-one', y='pca-two', hue = kmeans.labels_, ax = ax[1])
y_hat = kmeans.labels_
y = list(df['Class'])

acc = 0
tot = 0
for i in range(len(y)):
    if not y_hat[i] and y[i]=="AML":
        acc+=1 
    if y_hat[i] and y[i]=="ALL":
        acc+=1
    tot+=1
print("Accuracy:", acc/tot * 100)

plt.show()

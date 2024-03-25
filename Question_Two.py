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

pca = PCA(n_components= 3) # might change
print(df)
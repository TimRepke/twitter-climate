import hdbscan
import numpy as np
from typing import Literal
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

layout = np.load('data/climate2/topics/layout_10000_TSNEArgs.npy')

neighbors = NearestNeighbors(n_neighbors=20)
neighbors_fit = neighbors.fit(layout)
distances, indices = neighbors_fit.kneighbors(layout)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.plot(distances)
plt.show()

max_k = 2000
neigh = NearestNeighbors(n_neighbors=max_k)
nbrs = neigh.fit(layout)
distances, indices = nbrs.kneighbors(layout)

avgs = []
x = []
for k in tqdm(range(3, max_k, 10)):
    avgs.append(distances[:, 1:k].mean(axis=1).mean())
    x.append(k)
plt.plot(x, avgs)
plt.title('mean mean')
plt.show()

avgs = []
for k in tqdm(range(3, max_k, 10)):
    avgs.append(distances[:, 1:k].mean(axis=1).max())
plt.plot(x, avgs)
plt.title('mean max')
plt.show()

avgs = []
for k in tqdm(range(3, max_k, 10)):
    avgs.append(distances[:, 1:k].mean(axis=1).max())
plt.plot(x, avgs)
plt.title('mean min')
plt.show()

kth = []
for k in tqdm(range(3, max_k, 10)):
    kth.append(distances[:, k].max())
plt.plot(x, kth)
plt.title('max kth neighbour')
plt.show()

kth = []
for k in tqdm(range(3, max_k, 10)):
    kth.append(distances[:, k].min())
plt.plot(x, kth)
plt.title('min kth neighbour')
plt.show()

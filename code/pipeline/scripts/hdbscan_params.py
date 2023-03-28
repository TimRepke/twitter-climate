import hdbscan
import numpy as np
from typing import Literal

layout = np.load('data/climate/topics/layout_10000_TSNEArgs.npy')

cluster_selection_epsilon: float = 0.0
alpha: float = 1.0
cluster_selection_method: Literal['eom', 'leaf'] = 'leaf'
allow_single_cluster: bool = False

for min_cluster_size in [5, 10, 15, 20, 30, 50, 100]:
    for min_samples in [5, 10, 15, 20, 30, 50, 100]:
        for cluster_selection_method in ['eom', 'leaf']:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                alpha=alpha,
                cluster_selection_method=cluster_selection_method,
                allow_single_cluster=allow_single_cluster
            )
            clusterer.fit(layout)
            labels = clusterer.labels_ + 1
            print('---', min_samples, min_cluster_size, cluster_selection_method, '---')
            cnts = np.unique(labels, return_counts=True)[1]
            print(f'num clusters: {len(cnts)}; std: {cnts.std()}; sum: {cnts.sum()}')
            print(cnts)

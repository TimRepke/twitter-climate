import hdbscan
import numpy as np
from typing import Literal
from matplotlib import pyplot as plt

layout = np.load('data/climate2/topics/layout_100000_TSNEArgs.npy')

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=2,
    min_samples=10,
    cluster_selection_epsilon=1,
    cluster_selection_method='eom',  # 'leaf' or 'eom'
    allow_single_cluster=False,
    alpha=1.,
)
clusterer.fit(layout)
tree = clusterer.single_linkage_tree_

cutoffs = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3]  # , 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10]
csizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100, 150, 200]
for cutoff in cutoffs:
    for min_cluster_size in csizes:
        cnts = np.unique(tree.get_clusters(cut_distance=cutoff, min_cluster_size=min_cluster_size),
                         return_counts=True)[1]
        if len(cnts) > 30:
            print(f'cutoff: {cutoff}, min_size: {min_cluster_size:,}, num clusters: {len(cnts):,}, '
                  f'mean c_size: {cnts[1:].mean():.1f}, median: {np.median(cnts[1:]):.1f}, '
                  f'outliers: {cnts[0]:,}, largest cluster: {cnts[1:].max():,}')
print('done')

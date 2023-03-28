import os
import json
import numpy as np
from typing import Literal
import plotly.graph_objects as go
from colorcet import glasbey
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, AgglomerativeClustering
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.stats import wasserstein_distance


def emd(a, b):
    earth = 0
    earth1 = 0
    s = len(a)
    diff_array = a - b
    for j in range(s):
        earth = (earth + diff_array[j])
        earth1 += abs(earth)
    return earth1 / s


DATASET = 'climate2'
LIMIT = 10000
TARGET_DIR = f'data/{DATASET}/topics'
DATE_FORMAT: Literal['monthly', 'yearly', 'weekly', 'daily'] = 'monthly'

EPS = 1e-12
N_CLUSTERS = 8

for boost in [[], ['retweets'], ['replies'], ['likes'], ['retweets', 'likes']]:
    for norm in ['col', 'row', 'abs']:
        with open(f'{TARGET_DIR}/temporal/tt_{LIMIT}_{DATE_FORMAT}_abs_{"_".join(boost)}.json') as f:
            data = json.load(f)
            vectors = np.array(data['z'])[1:]
            topics = data['y']
            groups = data['x']

        vectors = vectors.clip(max=np.percentile(vectors, 95))

        if norm == 'row':
            vectors = vectors / (vectors.sum(axis=0) + EPS)
        elif norm == 'col':
            vectors = (vectors.T / (vectors.sum(axis=1) + EPS)).T
        elif norm == 'both':
            vectors = vectors / (vectors.sum(axis=0) + EPS)
            vectors = (vectors.T / (vectors.sum(axis=1) + EPS)).T
        # elif norm == 'abs':
        #     pass

        # Plot the distances
        # for index, metric in enumerate(["cosine", "euclidean", "cityblock", 'jaccard',
        #                                 'canberra', 'correlation', wasserstein_distance, emd]):
        #     dists = pairwise_distances(vectors, vectors, metric=metric)
        #     plt.imshow(dists, interpolation="nearest", cmap=plt.cm.gnuplot2, vmin=0)
        #     plt.colorbar()
        #     plt.suptitle(f"Interclass {metric} distances for {norm}", size=18)
        #     plt.tight_layout()
        #     plt.show()

        # kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit(vectors)
        # labels = kmeans.labels_
        model = AgglomerativeClustering(n_clusters=N_CLUSTERS,
                                        affinity='precomputed',
                                        linkage='average')
        dists = pairwise_distances(vectors, vectors, metric=emd)
        model.fit(dists)
        labels = model.labels_

        # for c in np.unique(labels):
        #     for di in np.argwhere(labels == c):
        #         print(c, di[0])
        #         print(data['y'][di[0]])
        fig = go.Figure(go.Heatmap(
            z=np.vstack([vectors[labels == c] for c in np.unique(labels)]),
            x=[f'd:{d}' for d in data['x']],
            y=[data['y'][di[0]] for c in np.unique(labels) for di in np.argwhere(labels == c)],
            hoverongaps=False))
        print(np.unique(labels, return_counts=True))
        pos = 0
        for c, cnt in zip(*np.unique(labels, return_counts=True)):
            fig.add_hline(y=pos,
                          line_dash="dot",
                          annotation_text=f'cluster {c}',
                          annotation_position="bottom right",
                          annotation_font_size=10,
                          annotation_font_color="blue")
            pos += cnt
        # fig = make_subplots(rows=N_CLUSTERS, cols=1)
        # for i in range(N_CLUSTERS):
        #     fig.add_trace(go.Heatmap(
        #         z=vectors[labels == i],
        #         x=[f'd:{d}' for d in data['x']],
        #         y=[d for di, d in enumerate(data['y']) if labels[di] == i],
        #         hoverongaps=False), row=i + 1, col=1)

        fig.update_layout(title=f'Norm: {norm}, boost: {"_".join(boost)}, '
                                f'cluster dist: {np.unique(labels, return_counts=True)}')
        fig.show()
    for i, d in enumerate(data['x']):
        print(i, d)
    exit()

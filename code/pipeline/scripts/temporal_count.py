import os
import json
import numpy as np
from typing import Literal
import plotly.graph_objects as go
from colorcet import glasbey
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

DATASET = 'climate2'
LIMIT = 7000000
SOURCE_DIR = f'data/{DATASET}/topics_big'
TARGET_DIR = f'data/{DATASET}/topics_big/temporal_count'
os.makedirs(TARGET_DIR, exist_ok=True)

DATE_FORMAT: Literal['monthly', 'yearly', 'weekly', 'daily'] = 'monthly'
EPS = 1e-12

N_CLUSTERS = 4

for boost in [[], ['retweets', 'likes']]:  # ['retweets'], ['replies'], ['likes'],
    for norm in ['col', 'row', 'abs']:  # 'both',
        with open(f'{SOURCE_DIR}/temporal_{LIMIT}_{DATE_FORMAT}_{"_".join(boost or ["raw"])}_abs.json') as f:
            data = json.load(f)
            vectors = np.array(data['z'])[1:]
            topics = data['y']
            groups = data['x']
        print(norm, boost)
        print(vectors.sum(axis=1))
        print(np.median(vectors), np.max(vectors), np.mean(vectors),
              np.percentile(vectors, 90), np.percentile(vectors, 95), np.percentile(vectors, 99))
        vectors = vectors.clip(max=np.percentile(vectors, 98))
        # vectors[:,11] = 0

        if norm == 'row':
            vectors = vectors / (vectors.sum(axis=0) + EPS)
        elif norm == 'col':
            vectors = (vectors.T / (vectors.sum(axis=1) + EPS)).T
        elif norm == 'both':
            vectors = vectors / (vectors.sum(axis=0) + EPS)
            vectors = (vectors.T / (vectors.sum(axis=1) + EPS)).T
        # elif norm == 'abs':
        #     pass

        print(len(topics))
        print(vectors.shape)
        splits = [
            0,  # 0 2018-01
            6,  # 5 2018-07
            12,  # 12 2019-01
            18,  # 17 2019-07
            24,  # 24 2020-01
            30,  # 29 2020-07
            36,  # 36 2021-01
            42,  # 41 2021-07
            47  # 47 2021-12
        ]

        slices = np.vstack([
            vectors[:, i:j].sum(axis=1)
            for i, j in zip(splits[:-1], splits[1:])
        ])
        print(slices.shape)
        labels = np.argmax(slices, axis=0)
        print(labels)

        # bc = vectors[:, :24].sum(axis=1)
        # ac = vectors[:, 24:].sum(axis=1)
        # labels = np.zeros((len(topics),), dtype=int)
        # labels[ac >= bc] = 1

        # print(vectors.sum(axis=1))
        fig = go.Figure(go.Heatmap(
            # z=np.vstack([vectors[ac < bc], vectors[ac >= bc]]),
            z=np.vstack([vectors[labels == c][vectors[labels == c].sum(axis=1).argsort()] for c in np.unique(labels)]),
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
        for s in splits[1:-1]:
            fig.add_vline(x=s,
                          line_dash="dot",
                          annotation_text=f'split {s}',
                          annotation_position="top right",
                          annotation_font_size=10,
                          annotation_font_color="blue")

        fig.update_layout(title=f'Norm: {norm}, boost: {"_".join(boost or ["raw"])}, '
                                f'cluster dist: {np.unique(labels, return_counts=True)}')
        # fig.show()
        fig.write_html(f'{TARGET_DIR}/splits_{LIMIT}_{DATE_FORMAT}_{"_".join(boost or ["raw"])}_{norm}.html')

        x = np.arange(len(groups))
        for c in np.unique(labels):
            y = vectors[labels == c].sum(axis=0)
            plt.plot(x, y, label=f'cluster {c}')
        plt.title(f'Norm: {norm}, boost: {"_".join(boost or ["raw"])}')
        plt.legend()
        plt.show()

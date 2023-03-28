from scripts.util import read_supertopics, SuperTopic, get_spottopics, DateFormat, read_temp_dist, smooth
from typing import Literal, Optional
import numpy as np
from matplotlib import pyplot as plt
import re
import seaborn as sns
import csv
from itertools import chain, repeat
import pandas as pd
from sklearn.neighbors import KernelDensity
from multiprocessing import Pool

BOOST = ['raw',  # 0
         'retweets',  # 1
         'replies',  # 2
         'likes',  # 3
         'retweets_likes',  # 4
         'replies_likes',  # 5
         'retweets_replies',  # 6
         'retweets_likes_replies'  # 7
         ][0]

RELEVANCE_FILE = 'data/climate2/tweets_relevant_True_True_4_5_2018-01_2022-01_True_False_False.txt'
FILE_SUPERTOPICS = 'data/climate2/topics_big2/supertopics.csv'
FILE_LABELS = 'data/climate2/topics_big2/full_batched/labels.csv'
FILE_TSNE = 'data/climate2/tsne_full.csv'

print('Reading relevance...')
with open(RELEVANCE_FILE) as f:
    RELEVANT = set(int(line) for line in f)
print('Reading labels...')
LABELS_FULL = pd.read_csv(FILE_LABELS, index_col=0, names=['d', 'km', 'kp', 'fm', 'fp'])
KEY = 'km'
print('Reading tsne...')
TSNE_FULL = pd.read_csv(FILE_TSNE, index_col=0, names=['day', 'new', 'success', 'x', 'y'])
print('Reading annotations...')
annotations = read_supertopics(FILE_SUPERTOPICS)

print('Filtering for relevance...')
LABELS_FULL = LABELS_FULL.iloc[list(RELEVANT)]
print('Joining tables...')
DATA = LABELS_FULL.join(TSNE_FULL, how='left')
print('Assigning supertopics...')
for st in SuperTopic:
    keys = set(np.argwhere(annotations[:, st] > 0).reshape(-1, ))
    DATA[st.name] = DATA[KEY].map(lambda x: x in keys)
    print(f'  > {st.name}: {len(DATA[DATA[st.name]]):,} tweets')

BOUNDARY = '2020-03-01'
EPS = 1e-12

print('Find plot dimensions')
xmin = DATA.x.min()
xmax = DATA.x.max()
ymin = DATA.y.min()
ymax = DATA.y.max()
xbins = 100j
ybins = 100j

xx, yy = np.mgrid[xmin:xmax:xbins, ymin:ymax:ybins]
xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T

sts_plot = [(SuperTopic.Contrarian, 'red'), (SuperTopic.Causes, 'black'),
            (SuperTopic.POLITICS, 'blue'), (SuperTopic.Solutions, 'green'),
            (SuperTopic.Impacts, 'hotpink'), (SuperTopic.Movements, 'orange'),
            (SuperTopic.COVID, 'firebrick')]

print('Computing densities...')
densities = {}
for st, col in sts_plot:
    print(f'  > {st.name}')
    st_data = DATA[DATA[st.name]]

    x = st_data.x
    y = st_data.y
    xy_train = np.vstack([y, x]).T

    print('   - fitting')
    kde = KernelDensity(kernel='gaussian', metric='euclidean',
        bandwidth=2.0, atol=0.0005, rtol=0.01)
    kde.fit(xy_train)
    n_threads = 30
    print('   - scoring')
    with Pool(n_threads) as p:
        z = np.concatenate(p.map(kde.score_samples, np.array_split(xy_sample, n_threads)))

    z = np.exp(z)
    zz = np.reshape(z, xx.shape)
    densities[st.name] = zz

print('Plotting...')
plt.figure(figsize=(15, 15), dpi=150)
for st, col in sts_plot:
    st_data = DATA[DATA[st.name]]
    x = st_data.x
    y = st_data.y
    plt.scatter(x, y, marker='X', alpha=0.1, c=col, s=0.1)
    c = plt.contour(xx, yy, densities[st.name], 3, colors=col)
    fmt = {}
    for l in c.levels:
        fmt[l] = st.name
    plt.clabel(c, c.levels, fmt=fmt, inline=True, fontsize=10)

plt.savefig('data/climate2/figures/landscape.png')
plt.show()

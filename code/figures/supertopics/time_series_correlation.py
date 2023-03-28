from scripts.util import read_supertopics, SuperTopic, get_spottopics, DateFormat, read_temp_dist, smooth
from typing import Literal, Optional
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
import seaborn as sns
import csv
from itertools import chain, repeat
from scipy import stats

FILE_SUPERTOPICS = f'data/climate2/topics_big2/supertopics.csv'

FILE_TEMP_DIST = 'data/climate2/topics_big2/temporal_keep_majority/daily/temporal_daily_raw_abs.json'
NORM_SUM = ['all', 'relev'][1]
BOUNDARY = '2020-03-01'
SMOOTHING = 90
EPS = 1e-12

annotations = read_supertopics(FILE_SUPERTOPICS)

td_groups, td_topics, td_counts = read_temp_dist(FILE_TEMP_DIST)
supertopic_counts = []
for st in SuperTopic:
    if NORM_SUM == 'all' or st not in [SuperTopic.Interesting, SuperTopic.NotRelevant, SuperTopic.Other]:
        t_counts = td_counts.T[annotations[:, st] > 0].sum(axis=0)
        supertopic_counts.append(t_counts)
        print(st.name, f'{t_counts.sum():,}')

supertopic_counts = np.array(supertopic_counts)
BOUND = td_groups.index(BOUNDARY)
sts_plot = [SuperTopic.COVID, SuperTopic.Causes, SuperTopic.Impacts, SuperTopic.Solutions,
            SuperTopic.POLITICS, SuperTopic.Movements, SuperTopic.Contrarian,
            # SuperTopic.Other,  # SuperTopic.Interesting, SuperTopic.NotRelevant
            ]

tweets_per_day = np.sum(td_counts, axis=1)
tweets_per_topic = np.sum(td_counts, axis=0)

x = np.arange(len(td_groups))
xticks = []
xticklabels = []
for i, g in enumerate(td_groups):
    s = g.split('-')
    if int(s[1]) % 3 == 0 and int(s[2]) == 1:
        xticks.append(i)
        xticklabels.append(g)

ylims = {
    'abs': (0, 3000),
    'share': (0, 0.2),
    'self': (0, 0.1)
}

time_series = {}
ts_counts = {}

fig = plt.figure(figsize=(8, 20))
for i, st in enumerate(sts_plot, start=1):
    st_distributions = td_counts.T[annotations[:, st] > 0]
    st_daily_counts = st_distributions.sum(axis=0)
    n_st_tweets = td_counts.T[annotations[:, st] > 0].T
    y = st_daily_counts / (tweets_per_day + EPS)
    y_smooth = smooth([y], kernel_size=SMOOTHING, with_pad=True)[0]

    time_series[st.name] = y_smooth
    ts_counts[st.name] = st_daily_counts

    ax = plt.subplot(len(sts_plot), 1, i)
    ax.set_title(f'{st.name} ({n_st_tweets.shape[1]} topics)')
    ax.set_xticks(xticks)
    ax.set_xticklabels([tl[:7] for tl in xticklabels], rotation=45, fontsize=8)
    ax.plot(x, y_smooth, color='black')
plt.show()

corrs = np.zeros((len(time_series), len(time_series)))
for i1, (n1, ts1) in enumerate(time_series.items()):
    for i2, (n2, ts2) in enumerate(time_series.items()):
        if n1 != n2:
            p_corr = stats.pearsonr(ts1, ts2)
            print(f'{n1} to {n2}: pearson={p_corr[0]:.3f} | {p_corr[1]:.3f}')
            corrs[i1][i2] = p_corr[0]

fig = plt.figure(figsize=(10, 10))
plt.imshow(corrs, interpolation='none', aspect='equal', origin='lower', cmap='seismic', vmin=-1, vmax=1)
for (j, i), label in np.ndenumerate(corrs):
    plt.text(i, j, f'{label:.3f}', ha='center', va='center')
plt.xticks(np.arange(len(time_series)), time_series.keys())
plt.yticks(np.arange(len(time_series)), time_series.keys())
plt.tight_layout()
plt.colorbar()
plt.show()

print('|Topic | share: mean (std) | pre share | post share | count: mean (std) | pre count | post count |')
print('|----|----|----|----|----|----|----|')
for n, ts in time_series.items():
    ts2 = ts_counts[n]

    print(f'| {n} | '
          f'{np.mean(ts):.3f} ({np.std(ts):.3f}) | '
          f'{np.mean(ts[:BOUND]):.3f} ({np.std(ts[:BOUND]):.3f}) | '
          f'{np.mean(ts[BOUND:]):.3f} ({np.std(ts[BOUND:]):.3f}) | '
          f'{np.mean(ts2):.0f} ({np.std(ts2):.0f}) | '
          f'{np.mean(ts2[:BOUND]):.0f} ({np.std(ts2[:BOUND]):.0f}) | '
          f'{np.mean(ts2[BOUND:]):.0f} ({np.std(ts2[BOUND:]):.0f}) |')

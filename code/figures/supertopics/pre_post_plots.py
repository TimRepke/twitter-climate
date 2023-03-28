from scripts.util import read_supertopics, SuperTopic, get_spottopics, DateFormat, read_temp_dist, smooth
from typing import Literal, Optional
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
import seaborn as sns
import csv
from itertools import chain, repeat

BOOST = ['raw',  # 0
         'retweets',  # 1
         'replies',  # 2
         'likes',  # 3
         'retweets_likes',  # 4
         'replies_likes',  # 5
         'retweets_replies',  # 6
         'retweets_likes_replies'  # 7
         ][0]

FILE_SUPERTOPICS = f'data/climate2/topics_big2/supertopics.csv'

FILES_TEMP_DIST = {
    'keep (majority)': f'data/climate2/topics_big2/temporal_keep_majority/daily/temporal_daily_{BOOST}_abs.json',
    'fresh (majority)': f'data/climate2/topics_big2/temporal_fresh_majority/daily/temporal_daily_{BOOST}_abs.json'
}
DT = ['keep (majority)', 'fresh (majority)'][0]
FILE_TEMP_DIST = FILES_TEMP_DIST[DT]
NORM_SUM = ['all', 'relev'][1]
BOUNDARY = '2020-03-01'
SMOOTHING = 90
EPS = 1e-12

annotations = read_supertopics(FILE_SUPERTOPICS)

td_groups, td_topics, td_counts = read_temp_dist(FILE_TEMP_DIST)
supertopic_counts = []
for st in SuperTopic:
    if NORM_SUM == 'all' and st not in [SuperTopic.Interesting, SuperTopic.NotRelevant] or st not in [SuperTopic.Interesting, SuperTopic.NotRelevant, SuperTopic.Other]:
        t_counts = td_counts.T[annotations[:, st] > 0].sum(axis=0)
        supertopic_counts.append(t_counts)
        print(st.name, f'{t_counts.sum():,}')

supertopic_counts = np.array(supertopic_counts)
BOUND = td_groups.index(BOUNDARY)
sts_plot = [SuperTopic.COVID, SuperTopic.Causes, SuperTopic.Impacts, SuperTopic.Solutions,
            SuperTopic.Politics, SuperTopic.Movements, SuperTopic.Contrarian,
            # SuperTopic.Other,
            # SuperTopic.Interesting,
            # SuperTopic.NotRelevant,
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

pre_post_counts = {}
for i, st in enumerate(sts_plot, start=1):
    st_distributions = td_counts.T[annotations[:, st] > 0]
    st_daily_counts = st_distributions.sum(axis=0)

    pre_post_counts[st.name] = (st_daily_counts[:BOUND], st_daily_counts[BOUND:])

pre_post_counts_np = np.zeros((2, len(sts_plot)))
for i, (st, counts) in enumerate(pre_post_counts.items()):
    pre_post_counts_np[0][i] = counts[0].sum()
    pre_post_counts_np[1][i] = counts[1].sum()

yearly_counts = {'2018': {}, '2019': {}, '2020': {}, '2021': {}}
yearly_counts_np = np.zeros((4, len(sts_plot)))
for i, st in enumerate(sts_plot, start=1):
    for yi, yr in enumerate(yearly_counts.keys()):
        st_distributions = td_counts.T[annotations[:, st] > 0]
        st_daily_counts = st_distributions.sum(axis=0)
        yearly_counts[yr][st.name] = st_daily_counts[[gi for gi, g in enumerate(td_groups) if g[:4] == yr]]
        yearly_counts_np[yi][i - 1] = yearly_counts[yr][st.name].sum()

fig = plt.figure(figsize=(6, 12))
fig.suptitle('Number of tweets/supertopic before & after Mar 2020 (abs)')
labels = ['pre Mar 2020', 'post Mar 2020']
bottom = np.zeros((2,))
for i, st in enumerate(sts_plot):
    plt.bar(labels, pre_post_counts_np[:, i], width=0.4, label=st.name, bottom=bottom)
    bottom += pre_post_counts_np[:, i]
plt.legend()
plt.savefig(f'data/climate2/figures/pre_post/pre_post_abs.png')
plt.show()

fig = plt.figure(figsize=(6, 12))
fig.suptitle('Number of tweets/supertopic before & after Mar 2020 (normed)')
labels = ['pre Mar 2020', 'post Mar 2020']
bottom = np.zeros((2,))
for i, st in enumerate(sts_plot):
    plt.bar(labels, pre_post_counts_np[:, i] / pre_post_counts_np.sum(axis=1), width=0.4, label=st.name, bottom=bottom)
    bottom += pre_post_counts_np[:, i] / pre_post_counts_np.sum(axis=1)
plt.legend()
plt.savefig(f'data/climate2/figures/pre_post/pre_post_normed.png')
plt.show()

fig = plt.figure(figsize=(6, 12))
fig.suptitle('Number of tweets/supertopic per year (abs)')
labels = ['2018', '2019', '2020', '2021']
bottom = np.zeros((4,))
for i, st in enumerate(sts_plot):
    plt.bar(labels, yearly_counts_np[:, i], width=0.4, label=st.name, bottom=bottom)
    bottom += yearly_counts_np[:, i]
plt.legend()
plt.savefig(f'data/climate2/figures/pre_post/yrs_abs.png')
plt.show()

fig = plt.figure(figsize=(6, 12))
fig.suptitle('Number of tweets/supertopic per year (normed)')
labels = ['2018', '2019', '2020', '2021']
bottom = np.zeros((4,))
for i, st in enumerate(sts_plot):
    plt.bar(labels, yearly_counts_np[:, i] / (yearly_counts_np.sum(axis=1) + EPS),
            width=0.4, label=st.name, bottom=bottom)
    bottom += yearly_counts_np[:, i] / (yearly_counts_np.sum(axis=1) + EPS)
plt.legend()
plt.savefig(f'data/climate2/figures/pre_post/yrs_normed.png')
plt.show()

#############################################################################################
# same, but with numbers
#############################################################################################

fig = plt.figure(figsize=(6, 12))
fig.suptitle('Number of tweets/supertopic before & after Mar 2020 (abs)')
labels = ['pre Mar 2020', 'post Mar 2020']
bottom = np.zeros((2,))
for i, st in enumerate(sts_plot):
    bar = plt.bar(labels, pre_post_counts_np[:, i], width=0.4, label=st.name, bottom=bottom)
    plt.bar_label(bar, fmt='%d', padding=0, label_type='center', rotation='horizontal')
    bottom += pre_post_counts_np[:, i]
plt.legend()
plt.savefig(f'data/climate2/figures/pre_post/pre_post_abs_numbers.png')
plt.show()

fig = plt.figure(figsize=(6, 12))
fig.suptitle('Number of tweets/supertopic before & after Mar 2020 (normed)')
labels = ['pre Mar 2020', 'post Mar 2020']
bottom = np.zeros((2,))
for i, st in enumerate(sts_plot):
    bar = plt.bar(labels, pre_post_counts_np[:, i] / pre_post_counts_np.sum(axis=1), width=0.4, label=st.name, bottom=bottom)
    plt.bar_label(bar, fmt='%.2f', padding=0, label_type='center', rotation='horizontal')
    bottom += pre_post_counts_np[:, i] / pre_post_counts_np.sum(axis=1)
plt.legend()
plt.savefig(f'data/climate2/figures/pre_post/pre_post_normed_numbers.png')
plt.show()

fig = plt.figure(figsize=(6, 12))
fig.suptitle('Number of tweets/supertopic per year (abs)')
labels = ['2018', '2019', '2020', '2021']
bottom = np.zeros((4,))
for i, st in enumerate(sts_plot):
    bar = plt.bar(labels, yearly_counts_np[:, i], width=0.4, label=st.name, bottom=bottom)
    plt.bar_label(bar, fmt='%d', padding=0, label_type='center', rotation='horizontal')
    bottom += yearly_counts_np[:, i]
plt.legend()
plt.savefig(f'data/climate2/figures/pre_post/yrs_abs_numbers.png')
plt.show()

fig = plt.figure(figsize=(6, 12))
fig.suptitle('Number of tweets/supertopic per year (normed)')
labels = ['2018', '2019', '2020', '2021']
bottom = np.zeros((4,))
for i, st in enumerate(sts_plot):
    bar = plt.bar(labels, yearly_counts_np[:, i] / (yearly_counts_np.sum(axis=1) + EPS),
            width=0.4, label=st.name, bottom=bottom)
    plt.bar_label(bar, fmt='%.2f', padding=0, label_type='center', rotation='horizontal')
    bottom += yearly_counts_np[:, i] / (yearly_counts_np.sum(axis=1) + EPS)
plt.legend()
plt.savefig(f'data/climate2/figures/pre_post/yrs_normed_numbers.png')
plt.show()

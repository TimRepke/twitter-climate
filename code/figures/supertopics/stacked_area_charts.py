from scripts.util import read_supertopics, SuperTopic, get_spottopics, DateFormat, read_temp_dist, smooth
from typing import Literal, Optional
import numpy as np
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
FILE_TEMP_DIST = FILES_TEMP_DIST[['keep (majority)', 'fresh (majority)'][0]]
BOUNDARY = '2020-03-01'
SMOOTHING = 90
EPS = 1e-12
annotations = read_supertopics(FILE_SUPERTOPICS)

sts_plot = [SuperTopic.COVID, SuperTopic.Causes, SuperTopic.Impacts, SuperTopic.Solutions,
            SuperTopic.POLITICS, SuperTopic.Movements, SuperTopic.Contrarian,
            # SuperTopic.Other,  # SuperTopic.Interesting, SuperTopic.NotRelevant
            ]

td_groups, td_topics, td_counts = read_temp_dist(FILE_TEMP_DIST)
supertopic_counts = []
supertopic_counts_all = []
for st in SuperTopic:
    t_counts = td_counts.T[annotations[:, st] > 0].sum(axis=0)
    supertopic_counts_all.append(t_counts)
    if st in sts_plot:
        supertopic_counts.append(t_counts)
    print(st.name, f'{t_counts.sum():,}')

supertopic_counts = np.array(supertopic_counts)
supertopic_counts_all = np.array(supertopic_counts_all)
BOUND = td_groups.index(BOUNDARY)

tweets_per_day = np.sum(td_counts, axis=1)
tweets_per_topic = np.sum(td_counts, axis=0)
st_plot_counts = supertopic_counts
st_plot_shares = st_plot_counts / supertopic_counts.sum(axis=0)
st_plot_shares_all = st_plot_counts / tweets_per_day
st_plot_shares_smooth = smooth(st_plot_shares, kernel_size=SMOOTHING)
st_plot_shares_smooth_all = smooth(st_plot_shares_all, kernel_size=SMOOTHING)

x = np.arange(len(td_groups))
xticks = []
xticklabels = []
for i, g in enumerate(td_groups):
    s = g.split('-')
    if int(s[1]) % 3 == 0 and int(s[2]) == 1:
        xticks.append(i)
        xticklabels.append(g)

fig = plt.figure(figsize=(10, 10))
fig.suptitle('Smoothed daily share of tweets per supertopic')
plt.stackplot(x, st_plot_shares_smooth * 100,
    baseline=['zero', 'sym', 'wiggle'][0],
    labels=[st.name for st in sts_plot])
plt.legend()
plt.ylabel('% of tweets')
plt.xticks(xticks, xticklabels, rotation=30)
plt.savefig('data/climate2/figures/stacked_areas/stacked_supertopics.png')
plt.show()

fig = plt.figure(figsize=(10, 10))
fig.suptitle('Smoothed daily share of tweets per supertopic (normed by all)')
plt.stackplot(x, st_plot_shares_smooth_all * 100,
    baseline=['zero', 'sym', 'wiggle'][0],
    labels=[st.name for st in sts_plot])
plt.legend()
plt.ylabel('% of tweets')
plt.xticks(xticks, xticklabels, rotation=30)
plt.savefig('data/climate2/figures/stacked_areas/stacked_supertopics_rel2all.png')
plt.show()

fig = plt.figure(figsize=(10, 20))
fig.suptitle('Number of tweets per topic per day')
for i, st in enumerate(sts_plot, start=1):
    n_st_tweets = td_counts.T[annotations[:, st] > 0].T

    ax = plt.subplot(len(sts_plot), 1, i)
    ax.stackplot(x, smooth(n_st_tweets.T, kernel_size=SMOOTHING))
    ax.set_title(f'{st.name} ({n_st_tweets.shape[1]} topics)')
    ax.set_xticks(xticks)
    ax.set_xticklabels([tl[:7] for tl in xticklabels], rotation=45, fontsize=8)
plt.tight_layout()
plt.savefig('data/climate2/figures/stacked_areas/abs_tweets_per_supertopic.png')
plt.show()

fig = plt.figure(figsize=(10, 20))
fig.suptitle('Share of tweets per topic per day (normed by relevant STs)')
for i, st in enumerate(sts_plot, start=1):
    n_st_tweets = td_counts.T[annotations[:, st] > 0].T
    n_st_tweets_per_day = n_st_tweets.T/supertopic_counts.sum(axis=0)

    ax = plt.subplot(len(sts_plot), 1, i)
    ax.stackplot(x, smooth(n_st_tweets_per_day, kernel_size=SMOOTHING))
    ax.set_title(f'{st.name} ({n_st_tweets.shape[1]} topics)')
    ax.set_xticks(xticks)
    ax.set_xticklabels([tl[:7] for tl in xticklabels], rotation=45, fontsize=8)
plt.tight_layout()
plt.savefig('data/climate2/figures/stacked_areas/rel_tweets_per_supertopic.png')
plt.show()


fig = plt.figure(figsize=(10, 20))
fig.suptitle('Share of tweets per topic per day (normed by respective STs)')
for i, st in enumerate(sts_plot, start=1):
    n_st_tweets = td_counts.T[annotations[:, st] > 0].T
    n_st_tweets_per_day = n_st_tweets.sum(axis=1)

    ax = plt.subplot(len(sts_plot), 1, i)
    ax.stackplot(x, smooth(n_st_tweets.T / (n_st_tweets_per_day + EPS), kernel_size=SMOOTHING))
    ax.set_title(f'{st.name} ({n_st_tweets.shape[1]} topics)')
    ax.set_xticks(xticks)
    ax.set_xticklabels([tl[:7] for tl in xticklabels], rotation=45, fontsize=8)
plt.tight_layout()
plt.savefig('data/climate2/figures/stacked_areas/self_tweets_per_supertopic.png')
plt.show()

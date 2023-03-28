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
# 18.12.2019: first case
# 31.12.2019: cases reported to WHO
# 30.01.2020: public health emergency by WHO
# 11.03.2020: pandemic declared by WHO
BOUNDARY = '2020-01-01'
SHADE = ['2019-12-18', '2020-03-11']
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
SHADE_BOUNDS = [td_groups.index(SHADE[0]), td_groups.index(SHADE[1])]
sts_plot = [SuperTopic.COVID, SuperTopic.Causes, SuperTopic.Impacts, SuperTopic.Solutions,
            SuperTopic.Politics, SuperTopic.Movements, SuperTopic.Contrarian,
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
    'self': (0, 0.003)
}

for y_shared in [True, False]:
    for mode in 'abs', 'share', 'self':
        fig = plt.figure(figsize=(8, 20))
        fig.suptitle(f'{DT} | {BOOST} | {mode} | normed by {NORM_SUM}', y=1)
        for i, st in enumerate(sts_plot, start=1):
            st_distributions = td_counts.T[annotations[:, st] > 0]
            st_daily_counts = st_distributions.sum(axis=0)
            n_st_tweets = td_counts.T[annotations[:, st] > 0].T

            if mode == 'abs':
                y = st_daily_counts
            elif mode == 'share':
                y = st_daily_counts / (tweets_per_day + EPS)
            else:  # mode == 'self'
                y = st_daily_counts / st_daily_counts.sum()

            y_smooth = smooth([y], kernel_size=SMOOTHING, with_pad=True)[0]
            threshold = y.mean()

            ax = plt.subplot(len(sts_plot), 1, i)
            ax.set_title(f'{st.name} ({n_st_tweets.shape[1]} topics)')
            ax.set_xticks(xticks)
            ax.set_xticklabels([tl[:7] for tl in xticklabels], rotation=45, fontsize=8)

            if y_shared:
                ax.set_ylim(*ylims[mode])
                y_share_tag = '_sharedY'
            else:
                y_share_tag = ''

            ax.axhline(threshold, color='black', ls='--', lw=2, alpha=0.5)
            # ax.axvline(BOUND, color='black', lw=2, alpha=0.5)
            plt.axvspan(SHADE_BOUNDS[0], SHADE_BOUNDS[1], color='black', alpha=0.1, lw=0)
            ax.axvline(SHADE_BOUNDS[0], color='black', lw=1, alpha=0.2)
            ax.axvline(SHADE_BOUNDS[1], color='black', lw=1, alpha=0.2)

            # ax.fill_between(x, threshold, y_smooth, where=y_smooth > threshold, color='green', alpha=0.5)
            # ax.fill_between(x, y_smooth, threshold, where=y_smooth < threshold, color='red', alpha=0.5)
            ax.fill_between(x, threshold, y_smooth, where=y_smooth > threshold, color='#92dadd', alpha=0.5)
            ax.fill_between(x, y_smooth, threshold, where=y_smooth < threshold, color='#ffa90e', alpha=0.5)
            ax.plot(x, y_smooth, color='black')

            sns.regplot(x=x[:BOUND], y=y[:BOUND], ax=ax, scatter=False)
            sns.regplot(x=x[BOUND:], y=y[BOUND:], ax=ax, scatter=False)

            plt.setp(ax.collections[2], alpha=0.5)
            plt.setp(ax.collections[3], alpha=0.5)
            # axis.set_ylim(np.percentile(y, q=1), np.percentile(y, q=99))

        plt.tight_layout()

        plt.savefig(f'data/climate2/figures/rg_{DT[:4]}_{mode}_{NORM_SUM}{y_share_tag}.png')
        plt.show()

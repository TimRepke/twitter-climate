import math

from scripts.util import read_supertopics, SuperTopic, get_spottopics, DateFormat, read_temp_dist, smooth
from typing import Literal, Optional
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker
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
SMOOTHING = 14
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
sts_plot = [SuperTopic.COVID,
            SuperTopic.Impacts,
            SuperTopic.Movements,
            SuperTopic.Causes,
            SuperTopic.Contrarian,
            SuperTopic.Solutions,
            SuperTopic.Politics,
            # SuperTopic.Other,  # SuperTopic.Interesting, SuperTopic.NotRelevant
            ]

tweets_per_day = np.sum(td_counts, axis=1)
tweets_per_topic = np.sum(td_counts, axis=0)

x = np.arange(len(td_groups))
xminor = [
    (td_groups.index('2018-04-01'), None),
    (td_groups.index('2018-07-01'), '2018'),
    (td_groups.index('2018-10-01'), None),
    (td_groups.index('2019-04-01'), None),
    (td_groups.index('2019-07-01'), '2019'),
    (td_groups.index('2019-10-01'), None),
    (td_groups.index('2020-04-01'), None),
    (td_groups.index('2020-07-01'), '2020'),
    (td_groups.index('2020-10-01'), None),
    (td_groups.index('2021-04-01'), None),
    (td_groups.index('2021-07-01'), '2021'),
    (td_groups.index('2021-10-01'), None),
]
xmajor = [
    td_groups.index('2018-01-01'),
    td_groups.index('2019-01-01'),
    td_groups.index('2020-01-01'),
    td_groups.index('2021-01-01'),
    td_groups.index('2021-12-14')
]

ylims = {
    'abs': (0, 3000),
    'share': (0, 20),
    'self': (0, 0.3)
}

n_cols = 2
n_rows = math.ceil(len(sts_plot) / n_cols)

for y_shared in [False, True]:
    for mode in ['share', 'abs', 'self']:
        fig = plt.figure(figsize=(20, 10), dpi=300)
        # fig.suptitle(f'{DT} | {BOOST} | {mode} | normed by {NORM_SUM}', y=1)
        for i, st in enumerate(sts_plot, start=1):
            st_distributions = td_counts.T[annotations[:, st] > 0]
            st_daily_counts = st_distributions.sum(axis=0)
            n_st_tweets = td_counts.T[annotations[:, st] > 0].T

            if mode == 'abs':
                y = st_daily_counts
            elif mode == 'share':
                y = (st_daily_counts / (tweets_per_day + EPS)) * 100
            else:  # mode == 'self'
                y = (st_daily_counts / st_daily_counts.sum()) * 100

            y_smooth = smooth([y], kernel_size=SMOOTHING, with_pad=True)[0]
            threshold = y.mean()

            ax = plt.subplot(n_rows, n_cols, i)
            ax.set_title(f'{st.name} ({n_st_tweets.shape[1]} topics)',
                         fontdict={'fontsize': 22})
            ax.set_xticks([tick for tick, _ in xminor], minor=True)
            ax.set_xticks(xmajor, minor=False)
            ax.set_xticklabels([], minor=False)
            ax.tick_params(axis='x', which='major', length=8, width=1.5)
            ax.tick_params(axis='x', which='minor', labelsize=22)
            ax.tick_params(axis='y', labelsize=22, length=5, width=2)
            ax.margins(x=0)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            if mode == 'self':
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
            else:
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
            if st == SuperTopic.Solutions or st == SuperTopic.Politics:
                ax.set_xticklabels([label for _, label in xminor], minor=True, fontsize=20)
                # ax.set_xticklabels([tl[:7] for tl in xticklabels], rotation=45, fontsize=8)
            else:
                ax.set_xticklabels([])

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

        plt.savefig(f'data/climate2/figures/rg_{DT[:4]}_{mode}_{NORM_SUM}{y_share_tag}_2c.png')
        plt.savefig(f'data/climate2/figures/rg_{DT[:4]}_{mode}_{NORM_SUM}{y_share_tag}_2c.pdf')
        plt.show()
    #     break
    # break

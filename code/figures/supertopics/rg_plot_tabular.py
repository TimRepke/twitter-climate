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
EPS = 1e-12

timeframes = [
    ('All', '2018-01-01', '2021-12-14'),
    ('Pre', '2018-01-01', '2019-12-31'),
    ('Post', '2020-01-01', '2021-12-14'),
    ('2018', '2018-01-01', '2018-12-31'),
    ('2019', '2019-01-01', '2019-12-31'),
    ('2020', '2020-01-01', '2020-12-31'),
    ('2021', '2021-01-01', '2021-12-14'),
    ('Q1 2018', '2018-01-01', '2018-03-31'),
    ('Q2 2018', '2018-04-01', '2018-06-30'),
    ('Q3 2018', '2018-07-01', '2018-09-30'),
    ('Q4 2018', '2018-10-01', '2018-12-31'),
    ('Q1 2019', '2019-01-01', '2019-03-31'),
    ('Q2 2019', '2019-04-01', '2019-06-30'),
    ('Q3 2019', '2019-07-01', '2019-09-30'),
    ('Q4 2019', '2019-10-01', '2019-12-31'),
    ('Q1 2020', '2020-01-01', '2020-03-31'),
    ('Q2 2020', '2020-04-01', '2020-06-30'),
    ('Q3 2020', '2020-07-01', '2020-09-30'),
    ('Q4 2020', '2020-10-01', '2020-12-31'),
    ('Q1 2021', '2021-01-01', '2021-03-31'),
    ('Q2 2021', '2021-04-01', '2021-06-30'),
    ('Q3 2021', '2021-07-01', '2021-09-30'),
    ('Q4 2021', '2021-10-01', '2021-12-14'),
]

annotations = read_supertopics(FILE_SUPERTOPICS)

td_groups, td_topics, td_counts = read_temp_dist(FILE_TEMP_DIST)
supertopic_counts = []
for st in SuperTopic:
    if NORM_SUM == 'all' or st not in [SuperTopic.Interesting, SuperTopic.NotRelevant, SuperTopic.Other]:
        t_counts = td_counts.T[annotations[:, st] > 0].sum(axis=0)
        supertopic_counts.append(t_counts)
        print(st.name, f'{t_counts.sum():,}')

supertopic_counts = np.array(supertopic_counts)
sts_plot = [SuperTopic.COVID,
            SuperTopic.Politics,
            SuperTopic.Causes,
            SuperTopic.Movements,
            SuperTopic.Impacts,
            SuperTopic.Contrarian,
            SuperTopic.Solutions,
            # SuperTopic.Other,  # SuperTopic.Interesting, SuperTopic.NotRelevant
            ]

timeframes_ = [
    {'name': timeframe[0],
     'start': timeframe[1], 'start_i': td_groups.index(timeframe[1]),
     'end': timeframe[2], 'end_i': td_groups.index(timeframe[2])}
    for timeframe in timeframes
]

tweets_per_day = np.sum(td_counts, axis=1)
tweets_per_topic = np.sum(td_counts, axis=0)

for i, st in enumerate(sts_plot, start=1):
    print(f'====== {st.name} ======')
    rows = []
    st_distributions = td_counts.T[annotations[:, st] > 0]
    st_daily_counts = st_distributions.sum(axis=0)
    n_st_tweets = td_counts.T[annotations[:, st] > 0].T
    for timeframe in timeframes_:
        q_tweets = st_daily_counts[timeframe['start_i']:timeframe['end_i'] + 1]
        q_tweets_rel = (st_daily_counts / (tweets_per_day + EPS))[timeframe['start_i']:timeframe['end_i'] + 1]
        rows.append({
            'Frame': f'{timeframe["name"]}',
            'SUM': f'{q_tweets.sum():,.0f}',
            'P95-abs': f'{np.percentile(q_tweets, 95):,.0f}',
            'P98-abs': f'{np.percentile(q_tweets, 98):,.0f}',
            'AVG-abs': f'{np.mean(q_tweets):,.0f}',
            'MED-abs': f'{np.median(q_tweets):,.0f}',
            'MIN-abs': f'{np.min(q_tweets):,.0f}',
            'MAX-abs': f'{np.max(q_tweets):,.0f}',
            'STD-abs': f'{np.std(q_tweets):,.0f}',
            'P95-rel': f'{np.percentile(q_tweets_rel, 95):.2f}',
            'P98-rel': f'{np.percentile(q_tweets_rel, 98):.2f}',
            'AVG-rel': f'{np.mean(q_tweets_rel):.3f}',
            'MED-rel': f'{np.median(q_tweets_rel):.3f}',
            'MIN-rel': f'{np.min(q_tweets_rel):.3f}',
            'MAX-rel': f'{np.max(q_tweets_rel):.3f}',
            'STD-rel': f'{np.std(q_tweets_rel):.3f}',
            'AVG': f'{q_tweets.sum()/tweets_per_day[timeframe["start_i"]:timeframe["end_i"] + 1].sum():.3f}'
        })
    print(pd.DataFrame(rows).to_markdown())
    print()

# for mode in ['share', 'abs', 'self']:

#
# for y_shared in [False, True]:
#     for mode in ['share', 'abs', 'self']:
#         fig = plt.figure(figsize=(20, 10), dpi=300)
#         # fig.suptitle(f'{DT} | {BOOST} | {mode} | normed by {NORM_SUM}', y=1)
#         for i, st in enumerate(sts_plot, start=1):
#             st_distributions = td_counts.T[annotations[:, st] > 0]
#             st_daily_counts = st_distributions.sum(axis=0)
#             n_st_tweets = td_counts.T[annotations[:, st] > 0].T
#
#             if mode == 'abs':
#                 y = st_daily_counts
#             elif mode == 'share':
#                 y = (st_daily_counts / (tweets_per_day + EPS)) * 100
#             else:  # mode == 'self'
#                 y = (st_daily_counts / st_daily_counts.sum()) * 100
#
#             threshold = y.mean()
#
#         plt.tight_layout()
#
#         plt.savefig(f'data/climate2/figures/rg_{DT[:4]}_{mode}_{NORM_SUM}{y_share_tag}_2c.png')
#         plt.savefig(f'data/climate2/figures/rg_{DT[:4]}_{mode}_{NORM_SUM}{y_share_tag}_2c.pdf')
#         plt.show()
#     #     break
#     # break

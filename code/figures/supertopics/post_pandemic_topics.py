from scripts.util import read_supertopics, SuperTopic, get_spottopics, DateFormat, read_temp_dist, smooth
from typing import Literal, Optional
import numpy as np
from contextlib import redirect_stdout

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
BOUNDARY = '2020-03-01'
SMOOTHING = 90
EPS = 1e-12

annotations = read_supertopics(FILE_SUPERTOPICS)

td_groups, td_topics, td_counts = read_temp_dist(FILE_TEMP_DIST)
BOUND = td_groups.index(BOUNDARY)
post_pandemic_counts = td_counts[BOUND:].sum(axis=0)
post_pandemic_shares = post_pandemic_counts / td_counts.sum(axis=0)

sts_plot = [SuperTopic.COVID, SuperTopic.Causes, SuperTopic.Impacts, SuperTopic.Solutions,
            SuperTopic.POLITICS, SuperTopic.Movements, SuperTopic.Contrarian,
            # SuperTopic.Other,  # SuperTopic.Interesting, SuperTopic.NotRelevant
            ]

tpc_nms = np.arange(post_pandemic_shares.shape[0])

with open('data/climate2/figures/pre_post/threshold_topics.txt', 'w') as f_out, redirect_stdout(f_out):
    for threshold in [0.6, 0.7, 0.8, 0.9]:
        print('===================================================')
        print(f'====            THRESHOLD = {threshold:.1f}               =====')
        print('===================================================')

        for st in sts_plot:
            print(f' ----> {st.name} <----')
            mask = (annotations[:, st] > 0) & (post_pandemic_shares > threshold)

            shares = post_pandemic_shares[mask]
            counts = post_pandemic_counts[mask]
            topics = tpc_nms[mask]

            print(f' -> {sum(mask)} topics of {sum(annotations[:, st] > 0)} topics in that super-topic '
                  f'({sum(mask) / sum(annotations[:, st] > 0):.2%})')
            print(f' -> {counts.sum():,} tweets of {td_counts.T[annotations[:, st] > 0].sum():,} '
                  f'tweets in that super-topic ({counts.sum()/td_counts.T[annotations[:, st] > 0].sum():.2%})')

            for topic, share, count in sorted(zip(topics, shares, counts), key=lambda e: e[2], reverse=True):
                print(f'   > Topic {topic}: {count:,} tweets past Mar 2020 ({share:.2%} of topic tweets)')

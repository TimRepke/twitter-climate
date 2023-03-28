from scripts.util import read_supertopics, SuperTopic, get_spottopics, DateFormat, read_temp_dist
from typing import Literal
import numpy as np
import json
from prettytable import PrettyTable

DATASET = 'climate2'
LIMIT = 7000000
DATE_FORMAT: DateFormat = 'monthly'
NORM: Literal['abs', 'col', 'row'] = 'abs'
BOOST = ['raw',  # 0
         'retweets',  # 1
         'replies',  # 2
         'likes',  # 3
         'retweets_likes',  # 4
         'replies_likes',  # 5
         'retweets_replies',  # 6
         'retweets_likes_replies'  # 7
         ][0]
# SOURCE_DIR = f'data/{DATASET}/topics_big2'
# TWEETS_FILE = f'data/{DATASET}/tweets_filtered_{LIMIT}.jsonl'
# LABELS_FILE = f'{SOURCE_DIR}/labels_{LIMIT}_tsne.npy'
EPS = 1e-12
FILE_SUPERTOPICS = f'data/{DATASET}/topics_big2/supertopics.csv'
# FILE_TEMP_DIST = f'data/{DATASET}/topics_big2/temporal_sampled/{DATE_FORMAT}/temporal_{LIMIT}_{DATE_FORMAT}_{BOOST}_{NORM}.json'
FILE_TEMP_DIST = 'data/climate2/topics_big2/temporal_keep_majority/monthly/temporal_monthly_raw_abs.json'
print(FILE_TEMP_DIST)
groups, topics, distributions = read_temp_dist(FILE_TEMP_DIST)

annotations = read_supertopics(FILE_SUPERTOPICS)
spot_topics = get_spottopics(distributions, threshold=0.4, min_size=500)

# print(topics)
# print(distributions.sum(axis=0))
print(distributions.shape)
print(annotations.shape)
print(spot_topics.shape)
tab = PrettyTable(field_names=['supertopic', 'N topics', 'N spottopics', 'spots/topics',
                               'N tweets', 'N tweet spot', 'spottweets/tweets', 'avg tweets/topic (std)', 'max peak'])

for st in SuperTopic:
    n_topics = annotations[:, st].sum()
    n_spots = annotations[:, st][spot_topics].sum()
    n_topic_tweets = distributions.T[annotations[:, st] > 0].sum()
    mean_tweets_per_topic = distributions.T[annotations[:, st] > 0].mean()
    std_tweets_per_topic = distributions.T[annotations[:, st] > 0].std()
    n_spot_tweets = distributions.T[spot_topics][annotations[:, st][spot_topics] > 0].sum()
    tab.add_row([st.name,
                 f'{n_topics} ({n_topics / distributions.shape[1]:.1%})',
                 f'{n_spots} ({n_spots / len(spot_topics):.1%})',
                 f'{n_spots / n_topics:.2%}',
                 f'{n_topic_tweets:,} ({n_topic_tweets / (distributions.sum() + EPS):.1%})',
                 f'{n_spot_tweets:,} ({n_spot_tweets / (distributions.T[spot_topics].sum() + EPS):.1%})',
                 f'{n_spot_tweets / n_topic_tweets:.1%}',
                 f'{mean_tweets_per_topic:.1f} ({std_tweets_per_topic:.1f})',
                 groups[distributions.T[annotations[:, st] > 0].sum(axis=0).argmax()]
                 ])

tab.add_row(['TOTAL',
             distributions.shape[1],
             len(spot_topics),
             '––',
             f'{distributions.sum():,}',
             f'{distributions.T[spot_topics].sum():,}',
             f'{distributions.T[spot_topics].sum() / distributions.sum():.1%}',
             f'{distributions.mean():.1f} ({distributions.std():.1f})',
             groups[distributions.T.sum(axis=0).argmax()]
             ])
print(tab)

print('annotated topics:', sum(annotations.sum(axis=1) > 0))
print('num topics:', len(topics))
print('num spot topics:', len(spot_topics))

# when does each spot topic "peak"
r = []
for spt in spot_topics:
    r.append((spt[0], groups[distributions.T[spt].argmax()]))
rs = sorted(r, key=lambda x: x[1])
print(rs)

from scripts.util import read_supertopics, SuperTopic, get_spottopics, DateFormat, read_temp_dist, smooth
from typing import Literal, Optional
import numpy as np
from tqdm import tqdm
import pandas as pd
import json

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
FILE_TWEETS = 'data/climate2/tweets_filtered_15000000.jsonl'
TARGET_FILE = 'data/climate2/tweets_filtered_annotated_all.jsonl'

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

print('Writing filtered and enriched data...')
with open(FILE_TWEETS, 'r') as f_tweets, open(TARGET_FILE, 'w') as f_out:
    for line in tqdm(f_tweets, total=len(LABELS_FULL)):
        tweet = json.loads(line)
        tweet_id = int(tweet['id'])

        if tweet_id in DATA.index:
            data = DATA.loc[tweet_id]
            # majority votes are -1 if no neighbours could be found (https://github.com/nmslib/hnswlib/issues/373)
            tweet['t_km'] = int(data.km)  # topic label (keep old labels, new with majority vote)
            tweet['t_kp'] = int(data.kp)  # topic label (keep old labels, new with proximity vote)
            tweet['t_fm'] = int(data.fm)  # topic label (drop old labels, new with majority vote)
            tweet['t_fp'] = int(data.fp)  # topic label (drop old labels, new with proximity vote)
            # SuperTopic Annotations (based on t_km)
            tweet['st_int'] = int(data[SuperTopic.Interesting.name])
            tweet['st_nr'] = int(data[SuperTopic.NotRelevant.name])
            tweet['st_cov'] = int(data[SuperTopic.COVID.name])
            tweet['st_pol'] = int(data[SuperTopic.POLITICS.name])
            tweet['st_mov'] = int(data[SuperTopic.Movements.name])
            tweet['st_imp'] = int(data[SuperTopic.Impacts.name])
            tweet['st_cau'] = int(data[SuperTopic.Causes.name])
            tweet['st_sol'] = int(data[SuperTopic.Solutions.name])
            tweet['st_con'] = int(data[SuperTopic.Contrarian.name])
            tweet['st_oth'] = int(data[SuperTopic.Other.name])
            # x,y placement on landscape
            tweet['x'] = data.x
            tweet['y'] = data.y
            # True is this tweet was in the sample used to fit the topic model
            tweet['sample'] = not data.new

            f_out.write(json.dumps(tweet) + '\n')

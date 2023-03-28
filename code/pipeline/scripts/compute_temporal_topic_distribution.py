from utils.topics.utils import date2group
import json
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

DATASET = 'climate2'
EPS = 1e-12

SOURCE_DIR = f'data/{DATASET}/topics_big2'
RELEVANCE_FILE = f'data/{DATASET}/tweets_relevant_True_True_4_5_2018-01_2022-01_True_False_False.txt'

FILE_TWEETS_SAMPLED = f'data/{DATASET}/tweets_filtered_7000000.jsonl'
FILE_LABELS_SAMPLED = f'{SOURCE_DIR}/labels_7000000_tsne.npy'
FILE_TARGET_SAMPLED = f'{SOURCE_DIR}/temporal_sampled/'

FILE_TWEETS_FULL = f'data/{DATASET}/tweets_filtered_15000000.jsonl'
FILE_LABELS_FULL = f'{SOURCE_DIR}/full_batched/labels.csv'
FILE_TARGETS_FULL = {'fm': f'{SOURCE_DIR}/temporal_fresh_majority/',
                     'fp': f'{SOURCE_DIR}/temporal_fresh_proximity/',
                     'km': f'{SOURCE_DIR}/temporal_keep_majority/',
                     'kp': f'{SOURCE_DIR}/temporal_keep_proximity/'}

print('Reading labels...')
LABELS_FULL = pd.read_csv(FILE_LABELS_FULL, index_col=0, names=['d', 'km', 'kp', 'fm', 'fp'])
N_TOPICS = 983

with open(RELEVANCE_FILE) as f:
    RELEVANT = set(int(line) for line in f)

# ls = np.load(FILE_LABELS_SAMPLED)
# with open(FILE_TWEETS_SAMPLED) as ft:
#     LABELS_SAMPLED = {json.loads(line)['id']: label for line, label in zip(ft, ls)}

aggregate = {'daily': {}, 'monthly': {}}  # , 'weekly': {}}
COLS = ['km', 'fm']  # , 'fp','kp']
for src in COLS:
    for v in aggregate.values():
        v[src] = {}

print('Reading and aggregating all the data...')
with open(FILE_TWEETS_FULL) as f_tweets:
    for i, line in tqdm(enumerate(f_tweets), total=len(LABELS_FULL)):
        if i not in RELEVANT:
            continue
        tweet = json.loads(line)
        tweet_id = int(tweet['id'])
        for src in COLS:
            topic = LABELS_FULL.loc[tweet_id][src]
            if topic < 0:
                topic = 0

            for date_format in aggregate.keys():
                group = date2group(tweet['created_at'], date_format)

                if group not in aggregate[date_format][src]:
                    aggregate[date_format][src][group] = {
                        'raw': np.zeros((N_TOPICS,)),
                        'retweets': np.zeros((N_TOPICS,)),
                        'likes': np.zeros((N_TOPICS,)),
                        'replies': np.zeros((N_TOPICS,))
                    }

                aggregate[date_format][src][group]['raw'][topic] += 1
                aggregate[date_format][src][group]['retweets'][topic] += tweet.get('retweets_count', 0)
                aggregate[date_format][src][group]['likes'][topic] += tweet.get('likes_count', 0)
                aggregate[date_format][src][group]['replies'][topic] += tweet.get('replies_count', 0)

for src in COLS:
    for date_format in aggregate.keys():
        print(f'Rearranging counts into np arrays for {src} and {date_format}...')
        time_groups = sorted(aggregate[date_format][src].keys())
        distributions = {
            'raw': np.array([aggregate[date_format][src][group]['raw'].tolist() for group in time_groups]),
            'retweets': np.array([aggregate[date_format][src][group]['retweets'].tolist() for group in time_groups]),
            'likes': np.array([aggregate[date_format][src][group]['likes'].tolist() for group in time_groups]),
            'replies': np.array([aggregate[date_format][src][group]['replies'].tolist() for group in time_groups])
        }

        target_dir = os.path.join(FILE_TARGETS_FULL[src], date_format)
        os.makedirs(target_dir, exist_ok=True)
        for boost in [[], ['retweets'], ['replies'], ['likes'], ['retweets', 'likes'], ['replies', 'likes'],
                      ['retweets', 'replies'], ['retweets', 'likes', 'replies']]:
            boost_prefix = '_'.join(boost or ['raw'])

            distribution = distributions['raw']
            for b in boost:
                distribution += distributions[b]

            for norm in ['abs']:  # 'col', 'row',
                print(f'Computing temporal distribution for boost: "{boost_prefix}" and normalisation: "{norm}"')

                if norm == 'abs':
                    topic_dist = distribution.copy()
                elif norm == 'row':
                    topic_dist = distribution / (distribution.sum(axis=0) + EPS)
                else:  # norm == 'col'
                    topic_dist = (distribution.T / (distribution.sum(axis=1) + EPS)).T

                with open(f'{target_dir}/temporal_{date_format}_{boost_prefix}_{norm}.json', 'w') as f_out:
                    f_out.write(json.dumps({
                        'groups': time_groups,
                        'distribution': topic_dist.tolist()
                    }))

import json
import random
import numpy as np
import os

print('reading dump')
with open('data/geoengineering/topics_finn2/dump_full_finn2.json', 'r') as f:
    dump = json.load(f)

print('reading labels')
labels = np.load('data/geoengineering/topics_finn2.npy')

print('selecting tweets')
downsampled_tweets = []
for topic in np.unique(labels):
    tweet_idxs = np.argwhere(labels == topic).reshape(-1, )
    np.random.shuffle(tweet_idxs)

    for i in tweet_idxs[:500]:
        downsampled_tweets.append(dump['tweets'][i])

dump['tweets'] = downsampled_tweets

print('dumping smaller dump')
with open('data/geoengineering/topics_finn2/dump_geo_finn2.json', 'w') as f:
    json.dump(dump, f)

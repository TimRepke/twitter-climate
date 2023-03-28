import json
import random
import numpy as np
import os

print('reading dump')
with open('data/climate2/topics_big2/dump_7000000_monthly.json', 'r') as f:
    dump = json.load(f)

print('reading labels')
labels = np.load('data/climate2/topics_big2/labels_7000000_tsne.npy')

print('selecting tweets')
downsampled_tweets = []
for topic in np.unique(labels):
    tweet_idxs = np.argwhere(labels == topic).reshape(-1, )
    np.random.shuffle(tweet_idxs)

    for i in tweet_idxs[:100]:
        downsampled_tweets.append(dump['tweets'][i])

dump['tweets'] = downsampled_tweets

print('appending pre_post')
pre_post_file = 'data/climate2/topics_big2/pre_post_vectors_7000000.json'
if os.path.exists(pre_post_file):
    with open(pre_post_file, 'r') as f:
        pre_post = json.load(f)
dump['prePost'] = pre_post

print('dumping smaller dump')
with open('data/climate2/topics_big2/dump_7000000_monthly_downsampled.json', 'w') as f:
    json.dump(dump, f)

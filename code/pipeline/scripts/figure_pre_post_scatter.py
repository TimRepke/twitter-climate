import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

print('reading dump')
with open('data/climate2/topics_big2/dump_7000000_monthly.json', 'r') as f:
    dump = json.load(f)

g2idx = {g: idx for idx, g in enumerate(dump['groups'])}
n_topics = len(dump['topics'])

print('initialising vectors')
vectors = np.zeros((n_topics, len(dump['groups'])))
boosts = {
    'likes': np.zeros((n_topics, len(dump['groups']))),
    'replies': np.zeros((n_topics, len(dump['groups']))),
    'rt': np.zeros((n_topics, len(dump['groups'])))
}

print('constructing distributions')
for tweet in tqdm(dump['tweets']):
    vectors[tweet['topic']][g2idx[tweet['group']]] += 1
    boosts['rt'][tweet['topic']][g2idx[tweet['group']]] += tweet['retweets']
    boosts['likes'][tweet['topic']][g2idx[tweet['group']]] += tweet['likes']
    boosts['replies'][tweet['topic']][g2idx[tweet['group']]] += tweet['replies']

del dump['tweets']


def plot(boosters=None, log=False):
    vecs = vectors.copy()
    for boost in boosters or []:
        vecs += boosts[boost]

    topic_sizes = vecs.sum(axis=1)
    norm = 1 / topic_sizes.max()
    positions = np.vstack([vecs[:, :boundary].sum(axis=1), vecs[:, boundary:].sum(axis=1)]).T

    for i, (pos, size) in enumerate(zip(positions, topic_sizes)):
        plt.scatter(pos[0], pos[1], s=(size * norm) * 1000, label=f'Topic {i}', marker='.', alpha=1)
    plt.title(' + '.join(['Count'] + (boosters or [])))
    if log:
        plt.yscale('log')
        plt.xscale('log')
    # plt.show()

    return {
        'vectors': positions.tolist(),
        'max': topic_sizes.max()
    }


pre_post = {}

boundary = g2idx['2020-03']

plt.figure(figsize=(20, 30))
plt.subplot(321)
pre_post['count'] = plot(log=True)
plt.subplot(322)
pre_post['likes'] = plot(['likes'], log=True)
plt.subplot(323)
pre_post['retweets'] = plot(['rt'], log=True)
plt.subplot(324)
pre_post['replies'] = plot(['replies'], log=True)
plt.subplot(325)
pre_post['likes_retweets_replies'] = plot(['likes', 'rt', 'replies'], log=True)
plt.show()

print(json.dumps(pre_post))

plt.clf()

plt.figure(figsize=(20, 30))
plt.subplot(321)
plot(log=False)
plt.subplot(322)
plot(['likes'], log=False)
plt.subplot(323)
plot(['rt'], log=False)
plt.subplot(324)
plot(['replies'], log=False)
plt.subplot(325)
plot(['likes', 'rt', 'replies'], log=False)
plt.show()

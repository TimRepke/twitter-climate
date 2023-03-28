import numpy as np

labels = np.load('data/climate2/topics_big2/labels_7000000_tsne.npy')
topic_ids, topic_sizes = np.unique(labels, return_counts=True)
print(f'Num topics: {len(topic_ids)}, mean size: {topic_sizes[1:].mean():.1f}, '
      f'median size: {np.median(topic_sizes[1:]):.1f}, num outliers: {topic_sizes[0]:,},'
      f'largest cluster: {topic_sizes[1:].max():,}')

n_topics = len(topic_sizes)
n_tweets = topic_sizes.sum()
print(f'Total num tweets: {n_tweets:,}')
for percentile in [50, 60, 70, 80, 90]:
    num_topics_in = n_topics * (percentile / 100)
    percentile_cutoff = np.percentile(topic_sizes, percentile)
    num_tweets_in = topic_sizes[topic_sizes > percentile_cutoff].sum()
    print(f'Percentile {percentile} contains {num_topics_in:.0f} topics with '
          f'{num_tweets_in:,} ({num_tweets_in / n_tweets:.2%}) tweets '
          f'where each topic contains > {percentile_cutoff:.2f} tweets')

for cutoff in [200, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 400000]:
    n_topics_in = sum(topic_sizes > cutoff)
    n_tweets_in = topic_sizes[topic_sizes > cutoff].sum()
    print(f'There are {n_topics_in} topics containing > {cutoff:,} tweets, '
          f'which equals {n_tweets_in:,} ({n_tweets_in / n_tweets:.2%}) tweets')

import json

# DATASET = 'geoengineering'
DATASET = 'climate'

LIMIT = 10000
FILE = f'data/{DATASET}/tweets_sentiment_{LIMIT}.jsonl'

with open(FILE) as f:
    for line in f:
        tweet = json.loads(line)
        print('-----')
        print(tweet['text'])

        titles = list(tweet['sentiments'].keys())
        values = [f'{v[0][0]} ({v[0][1]:.2f})' for v in tweet['sentiments'].values()]
        lengths = [max(len(t), len(v)) for t, v in zip(titles, values)]

        print(' | '.join([t.ljust(l) for t, l in zip(titles, lengths)]))
        print(' | '.join([v.ljust(l) for v, l in zip(values, lengths)]))
        #
        # print('  ->', ', '.join(
        #     [f'{k}: {v[0][0]} ({v[0][1]:.2f})'
        #      for k, v in t['sentiments'].items()]))

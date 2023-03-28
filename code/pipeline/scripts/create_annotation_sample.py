import json
from collections import defaultdict
import random

tweets = defaultdict(list)
with open('data/climate2/tweets_classified_7000000_False.jsonl', 'r') as f:
    for l in f:
        t = json.loads(l)
        label = list(t['classes']['cards'].keys())[0][0]
        tweets[label].append(t)

sample_size = 20
subset = []
for label, tweets_ in tweets.items():
    lst = tweets_
    random.shuffle(lst)
    subset += lst[:sample_size]

random.shuffle(subset)
print(json.dumps(subset))



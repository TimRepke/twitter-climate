import json
from collections import Counter

with open('data/climate/tweets_clean.jsonl') as f:
    ht = [json.loads(l)['meta']['n_hashtags'] for l in f]

c = Counter(ht)
c.most_common()

print(f'total {len(ht):,}')
print(f'0 hashtags: {c[0]:,} ({c[0] / len(ht):.2%})')
print(f'1 hashtag: {c[1]:,} ({c[1] / len(ht):.2%})')
print(f'>1 hashtag: {len(ht) - c[1] - c[0]:,} ({(len(ht) - c[1] - c[0]) / len(ht):.2%})')

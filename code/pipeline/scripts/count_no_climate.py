import json
import numpy as np
from collections import Counter

print('reading dump')
with open('data/climate2/topics_big/dump_7000000_monthly.json', 'r') as f:
    dump = json.load(f)

print('iterating')
no_climate = [t['topic'] for t in dump['tweets'] if 'climate' not in t['text'].lower()]

cnts = Counter(no_climate).most_common()

for t, c in cnts:
    print(f'Topic {t} with {dump["topics"][t]["n_tweets"]} tweets has {c} tweets without "climate"')

print(f'Num tweets: {len(dump["tweets"]):,}, num tweets w/o "climate": '
      f'{len(no_climate):,} ({len(no_climate)/len(dump["tweets"]):.2%})')
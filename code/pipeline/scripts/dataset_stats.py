from tqdm import tqdm
from collections import defaultdict, Counter
import numpy as np
import json

SOURCE_FILES = [
    # '../../data/climate2/tweets_filtered_10000.jsonl',
    # '../../data/climate2/tweets_filtered_100000.jsonl',
    # '../../data/climate2/tweets_filtered_1000000.jsonl',
    '../../data/climate2/tweets_filtered.jsonl',
    # '../../data/climate2/tweets_clean.jsonl'
]

if __name__ == '__main__':
    for sf in SOURCE_FILES:
        print(sf)
        with open(sf) as f_in:
            authors = Counter([json.loads(l)['author_id'] for l in tqdm(f_in)])
            counts = np.array(list(authors.values()))

            print(f'Number of tweets total: {counts.sum():,}')
            print(f'Number of users: {len(authors):,}')
            print(f'Number of tweets per user (mean): {np.mean(counts):.2f}')
            print(f'Number of tweets per user (median): {np.median(counts):.2f}')
            print(f'Number of tweets per user (max): {np.max(counts):.2f}')
            print(f'Number of tweets per user (50th percentile): {np.percentile(counts, 50):.2f}')
            print(f'Number of tweets per user (75th percentile): {np.percentile(counts, 75):.2f}')
            print(f'Number of tweets per user (80th percentile): {np.percentile(counts, 80):.2f}')
            print(f'Number of tweets per user (90th percentile): {np.percentile(counts, 90):.2f}')
            print(f'Number of tweets per user (95th percentile): {np.percentile(counts, 95):.2f}')
            print(f'Number of tweets per user (98th percentile): {np.percentile(counts, 98):.2f}')
            print(f'Number of tweets per user (99th percentile): {np.percentile(counts, 99):.2f}')
        print('-------------')

# ../../data/climate2/tweets_filtered_10000.jsonl
# Number of tweets total: 10,002
# Number of users: 9,598
# Number of tweets per user (mean): 1.04
# Number of tweets per user (median): 1.00
# Number of tweets per user (max): 13.00
# Number of tweets per user (50th percentile): 1.00
# Number of tweets per user (75th percentile): 1.00
# Number of tweets per user (80th percentile): 1.00
# Number of tweets per user (90th percentile): 1.00
# Number of tweets per user (95th percentile): 1.00
# Number of tweets per user (98th percentile): 2.00
# Number of tweets per user (99th percentile): 2.00
# -------------
# ../../data/climate2/tweets_filtered_100000.jsonl
# Number of tweets total: 100,486
# Number of users: 84,388
# Number of tweets per user (mean): 1.19
# Number of tweets per user (median): 1.00
# Number of tweets per user (max): 125.00
# Number of tweets per user (50th percentile): 1.00
# Number of tweets per user (75th percentile): 1.00
# Number of tweets per user (80th percentile): 1.00
# Number of tweets per user (90th percentile): 2.00
# Number of tweets per user (95th percentile): 2.00
# Number of tweets per user (98th percentile): 3.00
# Number of tweets per user (99th percentile): 4.00
# -------------
# ../../data/climate2/tweets_filtered_1000000.jsonl
# Number of tweets total: 106,2276
# Number of users: 596,370
# Number of tweets per user (mean): 1.78
# Number of tweets per user (median): 1.00
# Number of tweets per user (max): 1,179.00
# Number of tweets per user (50th percentile): 1.00
# Number of tweets per user (75th percentile): 2.00
# Number of tweets per user (80th percentile): 2.00
# Number of tweets per user (90th percentile): 3.00
# Number of tweets per user (95th percentile): 4.00
# Number of tweets per user (98th percentile): 8.00
# Number of tweets per user (99th percentile): 12.00
# -------------
# ../../data/climate2/tweets_filtered.jsonl
# Number of tweets total: 14,871,854
# Number of users: 3,321,244
# Number of tweets per user (mean): 4.48
# Number of tweets per user (median): 1.00
# Number of tweets per user (max): 17,109.00
# Number of tweets per user (50th percentile): 1.00
# Number of tweets per user (75th percentile): 3.00
# Number of tweets per user (80th percentile): 4.00
# Number of tweets per user (90th percentile): 7.00
# Number of tweets per user (95th percentile): 14.00
# Number of tweets per user (98th percentile): 30.00
# Number of tweets per user (99th percentile): 50.00
# -------------
# ../../data/climate2/tweets_clean.jsonl
# Number of tweets total: 20,213,738
# Number of users: 3,879,955
# Number of tweets per user (mean): 5.21
# Number of tweets per user (median): 1.00
# Number of tweets per user (max): 55,885.00
# Number of tweets per user (50th percentile): 1.00
# Number of tweets per user (75th percentile): 3.00
# Number of tweets per user (80th percentile): 4.00
# Number of tweets per user (90th percentile): 8.00
# Number of tweets per user (95th percentile): 16.00
# Number of tweets per user (98th percentile): 34.00
# Number of tweets per user (99th percentile): 60.00

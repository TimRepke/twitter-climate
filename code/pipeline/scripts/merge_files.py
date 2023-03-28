import ujson as json
import os
from tqdm import tqdm
from collections import Counter
from utils.tweets import s2time


def iterate_tweets(folder, files: list[str] = None):
    if files is None:
        files = os.listdir(folder)
    for file_ in files:
        if file_.startswith('cc_') and file_.endswith('json'):
            with open(os.path.join(folder, file_)) as f:
                for line in f:
                    o = json.loads(line)
                    if 'created_at' in o:
                        yield file_, o


def format_tweet(o):
    o['favorites_count'] = o['public_metrics']['like_count']
    o['retweets_count'] = o['public_metrics']['retweet_count']
    o['replies_count'] = o['public_metrics']['reply_count']
    o['quotes_count'] = o['public_metrics']['quote_count']

    if 'entities' in o:
        o['urls'] = [e['expanded_url'] for e in o['entities'].get('urls', [])]
        del o['entities']
    if 'referenced_tweets' in o:
        del o['referenced_tweets']
    if 'in_reply_to_user_id' in o:
        del o['in_reply_to_user_id']
    del o['public_metrics']
    return o


def get_group(folder, group):
    cnt = 0
    for _, t in tqdm(iterate_tweets(folder, list(srt_groups[group]['files']))):
        if cnt >= srt_groups[group]['cnt']:
            break
        if t['created_at'][:7] == group:
            cnt += 1
            yield t


print('Reading in groups...')
srt_groups = dict()
for file, t in tqdm(iterate_tweets('data/climate2/raw')):
    g = t['created_at'][:7]
    if g not in srt_groups:
        srt_groups[g] = {'cnt': 0, 'files': set()}
    srt_groups[g]['cnt'] += 1
    srt_groups[g]['files'].add(file)
# srt_groups = Counter(t['created_at'][:7] for t in tqdm(iterate_tweets('data/climate2/raw')))

sort_group_keys = sorted(srt_groups.keys())
print(f'Sorting by date in {len(sort_group_keys)} groups from {sort_group_keys[0]} to {sort_group_keys[-1]}...')
with open('data/climate2/tweets_raw.jsonl', 'w') as f_out:
    for sort_group in sort_group_keys:
        print(f'Sorting tweets with partial date {sort_group}')
        tweets = [format_tweet(t) for t in get_group('data/climate2/raw', sort_group)]
        for tweet in sorted(tweets, key=lambda t: t['created_at']):
            f_out.write(json.dumps(tweet) + '\n')

import json
import requests
from tqdm import tqdm
import os

# DATASET = 'geoengineering'
DATASET = 'climate'

LIMIT = 10000
SOURCE_FILE = f'data/{DATASET}/tweets_filtered_{LIMIT}.jsonl'
TARGET_FILE = f'data/{DATASET}/urls_resolved_{LIMIT}.jsonl'

if os.path.exists(TARGET_FILE):
    print(f'The file {TARGET_FILE} already exists. If you are sure you want to proceed, delete it first.')
    exit(1)

with open(SOURCE_FILE) as f_in, open(TARGET_FILE, 'w') as f_out:
    for i, l in enumerate(tqdm(f_in)):
        tweet = json.loads(l)
        if tweet['meta']['n_urls'] > 0:
            for url in tweet['meta']['urls']:
                new_url = url
                error = None
                try:
                    response_header = requests.head(url, allow_redirects=True, timeout=2)
                    new_url = response_header.url
                except requests.exceptions.ConnectTimeout as e:
                    print('\n', url, e)
                    error = str(e)
                    new_url = e.request.url
                except requests.exceptions.ConnectionError as e:
                    print('\n', url, e)
                    error = str(e)
                    new_url = e.request.url
                except requests.exceptions.ReadTimeout as e:
                    print('\n', url, e)
                except requests.exceptions.InvalidURL as e:
                    print('\n', url, e)
                except Exception as e:
                    print('\n', url, e)

                if new_url != url:
                    f_out.write(json.dumps({
                        'tweet_id': tweet['twitterbasemodel_ptr_id'],
                        'created_at': tweet['created_at'],
                        'orig_url': url,
                        'real_url': new_url,
                        'error': error
                    }) + '\n')
                    f_out.flush()

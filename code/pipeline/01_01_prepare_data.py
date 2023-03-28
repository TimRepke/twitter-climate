import json
from typing import Optional
from tap import Tap
from utils.io import exit_if_exists, produce_batches
from utils.tweets import clean_tweet, get_hashtags, get_mentions, get_urls
from tqdm import tqdm


class DataPrepArgs(Tap):
    dataset: str = 'geoengineering'
    batch_size: int = 20000
    skip_lines: int = 0
    source: str
    target: str


def process_tweet(tweet):
    if tweet["text"]:
        tweet['clean_text'] = clean_tweet(tweet['text'],
                                          remove_hashtags=True,
                                          remove_urls=True,
                                          remove_mentions=True,
                                          remove_nonals=True)
        hashtags = get_hashtags(tweet['text'])
        urls = get_urls(tweet['text'])
        mentions = get_mentions(tweet['text'])
        n_tokens = len(tweet['clean_text'].split())
        tweet['meta'] = {
            'n_tokens': n_tokens,
            'n_tokens_raw': len(hashtags) + len(urls) + len(mentions) + n_tokens,
            'n_hashtags': len(hashtags),
            'hashtags': hashtags,
            'n_urls': len(urls),
            'urls': urls,
            'n_mentions': len(mentions),
            'mentions': mentions
        }
    return tweet


def prepare_dataset(dataset: str,
                    batch_size: int,
                    skip_first_n_lines: int = 0,
                    source_f: Optional[str] = None,
                    target_f: Optional[str] = None):
    if source_f is None:
        source_f = f'data/{dataset}/tweets_raw.jsonl'
    if target_f is None:
        target_f = f'data/{dataset}/tweets_clean.jsonl'

    exit_if_exists(target_f)

    with open(target_f, 'w') as f_out:
        for tweets_batch in tqdm(produce_batches(source_f, batch_size, skip_first_n_lines)):
            print('Processing batch...')
            tweets = [process_tweet(t) for t in tweets_batch]
            print('Writing batch...')
            [f_out.write(json.dumps(t) + '\n') for t in tweets]


if __name__ == '__main__':
    args = DataPrepArgs().parse_args()
    prepare_dataset(
        dataset=args.dataset,
        batch_size=args.batch_size,
        source_f=args.source,
        target_f=args.target,
        skip_first_n_lines=args.skip_lines
    )

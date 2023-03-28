import json
from tqdm import tqdm
from utils.tweets import s2time
from typing import Literal, Optional
import plotly.graph_objects as go
import os
from collections import defaultdict
from tap import Tap
from utils.tweets import TweetFilter


class FilterArgs(Tap):
    dataset: str = 'climate2'
    limit: int = 7000000
    only_en: bool = True
    from_date: Optional[str] = '2018-01'
    to_date: Optional[str] = '2022-01'
    allow_lang_null: bool = True
    has_climate: bool = True
    min_tokens: int = 4
    max_hashtags: int = 5
    allow_duplicates: bool = False
    duplicate_include_author: bool = False

    date_format: Literal['monthly', 'yearly', 'weekly', 'daily'] = 'monthly'

    @property
    def source_file(self) -> str:
        return f'data/{self.dataset}/tweets_clean.jsonl'

    @property
    def target_folder(self) -> str:
        return f'data/{self.dataset}/stats'

    @property
    def format(self):
        return {'yearly': '%Y', 'monthly': '%Y-%m', 'weekly': '%Y-%W', 'daily': '%Y-%m-%d'}[self.date_format]


if __name__ == '__main__':

    args = FilterArgs().parse_args()
    os.makedirs(args.target_folder, exist_ok=True)

    groups = defaultdict(lambda: {
        'total': 0,
        'duplicate': 0,
        'not_en': 0,
        'lang_null': 0,
        'leq_min_tokens': 0,
        'geq_max_hashtags': 0,
        'relevant': 0,
        'not_relevant': 0,
        'no_climate': 0
    })
    print('Processing...')
    with open(args.source_file, 'r') as f_in:

        tweet_filter = TweetFilter(only_en=args.only_en, allow_lang_null=args.allow_lang_null,
                                   min_tokens=args.min_tokens, allow_duplicates=args.allow_duplicates,
                                   max_hashtags=args.max_hashtags, from_date=args.from_date,
                                   to_date=args.to_date, duplicate_include_author=args.duplicate_include_author,
                                   has_climate=args.has_climate)

        for line_i, line in tqdm(enumerate(f_in)):
            tweet_o = json.loads(line)

            result = tweet_filter.is_relevant(tweet_o)

            grp = s2time(tweet_o['created_at']).strftime(args.format)

            groups[grp]['total'] += 1
            if result.accept_lang and result.has_min_tokens and result.has_max_hashtags and result.has_climate:
                if not result.duplicate:
                    # relevant and non-duplicate
                    groups[grp]['relevant'] += 1
                else:
                    groups[grp]['not_relevant'] += 1
                    groups[grp]['duplicate'] += 1
            else:
                lang = tweet_o.get('lang', None)
                groups[grp]['not_relevant'] += 1
                if not tweet_o.get('lang', None) == 'en':
                    groups[grp]['not_en'] += 1
                    if lang is None:
                        groups[grp]['lang_null'] += 1
                if not result.has_min_tokens:
                    groups[grp]['leq_min_tokens'] += 1
                if not result.has_max_hashtags:
                    groups[grp]['geq_max_hashtags'] += 1
                if not result.has_climate:
                    groups[grp]['no_climate'] += 1

        # clear up memory
        del tweet_filter

    srt_grps = list(sorted(groups.keys()))
    axis = [f'd:{k}' for k in srt_grps]
    for key in ['total', 'duplicate', 'not_en', 'lang_null', 'no_climate',
                'leq_min_tokens', 'geq_max_hashtags', 'relevant', 'not_relevant']:
        values = [groups[k][key] for k in srt_grps]

        fig = go.Figure([go.Bar(x=axis, y=values)])
        fig.write_html(f'{args.target_folder}/histogram_{args.date_format}_{key}.html')

import json
from tqdm import tqdm
import os
from urllib.parse import urlparse
from collections import Counter, defaultdict
from utils.tweets import s2time
import plotly.graph_objects as go

# DATASET = 'geoengineering'
DATASET = 'climate'

LIMIT = 10000
SOURCE_FILE_URLS = f'data/{DATASET}/urls_resolved_{LIMIT}.jsonl'
SOURCE_FILE_TWEETS = f'data/{DATASET}/tweets_sentiment_{LIMIT}.jsonl'

FORMATS = {'yearly': '%Y', 'monthly': '%Y-%m', 'weekly': '%Y-%W', 'daily': '%Y-%m-%d'}
FORMAT = 'monthly'
GROUPING = FORMATS[FORMAT]

TARGET_FOLDER = f'data/{DATASET}/websites'
os.makedirs(TARGET_FOLDER, exist_ok=True)

TARGET_POPULAR = f'{TARGET_FOLDER}/most_popular_{FORMAT}_{LIMIT}.csv'
TARGET_PLOT = f'{TARGET_FOLDER}/most_popular_{FORMAT}_{LIMIT}.html'

PLOT_EXCLUDE = ['twitter.com', 't.co']
PLOT_LIMIT = 30

# if os.path.exists(TARGET_FILE):
#     print(f'The file {TARGET_FILE} already exists. If you are sure you want to proceed, delete it first.')
#     exit(1)


def parse_line(line):
    info = json.loads(line)
    info['host'] = urlparse(info['real_url']).hostname
    info['group'] = s2time(info['created_at']).strftime(GROUPING)
    return info


if __name__ == '__main__':
    with open(SOURCE_FILE_URLS) as f_in:
        print(f'Reading data from {SOURCE_FILE_URLS}...')
        data = [parse_line(l) for l in f_in]

    print('Grouping data...')
    grouped_url = defaultdict(list)
    grouped_date = defaultdict(list)
    for d in data:
        grouped_url[d['host']].append(d['group'])
        grouped_date[d['group']].append(d['host'])

    date_groups = sorted(grouped_date.keys())
    url_timeseries = {}

    print(f'Writing results to {TARGET_POPULAR}')
    with open(TARGET_POPULAR, 'w') as f_out:
        f_out.write(f'HOST | TOTAL | FIRST | LAST | {" | ".join(date_groups)}\n')
        for host, cnt in Counter([d['host'] for d in data]).most_common():
            distribution = Counter(grouped_url[host])
            sorted_keys = sorted(distribution.keys())

            url_timeseries[host] = {
                'cnt': cnt,
                'distribution': [distribution.get(g, 0) for g in date_groups]
            }

            f_out.write(f'{host} | {cnt:,} | {sorted_keys[0]} | {sorted_keys[-1]} | '
                        f'{" | ".join([str(d) for d in url_timeseries[host]["distribution"]])} \n')

    plot_urls = [d for d in sorted(url_timeseries.items(), key=lambda u: u[1]['cnt'], reverse=True)
                 if d[0] not in PLOT_EXCLUDE][:PLOT_LIMIT]

    fig = go.Figure(data=go.Heatmap(
        z=list(reversed([d[1]['distribution'] for d in plot_urls])),
        x=[f'd:{d}' for d in date_groups],
        y=list(reversed([d[0] for d in plot_urls])),
        hoverongaps=False))
    fig.write_html(TARGET_PLOT)


    # {
    #     'tweet_id': tweet['twitterbasemodel_ptr_id'],
    #     'created_at': tweet['created_at'],
    #     'orig_url': url,
    #     'real_url': response_header.url
    # }

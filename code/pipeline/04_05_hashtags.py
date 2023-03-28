import datetime
import os
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import plotly.graph_objs as go
import plotly.express as px
from typing import Literal, Union
from utils.tweets import Tweets
from utils.hashtags import GroupedHashtags
from utils.tweets import s2time
import json
from collections import defaultdict
import scipy.spatial


def line2tweet(line):
    tweet = json.loads(line)
    return {
        'text': tweet['clean_text'],
        'created_at': tweet['created_at'],
        'group': s2time(tweet['created_at']).strftime(GROUPING),
        'hashtags': tweet['meta']['hashtags']
    }


# DATASET = 'geoengineering'
DATASET = 'climate2'
LIMIT = 1000000
SOURCE_FILE = f'data/{DATASET}/tweets_filtered_{LIMIT}.jsonl'

MIN_DF = 5
MAX_DF = .8
BINARY = False

VECTORISER = ['tfidf', 'count'][0]
MODE = ['hashtags', 'tokens', 'mixed'][0]

FORMATS = {'yearly': '%Y', 'monthly': '%Y-%m', 'weekly': '%Y-%W', 'daily': '%Y-%m-%d'}
FORMAT = 'weekly'
GROUPING = FORMATS[FORMAT]

TOKEN_PRINT_LIMIT = 20
TOKEN_PLOT_LIMIT = 6

SIMILARITY_METRIC = ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                     'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
                     'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis',
                     'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
                     'seuclidean', 'sokalmichener', 'sokalsneath',
                     'sqeuclidean', 'wminkowski', 'yule'][5]

TARGET_FOLDER = f'data/{DATASET}/similarities'
TARGET_BASE = f'{TARGET_FOLDER}/{MODE}_{LIMIT}_{VECTORISER}_{FORMAT}_{MIN_DF}_{str(MAX_DF).replace(".", "")}_{BINARY}'
os.makedirs(TARGET_FOLDER, exist_ok=True)

if __name__ == '__main__':
    print('Loading tweets...')
    with open(SOURCE_FILE) as f_in:
        tweets = [line2tweet(l) for l in f_in]
    print(f'Number of tweets: {len(tweets):,}')

    print('Grouping tweets...')
    grouped_tweets = defaultdict(list)
    [grouped_tweets[t['group']].append(t) for t in tweets]

    groups = sorted(grouped_tweets.keys())
    print(f'Number of groups: {len(groups)}')

    if MODE == 'hashtags':
        fake_docs = [' '.join([ht for t in grouped_tweets[g] for ht in t['hashtags']]) for g in groups]
    elif MODE == 'tokens':
        fake_docs = [' '.join([t['text'] for t in grouped_tweets[g]]) for g in groups]
    else:  # MODE == 'mixed'
        fake_docs = [' '.join([t['text'] for t in grouped_tweets[g]]) +
                     ' '.join([ht for t in grouped_tweets[g] for ht in t['hashtags']])
                     for g in groups]

    vectoriser = TfidfVectorizer if VECTORISER == 'tfidf' else CountVectorizer
    vectoriser = vectoriser(min_df=MIN_DF, max_df=MAX_DF, binary=BINARY, token_pattern=r'(?u)(?:\b|#)\w\w+\b')
    grouped_hashtags = GroupedHashtags(groups, fake_docs, vectoriser=vectoriser)

    print(f'Vocab size: {grouped_hashtags.vocab_size:,}')

    with open(f'{TARGET_BASE}_top_tokens.txt', 'w') as f_out:
        f_out.write(f'Vocab size: {grouped_hashtags.vocab_size}\n')
        f_out.write(f'Number of groups: {len(groups)}\n')
        relevant_tags = grouped_hashtags.most_common(top_n=TOKEN_PRINT_LIMIT, include_count=True,
                                                     include_hashtag=True, least_common=False)
        for g, popular in relevant_tags:
            f_out.write(f'=== {g} === \n')
            f_out.write('  > ' + ' '.join([f'{ht} ({cnt:.3f})' for ht, cnt in popular]) + '\n')

    group_similarities = grouped_hashtags.pairwise_similarities(metric=SIMILARITY_METRIC)
    mc = grouped_hashtags.most_common(top_n=TOKEN_PLOT_LIMIT, include_count=False,
                                      include_hashtag=True, least_common=False)

    # small hack (add prefix) so plotly doesn't try to interpret this as a date format
    xy_labels = [f'd:{g}' for g in groups]

    fig = go.Figure([go.Heatmap(z=group_similarities, x=xy_labels, y=xy_labels, hoverinfo='text',
                                text=[[
                                    f"""
                                   {gi}: {', '.join(li)} <br>
                                   {gj}: {', '.join(lj)}
                                    """
                                    for gi, li in mc]
                                    for gj, lj in mc]),
                     ],
                    layout=go.Layout(
                        width=600, height=600,
                        xaxis=dict(scaleanchor='y', constrain='domain', constraintoward='center'),
                        yaxis=dict(zeroline=False, autorange='reversed', constrain='domain')
                    ))

    fig.update_yaxes(autorange=True, tickformat=GROUPING)
    fig.update_xaxes(tickformat=GROUPING)

    events = [
        (datetime.date(year=2016, month=11, day=7), 'COP22 Marrakech'),  # COP 21 in Marrakech, Morocco, 7–18 Nov
        (datetime.date(year=2017, month=1, day=20), 'Donald Trump'),  # Trump inauguration as president
        (datetime.date(year=2017, month=8, day=17), 'Hurricane Harvey'),  # Greta Thunberg started her strike
        (datetime.date(year=2017, month=11, day=6), 'COP23 Bonn'),  # COP23 Bonn, GER, 	6- 17 November 2017
        (datetime.date(year=2018, month=8, day=20), 'Greta'),  # Greta Thunberg started her strike
        (datetime.date(year=2018, month=12, day=2), 'COP24 Katowice'),  # COP24 in Poland 2-15 Dec
        (datetime.date(year=2019, month=9, day=1), 'Australian Bushfires'),  # out-of-control fires sprung up
        # (datetime.date(year=2019, month=12, day=2), 'COP25 Madrid'),  # COP25 Madrid, Spain,  2-13 Dec 2019
        (datetime.date(year=2019, month=12, day=24), 'First Covid Case'),  # First patient in Wuhan Hospital
        (datetime.date(year=2020, month=3, day=11), 'Declared Pandemic'),  # WHO declared COVID-19 as pandemic
        (datetime.date(year=2020, month=12, day=27), 'Vaccine Rollout'),  # EU officially began vaccine rollout
        (datetime.date(year=2021, month=10, day=31), 'COP26 Glasgow'),  # COP26 Glasgow, Scotland, 31 Oct - 13 Nov
    ]
    events = [{'date': dt,
               'group': dt.strftime(GROUPING),
               'group_idx': groups.index(dt.strftime(GROUPING)),
               'text': txt}
              for dt, txt in events if dt.strftime(GROUPING) in groups]

    for event in events:
        fig.add_hline(y=event['group_idx'],
                      line_dash="dot",
                      annotation_text=event['text'],
                      annotation_position="bottom right",
                      annotation_font_size=10,
                      annotation_font_color="blue")

    fig.write_html(f'{TARGET_BASE}_similarities.html')

    # top_tags = grouped_hashtags.most_common(top_n=1, include_count=False, include_hashtag=True,
    #                                         least_common=least_significant,
    #                                         from_vectoriser=from_vectoriser)
    # st.header('Hashtag Frequencies')
    # st.caption('Selection based on top tag per group.')
    # top_tags = [tt[0] for g, tt in top_tags]
    # tag_freqs = grouped_hashtags.get_frequencies(top_tags)
    # # tag_freqs = {tag: [grp.get(tag, 0) for grp in grouped_hashtags.groups.values()] for tag in top_tags}
    # tag_freqs['group'] = xy_labels
    # tag_freq = pd.DataFrame(tag_freqs)
    # fig = px.line(tag_freq, x='group', y=top_tags)
    # st.plotly_chart(fig)

    # st.header('Tweet Histogram')
    # res = tweets.histogram(grouping)
    # mx = max([r['freq'] for r in res])
    # fig = px.bar(res, x='grp', y='freq', range_y=(0, mx + (mx * 0.02)))
    # st.plotly_chart(fig)

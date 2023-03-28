import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from colorcet import glasbey
from collections import Counter
from datetime import datetime
from typing import Literal
import json
import os

from utils.tweets import clean_tweet
from utils import load_embedded_data_jsonl
from utils.topics.frankentopic import FrankenTopic, UMAPArgs, VectorizerArgs, TSNEArgs, KMeansArgs, HDBSCANArgs
from utils.topics.utils import FrankenTopicUtils, date2group
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# DATASET = 'geoengineering'
DATASET = 'climate2'
LIMIT = 100000

EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
# EMBEDDING_MODEL = 'vinai/bertweet-large'
EMB_INCLUDE_HASHTAGS = True

# SOURCE_FILE = f'data/{DATASET}/tweets_sentiment_{LIMIT}.jsonl'
SOURCE_FILE = f'data/{DATASET}/tweets_filtered_{LIMIT}.jsonl'
EMBEDDING_FILE = f'data/{DATASET}/tweets_embeddings_{LIMIT}_{EMB_INCLUDE_HASHTAGS}_' \
                 f'{EMBEDDING_MODEL.replace("/", "_")}.npy'

TARGET_DIR = f'data/{DATASET}/topics'
os.makedirs(TARGET_DIR, exist_ok=True)

DATE_FORMAT: Literal['monthly', 'yearly', 'weekly', 'daily'] = 'monthly'

if __name__ == '__main__':
    print('Loading tweets...')
    with open(SOURCE_FILE) as f_in:
        tweets = [json.loads(l) for l in f_in]

    print('Loading embeddings...')
    embeddings = np.load(EMBEDDING_FILE)

    LAYOUT_ARGS = [
        UMAPArgs(
            spread=1.,
            min_dist=0.05,
            n_neighbors=50,
            densmap=False,
            set_op_mix_ratio=0.5
        ),
        TSNEArgs(
            perplexity=40,
            early_exaggeration=20,
            metric='cosine',
            dof=0.6,
            initialization='pca'
        )][1]

    CLUSTERING_ARGS = [
        KMeansArgs(max_n_topics=40,
                   min_docs_per_topic=int((len(tweets) / 40) / 3)),
        HDBSCANArgs(
            min_samples=10,
            min_cluster_size=30,
            cluster_selection_epsilon=0.5,
            alpha=1.,
            cluster_selection_method='eom'
            # 10k params:
            # min_samples=5,
            # min_cluster_size=15,
            # cluster_selection_epsilon=1,
            # alpha=1.,
            # cluster_selection_method='eom'
        )
    ][1]
    print('Setting up FrankenTopic...')
    stop_words = list(ENGLISH_STOP_WORDS) + ['URL', 'MENTION', 'HASHTAG'] + ['url', 'mention', 'hashtag'] + ['rt', 'RT']
    topic_model = FrankenTopic(
        cluster_args=CLUSTERING_ARGS,
        n_words_per_topic=15,
        n_candidates=100,
        mmr_diversity=0.2,
        vectorizer_args=VectorizerArgs(max_df=.7, stop_words=stop_words, ngram_range=(1, 2)),
        dr_args=LAYOUT_ARGS,
        emb_backend=SentenceTransformerBackend,
        emb_model='paraphrase-multilingual-MiniLM-L12-v2',
        cache_layout=f'{TARGET_DIR}/layout_{LIMIT}_{LAYOUT_ARGS.__class__.__name__}.npy'
    )
    print('Fitting TopicModel...')
    topic_model.fit([clean_tweet(t['text']) for t in tweets], embeddings)

    print('Setting up utils...')
    topic_model_utils = FrankenTopicUtils(tweets=tweets,
                                          topic_model=topic_model,
                                          n_tokens_per_topic=20)

    print('Writing topic words to console...')
    topic_model_utils.list_topics(emotions_model=None,  # 'bertweet-sentiment',
                                  emotions_keys=['negative', 'neutral', 'positive'],
                                  include_mmr=False)

    print('Creating landscape figure...')
    fig = topic_model_utils.landscape(
        # emotions_model='bertweet-sentiment',
        # emotions_keys=['negative', 'positive'],
        # emotions_colours=['Reds', 'Greens'],
        keyword_source='tfidf',
        n_keywords_per_topic_legend=6,
        n_keywords_per_topic_map=4,
        include_text=False,
        colormap=glasbey*2
    )
    fig.write_html(f'{TARGET_DIR}/landscape_{LIMIT}.html')
    # fig.write_image('data/plt_emotions_static.png')

    # print('Creating stacked temporal topic figure...')
    # fig = topic_model_utils.temporal_stacked_fig(date_format=DATE_FORMAT,
    #                                              n_keywords_per_topic=5,
    #                                              keyword_source='tfidf',
    #                                              colorscheme=glasbey)
    # fig.write_html(f'{TARGET_DIR}/temporal_topics_stacked_{LIMIT}_{DATE_FORMAT}.html')

    print('Prepping dump...')
    tpc_cnts = np.unique(topic_model.labels, return_counts=True)
    output = {
        'tweets': [
            {
                'time': tw['created_at'],
                'id': tw['id'],
                'group': date2group(tw['created_at'], DATE_FORMAT),
                'topic': to,
                'text': tw['text'],
                'retweets': tw.get('retweets_count', 0),
                'likes': tw.get('favorites_count', 0),
                'replies': tw.get('replies_count', 0),
            } for tw, to in zip(tweets, topic_model.labels.tolist())
        ],
        'topics': [
            {
                'tfidf': ', '.join(topic_model_utils.get_keywords(ti, keyword_source='tfidf', n_keywords=15)),
                'mmr': ', '.join(topic_model_utils.get_keywords(ti, keyword_source='mmr', n_keywords=15)),
                'n_tweets': cnt
            }
            for ti, cnt in zip(tpc_cnts[0].tolist(), tpc_cnts[1].tolist())
        ]
    }

    os.makedirs(f'{TARGET_DIR}/temporal/', exist_ok=True)
    topic_dists = {}
    for boost in [[], ['retweets'], ['replies'], ['likes'], ['retweets', 'likes'], ['retweets', 'likes', 'replies']]:
        for norm in ['abs', 'row', 'col']:
            print(f'Computing temporal distribution for boost: "{"_".join(boost)}" and normalisation: "{norm}"')
            temporal_topics = topic_model_utils.get_temporal_distribution(date_format=DATE_FORMAT,
                                                                          boost=boost, skip_topic_zero=True)
            time_groups = sorted(temporal_topics.keys())

            topic_dist = np.array([temporal_topics[tg].tolist() for tg in time_groups])

            fig = go.Figure([go.Bar(x=[f'd:{d}' for d in time_groups],
                                    y=topic_dist.sum(axis=0))])
            fig.write_html(f'{TARGET_DIR}/histogram_{LIMIT}_{DATE_FORMAT}_{"_".join(boost)}.html')

            if norm == 'row':
                topic_dist = topic_dist / (topic_dist.sum(axis=0) + 0.0000001)
            elif norm == 'col':
                topic_dist = (topic_dist.T / (topic_dist.sum(axis=1) + 0.0000001)).T
            # elif norm == 'abs':
            #     pass

            fig = go.Figure(data=go.Heatmap(
                z=topic_dist.T,
                x=[f'd:{d}' for d in time_groups],
                y=[', '.join(topic_model_utils.get_keywords(t, keyword_source='mmr', n_keywords=4))
                   for t in topic_model_utils.topic_ids[1:]],
                hoverongaps=False))
            fig.write_html(f'{TARGET_DIR}/temporal_{LIMIT}_{DATE_FORMAT}_{norm}_{"_".join(boost)}.html')

            with open(f'{TARGET_DIR}/temporal/tt_{LIMIT}_{DATE_FORMAT}_{norm}_{"_".join(boost)}.json', 'w') as f:
                f.write(json.dumps({
                    'z': topic_dist.T.tolist(),
                    'x': time_groups,
                    'y': [', '.join(topic_model_utils.get_keywords(t, keyword_source='mmr', n_keywords=4))
                          for t in topic_model_utils.topic_ids[1:]],
                }))

            for ti, vals in enumerate(topic_dist.T):
                output['topics'][ti][f'{norm}_{"_".join(boost or ["raw"])}'] = vals.tolist()

    print('Writing dump...')
    output['groups'] = time_groups
    with open(f'{TARGET_DIR}/dump_{LIMIT}_{DATE_FORMAT}.json', 'w') as f:
        f.write(json.dumps(output))

    print('All done!')

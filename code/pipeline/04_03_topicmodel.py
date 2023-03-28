from typing import Optional, Literal, List
import os
from utils.cluster import ClusterJobBaseArguments
from copy import deepcopy


class TopicModelArgs(ClusterJobBaseArguments):
    model: Literal['minilm', 'bertopic'] = 'minilm'  # The embedding model to use.
    model_cache: str = 'data/models/'  # Location for cached models.

    file_tweets: Optional[str] = None  # The file containing tweets (relative to source root)
    file_layout: Optional[str] = None  # The file containing layout (relative to source root)
    output_directory: Optional[str] = None  # The directory to write outputs to
    file_labels: Optional[str] = None  # The npy file containing existing topic labels
    ignore_file_labels: bool = False  # Set this flag to override (possibly) existing topic labels

    limit: Optional[int] = 10000  # Size of the dataset
    dataset: Optional[str] = 'climate2'  # Name of the dataset
    excl_hashtags: bool = False  # Set this flag to exclude hashtags in the embedding
    projection: Literal['umap', 'tsne'] = 'tsne'  # The dimensionality reduction method to use

    # hdbscan args
    min_samples: int = 10
    min_cluster_size: int = 30
    cluster_selection_epsilon: float = 0.5
    alpha: float = 1.
    cluster_selection_method: Literal['eom', 'leaf'] = 'eom'

    n_tokens_listing: int = 20  # number of tokens per topic to show in listing and store in dump
    n_tokens_candidates: int = 150  # number of tokens to internally represent topics
    n_tokens_landscape: int = None  # number of tokens per topic in landscape rendering
    n_tokens_plot: int = 5
    mmr_diversity: float = 0.3  # diversity for MMR resampling
    temporal_grouping: Literal['monthly', 'yearly', 'weekly', 'daily'] = 'monthly'  # grouping for temporal topic dist

    cluster_jobname: str = 'twitter-topics'
    cluster_workdir: str = 'twitter'


def get_cluster_labels(layout_, min_samples, min_cluster_size,
                       cluster_selection_epsilon, alpha, cluster_selection_method):
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_samples=min_samples,
                                min_cluster_size=min_cluster_size,
                                cluster_selection_epsilon=cluster_selection_epsilon,
                                alpha=alpha,
                                cluster_selection_method=cluster_selection_method)
    clusterer.fit(layout_)
    labels_ = clusterer.labels_ + 1  # increment by one, so -1 (outlier) cluster becomes 0
    return labels_


def write_distributions(boosts: List[List[Literal['retweets', 'likes', 'replies']]],
                        norms: List[Literal['abs', 'row', 'col']]):
    ret = {}
    for boost in boosts:
        print(f'Computing temporal distribution for boost: "{"_".join(boost or ["raw"])}"')
        filename_part = f'{args.limit}_{args.temporal_grouping}_{"_".join(boost or ["raw"])}'

        temporal_topics = get_temporal_distribution(labels=labels, tweets=tweets,
                                                    date_format=args.temporal_grouping,
                                                    boost=boost, skip_topic_zero=True)
        time_groups = sorted(temporal_topics.keys())

        topic_dist = np.array([temporal_topics[tg].tolist() for tg in time_groups])

        fig = go.Figure([go.Bar(x=[f'd:{d}' for d in time_groups],
                                y=topic_dist.sum(axis=0))])
        fig.write_html(f'{target_dir}/histogram_{filename_part}.html')

        for norm in norms:
            print(f'Computing temporal distribution for boost: "{"_".join(boost)}" and normalisation: "{norm}"')

            if norm == 'row':
                topic_dist_ = topic_dist / (topic_dist.sum(axis=0) + 0.0000001)
            elif norm == 'col':
                topic_dist_ = (topic_dist.T / (topic_dist.sum(axis=1) + 0.0000001)).T
            else:  # norm == 'abs':
                topic_dist_ = topic_dist.copy()

            y = [', '.join(get_tokens(topics_mmr, topic=t, n_tokens=4)) for t in topic_ids[1:]]
            fig = go.Figure(data=go.Heatmap(
                z=topic_dist_.T,
                x=[f'd:{d}' for d in time_groups],
                y=y,
                hoverongaps=False))
            fig.write_html(f'{target_dir}/temporal_{filename_part}_{norm}.html')

            with open(f'{target_dir}/temporal_{filename_part}_{norm}.json', 'w') as f_dist:
                f_dist.write(json.dumps({
                    'z': topic_dist_.T.tolist(),
                    'x': time_groups,
                    'y': y,
                }))

            for ti, vals in enumerate(topic_dist.T):
                if ti not in ret:
                    ret[ti] = {}
                ret[ti][f'{norm}_{"_".join(boost or ["raw"])}'] = vals.tolist()
    return ret, time_groups


def get_dump():
    tpc_cnts = np.unique(labels, return_counts=True)
    output = {
        'groups': time_groups_,
        'tweets': [
            {
                'time': tw['created_at'],
                'id': tw['id'],
                'group': date2group(tw['created_at'], args.temporal_grouping),
                'topic': to,
                'text': tw['text'],
                'retweets': tw.get('retweets_count', 0),
                'likes': tw.get('favorites_count', 0),
                'replies': tw.get('replies_count', 0),
            } for tw, to in zip(tweets, labels.tolist())
        ],
        'topics': [
            {
                'tfidf': ', '.join(get_tokens(topics_tfidf, topic=ti, n_tokens=args.n_tokens_listing)),
                'mmr': ', '.join(get_tokens(topics_mmr, topic=ti, n_tokens=args.n_tokens_listing)),
                'n_tweets': cnt,
                **distributions[ti]
            }
            for ti, cnt in zip(tpc_cnts[0].tolist(), tpc_cnts[1].tolist())
        ]
    }
    return output


if __name__ == '__main__':
    args = TopicModelArgs(underscores_to_dashes=True).parse_args()
    if args.args_file is not None:
        print(f'Dropping keyword arguments and loading from file: {args.args_file}')
        args = TopicModelArgs().load(args.args_file)

    _include_hashtags = not args.excl_hashtags

    file_layout = args.file_layout or f'data/{args.dataset}/topics/layout_{args.limit}_{args.projection}.npy'
    file_tweets = args.file_tweets or f'data/{args.dataset}/tweets_filtered_{args.limit}.jsonl'
    target_dir = args.output_directory or f'data/{args.dataset}/topics/'
    file_labels = args.file_labels or os.path.join(target_dir, f'labels_{args.limit}_{args.projection}.npy')
    file_dump = os.path.join(target_dir, f'dump_{args.limit}_{args.temporal_grouping}.json')

    if args.mode == 'cluster':
        from utils.cluster import Config as SlurmConfig
        from utils.cluster.job import ClusterJob
        from utils.cluster.files import FileHandler

        s_config = SlurmConfig.from_args(args)
        file_handler = FileHandler(config=s_config,
                                   local_basepath=os.getcwd(),
                                   requirements_txt='requirements_cluster.txt',
                                   include_dirs=['pipeline', 'utils'],
                                   model_cache=args.model_cache,
                                   required_models=[args.model])
        s_job = ClusterJob(config=s_config, file_handler=file_handler)

        cluster_args = deepcopy(args)
        cluster_args.mode = 'local'
        cluster_args.file_layout = os.path.join(s_config.datadir_path, file_layout)
        cluster_args.file_tweets = os.path.join(s_config.datadir_path, file_tweets)
        cluster_args.model_cache = s_config.modeldir_path
        cluster_args.output_directory = os.path.join(s_config.datadir_path, target_dir)
        cluster_args.file_labels = os.path.join(s_config.datadir_path, file_labels)
        s_job.submit_job(main_script='pipeline/04_03_topicmodel.py', params=cluster_args)

    else:
        import json
        import numpy as np
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        from sklearn.feature_extraction.text import TfidfVectorizer
        from utils.topics.frankentopic import get_top_mmr, get_top_tfidf
        from utils.topics.utils import simple_topic_listing, get_temporal_distribution, get_tokens, date2group
        import plotly.graph_objects as go
        import os

        os.makedirs(target_dir, exist_ok=True)
        from colorcet import glasbey

        print('Loading layout...')
        layout = np.load(file_layout)

        if os.path.exists(file_labels) and not args.ignore_file_labels:
            print(f'Loading existing cluster labels from file {file_labels}')
            labels = np.load(file_labels)
        else:
            print('Clustering with HDBSCAN...')
            labels = get_cluster_labels(layout, min_samples=args.min_samples, min_cluster_size=args.min_cluster_size,
                                        cluster_selection_epsilon=args.cluster_selection_epsilon, alpha=args.alpha,
                                        cluster_selection_method=args.cluster_selection_method)
            print(f'Saving cluster assignments to {file_labels}')
            np.save(file_labels, labels)

        topic_ids, topic_sizes = np.unique(labels, return_counts=True)
        print(f'Num topics: {len(topic_ids)}, mean size: {topic_sizes[1:].mean():.1f}, '
              f'median size: {np.median(topic_sizes[1:]):.1f}, num outliers: {topic_sizes[0]:,},'
              f'largest cluster: {topic_sizes[1:].max():,}')

        print('Loading tweets...')
        with open(file_tweets) as f_in:
            tweets = [json.loads(line) for line in f_in]

        print('Grouping tweets...')
        grouped_texts = [
            [tweets[i]['clean_text'] for i in np.argwhere(labels == label).reshape(-1, )]
            for label in np.unique(labels)
        ]

        print('Vectorising groups...')
        stop_words = list(ENGLISH_STOP_WORDS) + ['url', 'mention', 'hashtag', 'rt']
        vectorizer = TfidfVectorizer(max_df=0.7, stop_words=stop_words, ngram_range=(1, 2),
                                     min_df=0, lowercase=True, use_idf=True, smooth_idf=True)
        tf_idf_vecs = vectorizer.fit_transform([' '.join(g) for g in grouped_texts])
        vocab = {v: k for k, v in vectorizer.vocabulary_.items()}

        # compute topic representations
        topics_tfidf = get_top_tfidf(vectors=tf_idf_vecs, token_lookup=vocab, n_tokens=args.n_tokens_candidates)
        topics_mmr = get_top_mmr(topics_tfidf, n_tokens=args.n_tokens_listing, mmr_diversity=args.mmr_diversity,
                                 model_cache_location=args.model_cache, model=args.model)

        # print topics to console
        simple_topic_listing(topics_tfidf=topics_tfidf, topics_mmr=topics_mmr,
                             n_tokens=args.n_tokens_listing, labels=labels)

        distributions, time_groups_ = write_distributions(boosts=[[],
                                                                  ['retweets'],
                                                                  ['replies'],
                                                                  ['likes'],
                                                                  ['retweets', 'likes'],
                                                                  ['retweets', 'likes', 'replies']],
                                                          norms=['abs', 'row', 'col'])

        print('Preparing dump...')
        dump = get_dump()
        print('Writing dump...')
        with open(file_dump, 'w') as f:
            f.write(json.dumps(dump))

        print(':-) All done (-:')

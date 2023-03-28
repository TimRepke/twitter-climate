import json
import os
from typing import Optional, Union, Literal
import numpy as np
from utils.models import ModelCache, SentenceTransformerBackend, AutoModelBackend
from utils.cluster import ClusterJobBaseArguments
from utils.io import exit_if_exists
from utils.tweets import line2txt_hashtags, line2txt_clean
from copy import deepcopy

PathLike = Union[str, bytes, os.PathLike]


class TweetExtendEmbeddingArgs(ClusterJobBaseArguments):
    model: Literal['minilm', 'bertopic'] = 'minilm'  # The embedding model to use.
    model_cache: str = 'data/models/'  # Location for cached models.

    file_sampled: Optional[str] = None  # The file containing all already processed (down-sampled) tweets
    file_full: Optional[str] = None  # The file containing all relevant tweets
    file_out: Optional[str] = None  # The file to write embeddings to (relative to source root)

    dataset: Optional[str] = 'climate2'  # Name of the dataset
    excl_hashtags: bool = False  # Set this flag to exclude hashtags in the embedding

    cluster_jobname: str = 'twitter-expand-embed'
    cluster_workdir: str = 'twitter'

    cluster_gpu: bool = True


def read_known_tweet_ids(source_sampled: PathLike):
    with open(source_sampled, 'r') as f_sampled:
        return set([json.loads(line)['id'] for line in f_sampled])


def produce_tweets(source_full: PathLike, known_ids: set, include_hashtags: bool = True):
    if include_hashtags:
        def txt(t: dict):
            return t['clean_text'].replace('MENTION', '').replace('URL', '').replace('HASHTAG', '') + \
                   (' '.join(t['meta']['hashtags']))
    else:
        def txt(t: dict):
            return t['clean_text'].replace('MENTION', '').replace('URL', '').replace('HASHTAG', '')

    with open(source_full, 'r') as f_full:
        for line in f_full:
            tweet = json.loads(line)
            if tweet['id'] not in known_ids:
                yield int(tweet['id']), txt(tweet)


def embed_texts(texts: list[str],
                model: Union[AutoModelBackend, SentenceTransformerBackend],
                verbose: bool = True):
    return model.embed_documents(texts, verbose=verbose)


def embed_remaining_tweets(file_sampled: PathLike,
                           file_full: PathLike,
                           file_target: PathLike,
                           model: Union[AutoModelBackend, SentenceTransformerBackend],
                           include_hashtags: bool = True,
                           verbose: bool = True):
    exit_if_exists(file_target)

    print('Reading the IDs of previously down-sampled and processed tweets...')
    known_tweet_ids = read_known_tweet_ids(file_sampled)
    print(f'Found {len(known_tweet_ids)} tweets.')

    print('Reading the full dataset, skipping all the tweets that are already known...')
    tweets = list(produce_tweets(file_full, known_ids=known_tweet_ids, include_hashtags=include_hashtags))
    print(f'Found {len(tweets)} tweets that need to be appended.')

    print('Embedding texts...')
    texts = [t[1] for t in tweets]
    embeddings = embed_texts(texts, model=model, verbose=verbose)

    print('Storing results...')
    with open(file_target, 'wb') as f_out:
        np.save(f_out, np.array([t[0] for t in tweets]))
        np.save(f_out, embeddings)


if __name__ == '__main__':
    args = TweetExtendEmbeddingArgs(underscores_to_dashes=True).parse_args()
    if args.args_file is not None:
        print(f'Dropping keyword arguments and loading from file: {args.args_file}')
        args = TweetExtendEmbeddingArgs().load(args.args_file)

    _include_hashtags = not args.excl_hashtags

    if args.mode == 'cluster':
        from utils.cluster import Config as SlurmConfig
        from utils.cluster.job import ClusterJob
        from utils.cluster.files import FileHandler

        s_config = SlurmConfig.from_args(args,
                                         env_vars_run={
                                             'OPENBLAS_NUM_THREADS': 1,
                                             'TRANSFORMERS_OFFLINE': 1
                                         })
        file_handler = FileHandler(config=s_config,
                                   local_basepath=os.getcwd(),
                                   requirements_txt='requirements_cluster.txt',
                                   include_dirs=['pipeline', 'utils'],
                                   model_cache=args.model_cache,
                                   required_models=[args.model])
        s_job = ClusterJob(config=s_config, file_handler=file_handler)
        cluster_args = deepcopy(args)
        cluster_args.file_sampled = os.path.join(s_config.datadir_path, f'data/{args.dataset}/{args.file_sampled}')
        cluster_args.file_full = os.path.join(s_config.datadir_path, f'data/{args.dataset}/{args.file_full}')
        cluster_args.file_out = os.path.join(s_config.datadir_path, f'data/{args.dataset}/{args.file_out}')
        cluster_args.model_cache = s_config.modeldir_path
        s_job.submit_job(main_script='pipeline/04_04_01_embed_remaining_tweets.py', params=cluster_args)
    else:
        _model_cache = ModelCache(args.model_cache)
        _model = _model_cache.get_embedder(args.model)

        embed_remaining_tweets(
            file_sampled=args.file_sampled,
            file_full=args.file_full,
            file_target=args.file_out,
            model=_model,
            include_hashtags=_include_hashtags,
            verbose=True
        )

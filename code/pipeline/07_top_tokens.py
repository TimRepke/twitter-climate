import json
import os
from typing import Optional, Union, Literal
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from utils.cluster import ClusterJobBaseArguments
from utils.io import exit_if_exists, batched_lines
from copy import deepcopy

PathLike = Union[str, bytes, os.PathLike]


class TopTokenArgs(ClusterJobBaseArguments):
    file_in: Optional[str] = None  # The file to read data from (relative to source root)
    file_out: Optional[str] = None  # The file to write embeddings to (relative to source root)

    limit: Optional[int] = 10000  # Size of the dataset
    dataset: Optional[str] = 'climate2'  # Name of the dataset
    excl_hashtags: bool = False  # Set this flag to exclude hashtags in the embedding

    cluster_jobname: str = 'twitter-toptokens'
    cluster_workdir: str = 'twitter'

    skiplines: int = 100  # this will read every Nth line to construct vocab
    vocab_ngram_min: int = 1
    vocab_ngram_max: int = 3
    vocab_min_df: float = 1
    vocab_max_df: float = 0.8

    cluster_gpu: bool = True


def clean_clean_text(txt):
    return txt.replace('MENTION', '').replace('URL', '').replace('HASHTAG', '')


def get_vectoriser(texts, ngram_min: int = 1, ngram_max: int = 1,
                   min_df: Union[int, float] = 1, max_df: float = 1.) -> tuple[CountVectorizer, np.ndarray]:
    vectoriser = CountVectorizer(ngram_range=(ngram_min, ngram_max), stop_words='english',
                                 min_df=min_df, max_df=max_df)
    vectors = vectoriser.fit_transform(texts)
    return vectoriser, vectors


def read_data(tweets_file: PathLike, skiplines: int = 1, include_hashtags: bool = True):
    with open(tweets_file, 'r') as f:
        def func(line):
            tweet = json.loads(line)
            txt = clean_clean_text(tweet['clean_text'])
            if include_hashtags:
                txt += (' '.join(tweet['meta']['hashtags']))
            return txt, tweet['created_at'][:7]

        return [func(line) for i, line in enumerate(f) if (i % skiplines) == 0]


def compute_pre_post_vectors(data: list[tuple[str, str]], vectoriser: CountVectorizer,
                             vectors: np.ndarray, target_f: PathLike):
    exit_if_exists(target_f)

    pre_filter = np.array([d[1] < '2020-02' for d in data])
    post_filter = ~pre_filter

    pre_scores = vectors[pre_filter].sum(axis=1)
    post_scores = vectors[pre_filter].sum(axis=1)

    print('Storing results...')
    # np.save(target_f, embeddings)


if __name__ == '__main__':
    args = TopTokenArgs(underscores_to_dashes=True).parse_args()
    if args.args_file is not None:
        print(f'Dropping keyword arguments and loading from file: {args.args_file}')
        args = TopTokenArgs().load(args.args_file)

    _include_hashtags = not args.excl_hashtags

    if args.file_in is None:
        file_in = f'data/{args.dataset}/tweets_filtered_{args.limit}.jsonl'
    else:
        file_in = args.file_in
    # if args.file_out is None:
    #     file_out = f'data/{args.dataset}/tweets_embeddings_{args.limit}_{_include_hashtags}_{args.model}.npy'
    # else:
    #     file_out = args.file_out

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
                                   required_models=[args.model],
                                   data_files=[file_in])
        s_job = ClusterJob(config=s_config, file_handler=file_handler)
        cluster_args = deepcopy(args)
        cluster_args.file_in = os.path.join(s_config.datadir_path, file_in)
        cluster_args.file_out = os.path.join(s_config.datadir_path, file_out)
        cluster_args.model_cache = s_config.modeldir_path
        s_job.submit_job(main_script='pipeline/03_01_embed_data.py', params=cluster_args)
    else:
        data_ = read_data(file_in, skiplines=args.skiplines, include_hashtags=_include_hashtags)
        vectoriser_, vectors_ = get_vectoriser(texts=[d[0] for d in data_],
                                               ngram_min=args.vocab_ngram_min, ngram_max=args.vocab_ngram_max,
                                               min_df=args.vocab_min_df, max_df=args.vocab_max_df)

        pp = compute_pre_post_vectors(data_, vectoriser_, vectors_, '')
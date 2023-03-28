import json
import os
from typing import Optional, Union, Literal
import numpy as np
from utils.models import ModelCache, SentenceTransformerBackend, AutoModelBackend
from utils.cluster import ClusterJobBaseArguments
from utils.io import exit_if_exists
from copy import deepcopy

PathLike = Union[str, bytes, os.PathLike]


class TweetEmbeddingArgs(ClusterJobBaseArguments):
    model: Literal['minilm', 'bertweet'] = 'minilm'  # The embedding model to use.
    model_cache: str = 'data/models/'  # Location for cached models.
    file_in: Optional[str] = None  # The file to read data from (relative to source root)
    file_out: Optional[str] = None  # The file to write embeddings to (relative to source root)

    limit: Optional[int] = 10000  # Size of the dataset
    dataset: Optional[str] = 'climate2'  # Name of the dataset
    excl_hashtags: bool = False  # Set this flag to exclude hashtags in the embedding

    cluster_jobname: str = 'twitter-embed'
    cluster_workdir: str = 'twitter'

    cluster_gpu: bool = True


def clean_clean_text(txt):
    return txt.replace('MENTION', '').replace('URL', '').replace('HASHTAG', '')


def line2txt_hashtags(line):
    tweet = json.loads(line)
    return clean_clean_text(tweet['clean_text']) + (' '.join(tweet['meta']['hashtags']))


def line2txt_clean(line):
    tweet = json.loads(line)
    return clean_clean_text(tweet['clean_text'])


def embed_tweets(
        source_f: PathLike,
        target_f: PathLike,
        model: Union[AutoModelBackend, SentenceTransformerBackend],
        include_hashtags: bool = True,
        verbose: bool = True):
    exit_if_exists(target_f)

    print('Loading texts...')
    with open(source_f) as f_in:
        if include_hashtags:
            texts = [line2txt_hashtags(line) for line in f_in]
        else:
            texts = [line2txt_clean(line) for line in f_in]

    print('Embedding texts...')
    embeddings = model.embed_documents(texts, verbose=verbose)

    print('Storing embeddings...')
    np.save(target_f, embeddings)


if __name__ == '__main__':
    args = TweetEmbeddingArgs(underscores_to_dashes=True).parse_args()
    if args.args_file is not None:
        print(f'Dropping keyword arguments and loading from file: {args.args_file}')
        args = TweetEmbeddingArgs().load(args.args_file)

    _include_hashtags = not args.excl_hashtags

    if args.file_in is None:
        file_in = f'data/{args.dataset}/tweets_filtered_{args.limit}.jsonl'
    else:
        file_in = args.file_in
    if args.file_out is None:
        file_out = f'data/{args.dataset}/tweets_embeddings_{args.limit}_{_include_hashtags}_{args.model}.npy'
    else:
        file_out = args.file_out

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
        _model_cache = ModelCache(args.model_cache)
        _model = _model_cache.get_embedder(args.model)

        embed_tweets(
            model=_model,
            source_f=file_in,
            target_f=file_out,
            include_hashtags=_include_hashtags,
        )

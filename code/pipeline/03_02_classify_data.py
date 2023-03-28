import json
import os
from typing import Optional, Union, Literal
from utils.models import ModelCache, ClassifierLiteral, SentenceTransformerBackend, AutoModelBackend
from utils.cluster import ClusterJobBaseArguments
from utils.io import exit_if_exists, batched_lines
from copy import deepcopy
import time
from datetime import datetime
import gc

PathLike = Union[str, bytes, os.PathLike]


class TweetClassifierArgs(ClusterJobBaseArguments):
    models: list[ClassifierLiteral] = ['cards', 'cardiff-sentiment', 'cardiff-emotion', 'cardiff-offensive',
                                       'cardiff-stance-climate', 'geomotions-orig', 'geomotions-ekman',
                                       'bertweet-sentiment', 'bertweet-emotions']
    nrc: bool = False  # set this flag to include NRC
    model_cache: str = 'data/models/'  # Location for cached models.
    file_in: Optional[str] = None  # The file to read data from (relative to source root)
    file_out: Optional[str] = None  # The file to write embeddings to (relative to source root)

    batch_size: Optional[int] = 10000  # size of the batch of tweets processed at once
    limit: Optional[int] = 10000  # Size of the dataset
    dataset: Optional[str] = 'climate2'  # Name of the dataset
    excl_hashtags: bool = False  # Set this flag to exclude hashtags in the embedding

    cluster_jobname: str = 'twitter-classify'
    cluster_workdir: str = 'twitter'

    cluster_gpu: bool = True


def clean_clean_text(txt):
    return txt.replace('MENTION', '').replace('URL', '').replace('HASHTAG', '')


def line2txt_hashtags(tweet):
    return clean_clean_text(tweet['clean_text']) + (' '.join(tweet['meta']['hashtags']))


def line2txt_clean(tweet):
    return clean_clean_text(tweet['clean_text'])


def classify_tweets(source_f: PathLike,
                    target_f: PathLike,
                    models: list[str],
                    nrc: bool,
                    model_cache: ModelCache,
                    batch_size: int,
                    include_hashtags: bool = True):
    with open(target_f, 'w') as f_out:
        for batch_i, lines_batch in enumerate(batched_lines(source_f, batch_size=batch_size)):
            print(f'== Processing Batch {batch_i} containing {len(lines_batch)} tweets ==')
            tweets_batch = [json.loads(line) for line in lines_batch]

            if include_hashtags:
                texts = [line2txt_hashtags(tweet) for tweet in tweets_batch]
            else:
                texts = [line2txt_clean(tweet) for tweet in tweets_batch]

            results = {}
            for model_name in models:
                start = time.time()
                print(f'[{datetime.now()}] Applying model {model_name}...')

                model = model_cache.get_classifier(model_name)
                results[model_name] = model.classify(texts)

                secs = time.time() - start
                print(f'  - Done after {secs // 60:.0f}m {secs % 60:.0f}s')

            if nrc:
                start = time.time()
                print(f'[{datetime.now()}] Applying NRC...')
                nrc_results = [NRCLex(t) for t in texts]
                nrc_results = [{emo: score for emo, score in r.affect_frequencies.items() if score > 0}
                               for r in nrc_results]
                secs = time.time() - start
                print(f'  - Done after {secs // 60:.0f}m {secs % 60:.0f}s')

            for i, tweet in enumerate(tweets_batch):
                tweet['classes'] = {model_name: results[model_name][i] for model_name in models}
                if nrc:
                    tweet['classes']['nrc'] = nrc_results[i]
                f_out.write(json.dumps(tweet) + '\n')
                f_out.flush()

            print('  - Memory cleanup')
            del model
            del tweets_batch
            del texts
            del nrc_results
            gc.collect()


if __name__ == '__main__':
    args = TweetClassifierArgs(underscores_to_dashes=True).parse_args()
    if args.args_file is not None:
        print(f'Dropping keyword arguments and loading from file: {args.args_file}')
        args = TweetClassifierArgs().load(args.args_file)

    _include_hashtags = not args.excl_hashtags

    if args.file_in is None:
        file_in = f'data/{args.dataset}/tweets_filtered_{args.limit}.jsonl'
    else:
        file_in = args.file_in
    if args.file_out is None:
        file_out = f'data/{args.dataset}/tweets_classified_{args.limit}_{_include_hashtags}.jsonl'
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
                                   required_models=args.models,
                                   data_files=[file_in])
        s_job = ClusterJob(config=s_config, file_handler=file_handler)
        cluster_args = deepcopy(args)
        cluster_args.file_in = os.path.join(s_config.datadir_path, file_in)
        cluster_args.file_out = os.path.join(s_config.datadir_path, file_out)
        cluster_args.model_cache = s_config.modeldir_path
        s_job.submit_job(main_script='pipeline/03_02_classify_data.py', params=cluster_args)
    else:
        from nrclex import NRCLex

        print('Initialising model cache')
        _model_cache = ModelCache(args.model_cache)

        print(f'Testing if output file exists already at {file_out}')
        # exit_if_exists(file_out)

        print('Running classifications...')
        classify_tweets(source_f=file_in,
                        target_f=file_out,
                        models=args.models,
                        model_cache=_model_cache,
                        include_hashtags=_include_hashtags,
                        batch_size=args.batch_size,
                        nrc=args.nrc)

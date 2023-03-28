import json
import os
from typing import Optional, Union, Literal
from utils.cluster import ClusterJobBaseArguments
from copy import deepcopy

PathLike = Union[str, bytes, os.PathLike]


class T2VArgs(ClusterJobBaseArguments):
    file_in: Optional[str] = None  # The file to read data from (relative to source root)
    file_emb: Optional[str] = None  # The file containing pre-trained doc2vec
    file_out: Optional[str] = None  # The file to write embeddings to (relative to source root)

    limit: Optional[int] = 7000000  # Size of the dataset
    dataset: Optional[str] = 'climate2'  # Name of the dataset
    excl_hashtags: bool = False  # Set this flag to exclude hashtags in the embedding

    cluster_jobname: str = 'twitter-top2vec'
    cluster_workdir: str = 'twitter'

    workers: int = 1


def clean_clean_text(txt):
    return txt.replace('MENTION', '').replace('URL', '').replace('HASHTAG', '')


def line2txt_hashtags(line):
    tweet = json.loads(line)
    return clean_clean_text(tweet['clean_text']) + (' '.join(tweet['meta']['hashtags']))


def line2txt_clean(line):
    tweet = json.loads(line)
    return clean_clean_text(tweet['clean_text'])


if __name__ == '__main__':
    args = T2VArgs(underscores_to_dashes=True).parse_args()
    if args.args_file is not None:
        print(f'Dropping keyword arguments and loading from file: {args.args_file}')
        args = T2VArgs().load(args.args_file)

    _include_hashtags = not args.excl_hashtags

    if args.file_in is None:
        file_in = f'data/{args.dataset}/tweets_filtered_{args.limit}.jsonl'
    else:
        file_in = args.file_in
    if args.file_emb is None:
        file_emb = f'data/{args.dataset}/doc2vec/tweets_top2vec_emb_{args.limit}_{_include_hashtags}'
    else:
        file_emb = args.file_emb
    if args.file_out is None:
        file_out = f'data/{args.dataset}/tweets_top2vec_{args.limit}_{_include_hashtags}'
    else:
        file_out = args.file_out

    if args.mode == 'cluster':
        from utils.cluster import Config as SlurmConfig
        from utils.cluster.job import ClusterJob
        from utils.cluster.files import FileHandler

        s_config = SlurmConfig.from_args(args)
        s_config.n_cpus = args.workers
        file_handler = FileHandler(config=s_config,
                                   local_basepath=os.getcwd(),
                                   requirements_txt='requirements_cluster.txt',
                                   include_dirs=['pipeline', 'utils'],
                                   data_files=[file_in])
        s_job = ClusterJob(config=s_config, file_handler=file_handler)
        cluster_args = deepcopy(args)
        cluster_args.file_in = os.path.join(s_config.datadir_path, file_in)
        cluster_args.file_emb = os.path.join(s_config.datadir_path, file_emb)
        cluster_args.file_out = os.path.join(s_config.datadir_path, file_out)
        s_job.submit_job(main_script='pipeline/06_02_top2vec.py', params=cluster_args)
    else:
        from utils.topics.top2vec import Top2Vec
        import json

        #
        # fn = 'data/climate2/tweets_filtered_10000.jsonl'

        print('Loading texts...')
        with open(file_in) as f_in:
            if _include_hashtags:
                texts = [line2txt_hashtags(line) for line in f_in]
            else:
                texts = [line2txt_clean(line) for line in f_in]
        print('running top2vec')
        model = Top2Vec(texts,
                        embedding_model_path=args.file_emb,
                        umap_args={'n_neighbors': 15,
                                   'n_components': 5,
                                   'metric': 'cosine',
                                   'verbose': True,
                                   'n_jobs': args.workers})
        print('saving model')
        model.save(file_out)

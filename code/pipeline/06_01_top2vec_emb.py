import json
import os
from typing import Optional, Union, Literal
from utils.cluster import ClusterJobBaseArguments
from copy import deepcopy

PathLike = Union[str, bytes, os.PathLike]


class T2VEmbArgs(ClusterJobBaseArguments):
    file_in: Optional[str] = None  # The file to read data from (relative to source root)
    file_out: Optional[str] = None  # The file to write embeddings to (relative to source root)

    limit: Optional[int] = 7000000  # Size of the dataset
    dataset: Optional[str] = 'climate2'  # Name of the dataset
    excl_hashtags: bool = False  # Set this flag to exclude hashtags in the embedding

    cluster_jobname: str = 'twitter-top2vec-emb'
    cluster_workdir: str = 'twitter'

    d2v_vector_size: int = 300
    d2v_min_count: int = 50
    d2v_hs: int = 1
    d2v_negative: int = 0
    d2v_epochs: int = 50
    d2v_sample: float = 1e-5
    d2v_window: int = 15
    d2v_dm: int = 0
    d2v_dbow_words: int = 1
    d2v_workers: int = 16  # will implicitly set cluster_n_cpus


def clean_clean_text(txt):
    return txt.replace('MENTION', '').replace('URL', '').replace('HASHTAG', '')


def line2txt_hashtags(line):
    tweet = json.loads(line)
    return clean_clean_text(tweet['clean_text']) + (' '.join(tweet['meta']['hashtags']))


def line2txt_clean(line):
    tweet = json.loads(line)
    return clean_clean_text(tweet['clean_text'])


if __name__ == '__main__':
    args = T2VEmbArgs(underscores_to_dashes=True).parse_args()
    if args.args_file is not None:
        print(f'Dropping keyword arguments and loading from file: {args.args_file}')
        args = T2VEmbArgs().load(args.args_file)

    _include_hashtags = not args.excl_hashtags

    if args.file_in is None:
        file_in = f'data/{args.dataset}/tweets_filtered_{args.limit}.jsonl'
    else:
        file_in = args.file_in
    if args.file_out is None:
        file_out = f'data/{args.dataset}/doc2vec/tweets_top2vec_emb_{args.limit}_{_include_hashtags}'
    else:
        file_out = args.file_out

    if args.mode == 'cluster':
        from utils.cluster import Config as SlurmConfig
        from utils.cluster.job import ClusterJob
        from utils.cluster.files import FileHandler

        s_config = SlurmConfig.from_args(args)
        s_config.cluster_n_cpus = args.d2v_workers
        file_handler = FileHandler(config=s_config,
                                   local_basepath=os.getcwd(),
                                   requirements_txt='requirements_cluster.txt',
                                   include_dirs=['pipeline', 'utils'],
                                   data_files=[file_in])
        s_job = ClusterJob(config=s_config, file_handler=file_handler)
        cluster_args = deepcopy(args)
        cluster_args.file_in = os.path.join(s_config.datadir_path, file_in)
        cluster_args.file_out = os.path.join(s_config.datadir_path, file_out)

        s_job.submit_job(main_script='pipeline/06_01_top2vec_emb.py', params=cluster_args)
    else:

        from gensim.utils import simple_preprocess
        from gensim.parsing.preprocessing import strip_tags
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        from utils.io import ensure_folder
        import logging

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        print('Loading texts...')
        with open(file_in) as f_in:
            if _include_hashtags:
                texts = [line2txt_hashtags(line) for line in f_in]
            else:
                texts = [line2txt_clean(line) for line in f_in]

        print('Creating tokenised tagged documents...')
        docs = [TaggedDocument(simple_preprocess(strip_tags(t), deacc=True), [i]) for i, t in enumerate(texts)]

        print('training model')
        model = Doc2Vec(docs,
                        vector_size=args.d2v_vector_size,
                        window=args.d2v_window,
                        min_count=args.d2v_min_count,
                        workers=args.d2v_workers,
                        hs=args.d2v_hs,
                        negative=args.d2v_negative,
                        epochs=args.d2v_epochs,
                        sample=args.d2v_sample,
                        dm=args.d2v_dm,
                        dbow_words=args.d2v_dbow_words)

        print('saving model')
        ensure_folder(file_out)
        model.save(file_out)

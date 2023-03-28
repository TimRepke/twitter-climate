import os
from typing import Optional, Union, Literal
import numpy as np
from utils.cluster import ClusterJobBaseArguments

PathLike = Union[str, bytes, os.PathLike]


class FullLandscapeArgs(ClusterJobBaseArguments):
    file_sampled: Optional[str] = None  # The file containing all already processed (down-sampled) tweets
    file_full: Optional[str] = None  # The file containing all relevant tweets
    file_emb_sample: Optional[str] = None
    file_emb_rest: Optional[str] = None
    file_tsne_sampled: Optional[str] = None
    index_dir: Optional[str] = None
    target_file: Optional[str] = None  # The file to write outputs to

    dataset: Optional[str] = 'climate2'  # Name of the dataset

    n_neighbours: int = 20
    metric: Literal['cosine', 'l2', 'ip'] = 'cosine'
    m: int = 16  # HNSW M for constriction
    efc: int = 200  # HNSW ef for construction
    efq: int = 200  # HNSW ef for query

    cluster_jobname: str = 'twitter-expand-tsne'
    cluster_workdir: str = 'twitter'


class NeighbourhoodIndex:
    def __init__(self,
                 tsne: np.ndarray, embeddings: np.ndarray, cache_location: PathLike,
                 n_neighbours: int, n_threads: int, metric: Literal['cosine', 'l2'],
                 efc: int, efq: int, m: int):
        self.tnse = tsne
        self.n_neighbours = n_neighbours
        self.n_threads = n_threads
        self.metric = metric
        self.m = m
        self.efq = efq
        self.efc = efc

        cache_file = os.path.join(cache_location, f'index_{metric}_{m}_{efc}.pkl')
        print(f'Cached index should be here: {cache_file}')
        if os.path.exists(cache_file):
            print(f'(loading existing index from disk {cache_location})')
            with open(cache_file, 'rb') as f:
                self.index = pickle.load(f)
        else:
            self.index = self._build_index(embeddings)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.index, f)

        self.index.set_ef(self.efq)  # ef should always be > k

    def _build_index(self, embeddings):
        ids = np.arange(len(embeddings))
        index = hnswlib.Index(space=self.metric, dim=embeddings.shape[1])
        index.set_num_threads(self.n_threads)
        index.init_index(max_elements=len(embeddings), ef_construction=self.efc, M=self.m)
        index.add_items(embeddings, ids, num_threads=self.n_threads)
        return index

    def get_neighbours(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.index.knn_query(embeddings, k=self.n_neighbours, num_threads=self.n_threads)

    def get_tsne_centroid(self, ids: np.ndarray):
        return np.mean(self.tnse[ids], axis=0)

    def get_tsne_location_iter(self, embeddings: np.ndarray) -> tuple[list[bool], np.ndarray]:
        succeeded = []
        locations = []

        for embedding in embeddings:
            try:
                ids, _ = self.get_neighbours(np.array([embedding]))
                locations.append(self.get_tsne_centroid(ids[0]))
                succeeded.append(True)
            except RuntimeError:
                succeeded.append(False)
                locations.append(np.array([.0, .0]))
        return succeeded, np.array(locations)

    def get_tsne_location_batch(self, embeddings: np.ndarray) -> tuple[list[bool], np.ndarray]:
        try:
            ids_batch, _ = self.get_neighbours(embeddings)
            return [True] * len(embeddings), np.array([self.get_tsne_centroid(ids) for ids in ids_batch])
        except RuntimeError:
            pass
        # the above thing may fail due to duplicates
        # see: https://github.com/nmslib/hnswlib/issues/373
        # instead, try for each embedding one by one
        return self.get_tsne_location_iter(embeddings)


def read_known_tweet_ids(source_sampled: PathLike):
    with open(source_sampled, 'r') as f_sampled:
        return [int(json.loads(line)['id']) for line in f_sampled]


def parse_json_line(line):
    t = json.loads(line)
    return int(t['id']), t['created_at'][:10]


def compute_full_landscape(file_sampled: PathLike,
                           file_full: PathLike,
                           file_emb_sample: PathLike,
                           file_emb_rest: PathLike,
                           file_tnse: PathLike,
                           index_cache: str,
                           target_file: PathLike,
                           n_neighbours: int,
                           n_threads: int,
                           metric: Literal['cosine', 'l2'],
                           efc: int,
                           efq: int,
                           m: int):
    # exit_if_exists(target_file)

    print('Reading ids of already projected tweets...')
    existing_ids = np.array(read_known_tweet_ids(file_sampled))
    existing_ids_map = {idx: i for i, idx in enumerate(existing_ids)}

    print('Reading tsne vectors...')
    tsne_vectors = np.load(file_tnse)

    print('Loading existing embeddings...')
    existing_embeddings = np.load(file_emb_sample)
    print('Loading ids and embeddings of unassigned tweets...')
    with open(file_emb_rest, 'rb') as f:
        new_ids = np.load(f)
        new_embeddings = np.load(f)
    new_ids_map = {idx: i for i, idx in enumerate(new_ids)}

    print('Building or loading neighbour index...')
    nn_index = NeighbourhoodIndex(tsne=tsne_vectors, embeddings=existing_embeddings, metric=metric,
                                  n_neighbours=n_neighbours, n_threads=n_threads, cache_location=index_cache,
                                  efc=efc, efq=efq, m=m)

    print('Read all ids...')
    with open(file_full, 'r') as f:
        tweets = [parse_json_line(line) for line in f]

    n_batches = 40000
    batch_size = int(len(tweets) / n_batches) + 1
    print(f'Going to process ~{n_batches} batches with {batch_size} tweets each.')
    with open(target_file, 'w') as f_out:
        for batch_i in tqdm(range(n_batches)):
            start = batch_i * batch_size
            end = (batch_i + 1) * batch_size
            batch = tweets[start:end]
            if len(batch) > 0:
                embeddings = np.array([new_embeddings[new_ids_map[t[0]]]
                                       for t in batch if t[0] in new_ids_map])

                succeeded, vectors = nn_index.get_tsne_location_batch(embeddings)

                vec_i = 0
                for tweet_id, tweet_day in batch:
                    if tweet_id in existing_ids_map:
                        vec = tsne_vectors[existing_ids_map[tweet_id]]
                        success = True
                        new = False
                    else:
                        vec = vectors[vec_i]
                        success = succeeded[vec_i]
                        new = True
                        vec_i += 1
                    f_out.write(f'{tweet_id},{tweet_day},{new:d},{success:d},{vec[0]:.5f},{vec[1]:.5f}\n')


if __name__ == '__main__':
    args = FullLandscapeArgs(underscores_to_dashes=True).parse_args()
    if args.args_file is not None:
        print(f'Dropping keyword arguments and loading from file: {args.args_file}')
        args = FullLandscapeArgs().load(args.args_file)

    if args.mode == 'cluster':
        from utils.cluster import Config as SlurmConfig
        from utils.cluster.job import ClusterJob
        from utils.cluster.files import FileHandler
        from copy import deepcopy

        s_config = SlurmConfig.from_args(args,
                                         env_vars_run={
                                             'OPENBLAS_NUM_THREADS': 1,
                                             'TRANSFORMERS_OFFLINE': 1
                                         })
        # s_config.qos = 'medium'
        file_handler = FileHandler(config=s_config,
                                   local_basepath=os.getcwd(),
                                   requirements_txt='requirements_cluster.txt',
                                   include_dirs=['pipeline', 'utils'])
        s_job = ClusterJob(config=s_config, file_handler=file_handler)
        cluster_args = deepcopy(args)

        cluster_args.file_sampled = os.path.join(s_config.datadir_path, f'data/{args.dataset}/{args.file_sampled}')
        cluster_args.file_full = os.path.join(s_config.datadir_path, f'data/{args.dataset}/{args.file_full}')
        cluster_args.file_emb_sample = os.path.join(s_config.datadir_path,
                                                    f'data/{args.dataset}/{args.file_emb_sample}')
        cluster_args.file_emb_rest = os.path.join(s_config.datadir_path, f'data/{args.dataset}/{args.file_emb_rest}')
        cluster_args.file_tsne_sampled = os.path.join(s_config.datadir_path,
                                                      f'data/{args.dataset}/{args.file_tsne_sampled}')
        cluster_args.index_dir = os.path.join(s_config.datadir_path, f'data/{args.dataset}/{args.index_dir}')
        cluster_args.target_file = os.path.join(s_config.datadir_path, f'data/{args.dataset}/{args.target_file}')

        s_job.submit_job(main_script='pipeline/04_04_03_join_landscape.py', params=cluster_args)
    else:
        from tqdm import tqdm
        import hnswlib
        from utils.io import exit_if_exists
        import json
        import pickle

        compute_full_landscape(
            file_sampled=args.file_sampled,
            file_full=args.file_full,
            file_tnse=args.file_tsne_sampled,
            file_emb_rest=args.file_emb_rest,
            file_emb_sample=args.file_emb_sample,
            target_file=args.target_file,
            index_cache=args.index_dir,
            n_threads=args.cluster_n_cpus,
            n_neighbours=args.n_neighbours,
            metric=args.metric,
            efc=args.efc,
            efq=args.efq,
            m=args.m
        )

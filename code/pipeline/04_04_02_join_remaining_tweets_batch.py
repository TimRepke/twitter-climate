import os
from typing import Optional, Union, Literal
import numpy as np
from utils.cluster import ClusterJobBaseArguments

PathLike = Union[str, bytes, os.PathLike]


class TweetExtendArgs(ClusterJobBaseArguments):
    file_sampled: Optional[str] = None  # The file containing all already processed (down-sampled) tweets
    file_full: Optional[str] = None  # The file containing all relevant tweets
    file_emb_sample: Optional[str] = None
    file_emb_rest: Optional[str] = None
    file_labels: Optional[str] = None

    target_folder: Optional[str] = None  # The path to write outputs to

    dataset: Optional[str] = 'climate2'  # Name of the dataset

    n_neighbours: int = 20
    metric: Literal['cosine', 'l2', 'ip'] = 'cosine'
    m: int = 16  # HNSW M for constriction
    efc: int = 200  # HNSW ef for construction
    efq: int = 200  # HNSW ef for query

    cluster_jobname: str = 'twitter-expand-join-batched'
    cluster_workdir: str = 'twitter'


class MajorityIndex:
    def __init__(self,
                 labels: np.ndarray, embeddings: np.ndarray, cache_location: PathLike,
                 n_neighbours: int, n_threads: int, metric: Literal['cosine', 'l2'],
                 efc: int, efq: int, m: int):
        self.labels = labels
        self.n_neighbours = n_neighbours
        self.n_threads = n_threads
        self.metric = metric
        self.m = m
        self.efq = efq
        self.efc = efc

        cache_file = os.path.join(cache_location, f'index_{metric}_{m}_{efc}.pkl')

        if os.path.exists(cache_file):
            print(f'(loading existing index from disk {cache_location})')
            with open(cache_file, 'rb') as f:
                self.index = pickle.load(f)
        else:
            self.index = self._build_index(labels, embeddings)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.index, f)

        self.index.set_ef(self.efq)  # ef should always be > k

    def _build_index(self, labels, embeddings):
        ids = np.arange(len(labels))
        index = hnswlib.Index(space=self.metric, dim=embeddings.shape[1])
        index.set_num_threads(self.n_threads)
        index.init_index(max_elements=len(labels), ef_construction=self.efc, M=self.m)
        index.add_items(embeddings, ids, num_threads=self.n_threads)
        return index

    def get_neighbours(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.index.knn_query(embeddings, k=self.n_neighbours, num_threads=self.n_threads)

    def majority_label_from_ids(self, ids):
        labels = self.labels[ids]
        unique_labels, counts = np.unique(labels, return_counts=True)
        return unique_labels[counts.argmax()]

    def get_majority_labels(self, embeddings: np.ndarray) -> np.ndarray:
        try:
            # try batch processing first
            # it may fail due to a bulk of duplicates
            # https://github.com/nmslib/hnswlib/issues/373
            ids, _ = self.get_neighbours(embeddings)
            return np.array([self.majority_label_from_ids(i) for i in ids])
        except RuntimeError:
            pass

        # go through it one-by-one and assign outlier class to the failing item instead
        return self.get_majority_labels_iter(embeddings)

    def get_majority_labels_iter(self, embeddings: np.ndarray):
        labels = []
        for embedding in embeddings:
            try:
                labels.append(self.get_majority_label(embedding))
            except RuntimeError:
                print('Failed to find neighbours')
                labels.append(-1)
        return np.array(labels)

    def get_majority_label(self, embedding: np.ndarray) -> np.ndarray:
        ids, _ = self.get_neighbours(np.array([embedding]))
        return self.majority_label_from_ids(ids[0])


class ProximityIndex:
    def __init__(self, labels: np.ndarray, embeddings: np.ndarray, n_jobs: int, metric: Literal['cosine', 'l2', 'ip']):
        self.topics = np.unique(labels)[1:]  # with this method, outliers are not allowed
        self.centroids = np.array([np.mean(embeddings[labels == topic_i], axis=0) for topic_i in self.topics])
        self.n_jobs = n_jobs
        self.metric = {
            'l2': 'euclidean',
            'cosine': 'cosine',
            'ip': 'euclidean'  # ip doesn't exist in sklearn, fallback to euclid
        }[metric]

    def get_closest(self, embedding: np.ndarray):
        distances = np.array([cosine(embedding, centroid) for centroid in self.centroids])
        return self.topics[distances.argmax()]

    def get_closest_bulk(self, embeddings: np.ndarray):
        return pairwise_distances(embeddings, self.centroids, metric=self.metric, n_jobs=self.n_jobs).argmax(axis=1)


def read_known_tweet_ids(source_sampled: PathLike):
    with open(source_sampled, 'r') as f_sampled:
        return [int(json.loads(line)['id']) for line in f_sampled]


def parse_json_line(line):
    t = json.loads(line)
    return int(t['id']), t['created_at'][:10]


def assign_topics(file_sampled: PathLike,
                  file_full: PathLike,
                  file_emb_sample: PathLike,
                  file_emb_rest: PathLike,
                  file_labels: PathLike,
                  target_folder: str,
                  n_neighbours: int,
                  n_threads: int,
                  metric: Literal['cosine', 'l2'],
                  efc: int,
                  efq: int,
                  m: int):
    exit_if_exists(os.path.join(target_folder, f'labels.csv'))
    os.makedirs(target_folder, exist_ok=True)

    print('Reading ids of already assigned tweets...')
    existing_ids = np.array(read_known_tweet_ids(file_sampled))
    existing_ids_map = {idx: i for i, idx in enumerate(existing_ids)}
    print('Reading labels of already assigned tweets...')
    existing_labels = np.load(file_labels)

    print('Loading existing embeddings...')
    existing_embeddings = np.load(file_emb_sample)
    print('Loading ids and embeddings of unassigned tweets...')
    with open(file_emb_rest, 'rb') as f:
        new_ids = np.load(f)
        new_embeddings = np.load(f)
    new_ids_map = {idx: i for i, idx in enumerate(new_ids)}

    def get_emb_by_id(tid):
        if tid in existing_ids_map:
            return existing_embeddings[existing_ids_map[tid]]
        return new_embeddings[new_ids_map[tid]]

    print('Building majority vote index...')
    majority_index = MajorityIndex(labels=existing_labels, embeddings=existing_embeddings, metric=metric,
                                   n_neighbours=n_neighbours, n_threads=n_threads, cache_location=target_folder,
                                   efc=efc, efq=efq, m=m)
    print('Building closest centroid index...')
    proximity_index = ProximityIndex(labels=existing_labels, embeddings=existing_embeddings, n_jobs=n_threads,
                                     metric=metric, )

    print('Prepare batches...')
    with open(file_full, 'r') as f:
        tweets = [parse_json_line(line) for line in f]

    n_batches = 40000
    batch_size = int(len(tweets) / n_batches) + 1
    print(f'Going to process ~{n_batches} batches with {batch_size} tweets each.')
    with open(os.path.join(target_folder, f'labels.csv'), 'w') as f_out:
        for batch_i in tqdm(range(n_batches)):
            start = batch_i * batch_size
            end = (batch_i + 1) * batch_size
            batch = tweets[start:end]
            if len(batch) > 0:
                embeddings = np.array([get_emb_by_id(t[0]) for t in batch])

                labels_majority = majority_index.get_majority_labels(embeddings)
                labels_proximity = proximity_index.get_closest_bulk(embeddings)

                for (tweet_id, tweet_day), label_majority, label_proximity in \
                        zip(batch, labels_majority, labels_proximity):
                    keep_majority = existing_labels[existing_ids_map[tweet_id]] \
                        if tweet_id in existing_ids_map else label_majority
                    keep_proximity = existing_labels[existing_ids_map[tweet_id]] \
                        if tweet_id in existing_ids_map else label_proximity

                    f_out.write(f'{tweet_id},{tweet_day},'
                                f'{keep_majority},{keep_proximity},'
                                f'{label_majority},{label_proximity}\n')


if __name__ == '__main__':
    args = TweetExtendArgs(underscores_to_dashes=True).parse_args()
    if args.args_file is not None:
        print(f'Dropping keyword arguments and loading from file: {args.args_file}')
        args = TweetExtendArgs().load(args.args_file)

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
        s_config.qos = 'medium'
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
        cluster_args.file_labels = os.path.join(s_config.datadir_path, f'data/{args.dataset}/{args.file_labels}')
        cluster_args.target_folder = os.path.join(s_config.datadir_path, f'data/{args.dataset}/{args.target_folder}')

        cluster_args.model_cache = s_config.modeldir_path
        s_job.submit_job(main_script='pipeline/04_04_02_join_remaining_tweets_batch.py', params=cluster_args)
    else:
        from tqdm import tqdm
        import hnswlib
        from scipy.spatial.distance import cosine
        from sklearn.metrics import pairwise_distances
        from utils.io import exit_if_exists
        import json
        import pickle

        assign_topics(
            file_sampled=args.file_sampled,
            file_full=args.file_full,
            file_labels=args.file_labels,
            file_emb_rest=args.file_emb_rest,
            file_emb_sample=args.file_emb_sample,
            target_folder=args.target_folder,
            n_threads=args.cluster_n_cpus,
            n_neighbours=args.n_neighbours,
            metric=args.metric,
            efc=args.efc,
            efq=args.efq,
            m=args.m
        )

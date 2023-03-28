from typing import Optional, Literal
from tap import Tap
import numpy as np
import hdbscan
from typing import Optional, Literal
import os
from pathlib import Path
from utils.cluster import ClusterJobBaseArguments
from copy import deepcopy


class ClusteringArgs(ClusterJobBaseArguments):
    file_in: Optional[str] = None  # The file containing tweet embeddings (relative to source root)

    limit: Optional[int] = 10000  # Size of the dataset
    dataset: Optional[str] = 'climate2'  # Name of the dataset
    projection: Literal['umap', 'tsne'] = 'tsne'  # The dimensionality reduction method to use

    min_cluster_size: int = 2
    min_samples: int = 10
    cluster_selection_epsilon: float = 1.
    cluster_selection_method: Literal['leaf', 'eom'] = 'eom'
    allow_single_cluster: bool = False
    alpha: float = 1.

    cluster_jobname: str = 'twitter-hdbscan'
    cluster_workdir: str = 'twitter'


if __name__ == '__main__':
    args = ClusteringArgs(underscores_to_dashes=True).parse_args()
    if args.args_file is not None:
        print(f'Dropping keyword arguments and loading from file: {args.args_file}')
        args = ClusteringArgs().load(args.args_file)

    if args.file_in is None:
        file_in = f'data/{args.dataset}/topics/layout_{args.limit}_{args.projection}.npy'
    else:
        file_in = args.file_in

    if args.mode == 'cluster':
        from utils.cluster import Config as SlurmConfig
        from utils.cluster.job import ClusterJob
        from utils.cluster.files import FileHandler

        s_config = SlurmConfig.from_args(args)
        file_handler = FileHandler(config=s_config,
                                   local_basepath=os.getcwd(),
                                   requirements_txt='requirements_cluster.txt',
                                   include_dirs=['pipeline', 'utils'])
        s_job = ClusterJob(config=s_config, file_handler=file_handler)

        cluster_args = deepcopy(args)
        cluster_args.mode = 'local'
        cluster_args.file_in = os.path.join(s_config.datadir_path, file_in)
        s_job.submit_job(main_script='pipeline/04_02_hdbscan_param_sweep.py', params=cluster_args)

    else:
        print('Loading layout...')
        layout = np.load(file_in)

        print('Initialising and fitting HDBSCAN...')
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            core_dist_n_jobs=args.cluster_n_cpus,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
            cluster_selection_method=args.cluster_selection_method,
            allow_single_cluster=args.allow_single_cluster,
            alpha=args.alpha)
        clusterer.fit(layout)

        print('Constructing single linkage tree...')
        tree = clusterer.single_linkage_tree_

        print('Running parameter sweep (keeping only those with > 100 topics)...')
        for cutoff in [0, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5]:
            # , 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10]:
            for min_cluster_size in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100, 150, 200, 500, 1000]:
                cnts = np.unique(tree.get_clusters(cut_distance=cutoff,
                                                   min_cluster_size=min_cluster_size),
                                 return_counts=True)[1]
                if len(cnts) > 100:
                    print(f'cutoff: {cutoff}, min_size: {min_cluster_size:,}, num clusters: {len(cnts):,}, '
                          f'mean c_size: {cnts[1:].mean():.1f}, median: {np.median(cnts[1:]):.1f}, '
                          f'outliers: {cnts[0]:,}, largest cluster: {cnts[1:].max():,}')
        print('done')

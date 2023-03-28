from typing import Optional, Literal
import os
from pathlib import Path
from utils.cluster import ClusterJobBaseArguments
from copy import deepcopy


class LayoutArgs(ClusterJobBaseArguments):
    model: Literal['minilm', 'bertopic'] = 'minilm'  # The embedding model to use.
    file_in: Optional[str] = None  # The file containing tweet embeddings (relative to source root)
    file_out: Optional[str] = None  # The file to write layout to (relative to source root)

    limit: Optional[int] = 10000  # Size of the dataset
    dataset: Optional[str] = 'climate2'  # Name of the dataset
    excl_hashtags: bool = False  # Set this flag to exclude hashtags in the embedding

    projection: Literal['umap', 'tsne'] = 'tsne'  # The dimensionality reduction method to use

    # tsne args
    tsne_perplexity: int = 40
    tsne_exaggeration: int = None
    tsne_early_exaggeration_iter: int = 250
    tsne_early_exaggeration: int = 20
    tsne_initialization: Literal['random', 'pca', 'spectral'] = 'pca'
    tsne_metric: str = 'cosine'
    tsne_n_jobs: int = 8
    tsne_dof: float = 0.6
    tsne_random_state: int = 3
    tsne_verbose: bool = True
    tsne_neighbors: Literal['annoy', 'exact', 'pynndescent', 'hnsw', 'approx', 'auto'] = 'hnsw'
    tsne_prefit_size: Optional[int] = None  # set this to fit tsne in stages, this is the size of the initial set
    tsne_prefit_perplexity: Optional[int] = None  # the perplexity used for the initial fit

    # umap args
    umap_n_neighbors: int = 15
    umap_n_components: int = 2
    umap_metric: str = 'cosine'
    umap_output_metric: str = 'euclidean'
    umap_min_dist: float = 0.1
    umap_spread: float = 1.0
    umap_local_connectivity: int = 1
    umap_repulsion_strength: float = 1.0
    umap_negative_sample_rate: int = 5
    umap_random_state: bool = None
    umap_densmap: bool = False
    umap_set_op_mix_ratio: float = 1.0
    umap_dens_lambda: float = 2.0
    umap_dens_frac: float = 0.3
    umap_dens_var_shift: float = 0.1

    cluster_jobname: str = 'twitter-layout'
    cluster_workdir: str = 'twitter'


def staged_tsne():
    from openTSNE.affinity import PerplexityBasedNN
    from openTSNE.initialization import pca
    from openTSNE import TSNEEmbedding

    prefit_perplexity = args.tsne_prefit_perplexity or 500

    print('Sampling random set for partial tSNE...')
    indices = np.random.permutation(list(range(embeddings.shape[0])))
    reverse = np.argsort(indices)
    x_sample = embeddings[indices[:args.tsne_prefit_size]]
    x_rest = embeddings[indices[args.tsne_prefit_size:]]

    print('Computing sample affinities...')
    sample_affinities = PerplexityBasedNN(x_sample,
                                          metric=args.tsne_metric,
                                          perplexity=prefit_perplexity,
                                          n_jobs=args.tsne_n_jobs,
                                          method=args.tsne_neighbors,
                                          random_state=args.tsne_random_state,
                                          verbose=args.tsne_verbose)
    print('Computing all affinities...')
    full_affinities = PerplexityBasedNN(embeddings,
                                        metric=args.tsne_metric,
                                        perplexity=args.tsne_perplexity,
                                        n_jobs=args.tsne_n_jobs,
                                        method=args.tsne_neighbors,
                                        random_state=args.tsne_random_state,
                                        verbose=args.tsne_verbose)
    print('Computing PCA for sample...')
    sample_init = pca(x_sample, random_state=args.tsne_random_state)

    print('Fitting tSNE on sample...')
    sample_embedder = TSNE(n_jobs=args.tsne_n_jobs,
                           verbose=args.tsne_verbose,
                           dof=args.tsne_dof)
    sample_embedding = sample_embedder.fit(affinities=sample_affinities,
                                           initialization=sample_init)

    print('Roughly placing the rest of the data to their nearest neighbour...')
    rest_init = sample_embedding.prepare_partial(x_rest, k=1, perplexity=1 / 3)
    # putting things back in the right order
    init_full = np.vstack((sample_embedding, rest_init))[reverse]
    # re-centering image space
    init_full = init_full / (np.std(init_full[:, 0]) * 10000)

    print('Setting up layout optimisation...')
    layout_base = TSNEEmbedding(embedding=init_full,
                                affinities=full_affinities,
                                dof=args.tsne_dof,
                                n_jobs=args.tsne_n_jobs,
                                verbose=args.tsne_verbose,
                                random_state=args.tsne_random_state, )
    print('Optimising layout 1/2...')
    layout1 = layout_base.optimize(n_iter=args.tsne_early_exaggeration_iter,
                                   exaggeration=args.tsne_early_exaggeration,
                                   momentum=0.5)
    print('Optimising layout 2/2...')
    layout2 = layout1.optimize(n_iter=int(args.tsne_early_exaggeration_iter / 2),
                               exaggeration=int(args.tsne_early_exaggeration / 3),
                               momentum=0.8)
    return layout2


if __name__ == '__main__':
    print('I will load pre-computed embeddings and reduce them to a 2D space!')
    args = LayoutArgs(underscores_to_dashes=True).parse_args()
    if args.args_file is not None:
        print(f'Dropping keyword arguments and loading from file: {args.args_file}')
        args = LayoutArgs().load(args.args_file)
    _include_hashtags = not args.excl_hashtags

    if args.file_in is None:
        file_in = f'data/{args.dataset}/tweets_embeddings_{args.limit}_{_include_hashtags}_{args.model}.npy'
    else:
        file_in = args.file_in
    if args.file_out is None:
        file_out = f'data/{args.dataset}/topics/layout_{args.limit}_{args.projection}.npy'
    else:
        file_out = args.file_out

    if args.mode == 'cluster':
        from utils.cluster import Config as SlurmConfig
        from utils.cluster.job import ClusterJob
        from utils.cluster.files import FileHandler

        s_config = SlurmConfig.from_args(args)
        file_handler = FileHandler(config=s_config,
                                   local_basepath=os.getcwd(),
                                   requirements_txt='requirements_cluster.txt',
                                   include_dirs=['pipeline', 'utils'],
                                   data_files=[file_in])
        s_job = ClusterJob(config=s_config, file_handler=file_handler)

        cluster_args = deepcopy(args)
        cluster_args.mode = 'local'
        cluster_args.file_in = os.path.join(s_config.datadir_path, file_in)
        cluster_args.file_out = os.path.join(s_config.datadir_path, file_out)
        s_job.submit_job(main_script='pipeline/04_01_layout.py', params=cluster_args)
    else:
        import numpy as np

        print('Loading embeddings...')
        embeddings = np.load(file_in)

        if args.projection == 'umap':
            print('Fitting UMAP...')
            from umap import UMAP

            umap_args = {k[5:]: v for k, v in args.as_dict().items() if k.startswith('umap')}
            mapper = UMAP(**umap_args)
            layout = mapper.fit_transform(embeddings)
        elif args.projection == 'tsne':
            from openTSNE import TSNE

            if args.tsne_prefit_size is not None:
                print('Fitting tSNE in parts...')
                layout = staged_tsne()
            else:
                print('Fitting tSNE (all in one)...')
                tsne_args = {k[5:]: v for k, v in args.as_dict().items() if k.startswith('tsne')}
                mapper = TSNE(**tsne_args)
                layout = mapper.fit(embeddings)
        else:
            raise NotImplementedError('Unknown projection method.')

        print('Storing layout...')
        os.makedirs(Path(file_out).parent, exist_ok=True)  # ensure the target directory exists first...
        np.save(file_out, layout)

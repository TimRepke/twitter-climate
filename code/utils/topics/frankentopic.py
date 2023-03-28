import os.path

import numpy as np
from scipy.sparse.csr import csr_matrix
from typing import Type, Optional, Union, Literal, List, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
import openTSNE
import hdbscan

from utils.models import ModelCache

TopicListing = List[List[Tuple[str, float]]]


@dataclass
class VectorizerArgs:
    min_df: Union[float, int] = 0
    max_df: Union[float, int] = 1.0
    ngram_range: tuple[int, int] = (1, 1)
    max_features: int = None
    lowercase: bool = True
    use_idf: bool = True
    smooth_idf: bool = True
    stop_words: Optional[Union[set[str], list[str], Literal['english']]] = None


@dataclass
class UMAPArgs:
    n_neighbors: int = 15
    n_components: int = 2
    metric: str = "cosine"
    output_metric: str = "euclidean"
    min_dist: float = 0.1
    spread: float = 1.0
    local_connectivity: int = 1
    repulsion_strength: float = 1.0
    negative_sample_rate: int = 5
    random_state: bool = None
    densmap: bool = False
    set_op_mix_ratio: float = 1.0
    dens_lambda: float = 2.0
    dens_frac: float = 0.3
    dens_var_shift: float = 0.1


@dataclass
class TSNEArgs:
    perplexity: int = 30
    exaggeration: float = None
    early_exaggeration_iter: int = 250
    early_exaggeration: float = 12
    initialization: Literal['random', 'pca', 'spectral'] = "random"
    metric: str = 'cosine'
    n_jobs: int = 8
    dof: float = 1.
    random_state: int = 3


@dataclass
class KMeansArgs:
    max_n_topics: int = 20  # if min_docs_per_topic is None -> exact num topics
    min_docs_per_topic: int = None  # if none, ignored


@dataclass
class HDBSCANArgs:
    min_cluster_size: int = 5
    min_samples: int = None
    cluster_selection_epsilon: float = 0.0
    alpha: float = 1.0
    cluster_selection_method: Literal['eom', 'leaf'] = 'leaf'
    allow_single_cluster: bool = False


def mmr(doc_embedding: np.ndarray,
        word_embeddings: np.ndarray,
        words: list[str],
        top_n: int = 5,
        diversity: float = 0.8) -> list[str]:
    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr_ = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr_)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]


def get_top_mmr(topics_tfidf: TopicListing, n_tokens: int, model_cache_location: str, model: str, mmr_diversity: float):
    topics_mmr = []
    print('Improving topic keywords...')
    model_cache = ModelCache(cache_dir=model_cache_location)
    embedder = model_cache.get_embedder(model)

    for topic in tqdm(topics_tfidf):
        words = [w[0] for w in topic]
        word_embeddings = embedder.embed_words(words, verbose=False)
        topic_embedding = embedder.embed_documents([' '.join(words)], verbose=False).reshape(1, -1)
        topic_words = mmr(topic_embedding, word_embeddings, words,
                          top_n=n_tokens, diversity=mmr_diversity)
        topics_mmr.append([
            (word, value) for word, value in topic if word in topic_words
        ])
    return topics_mmr


def get_top_tfidf(vectors, token_lookup, n_tokens: int = 20) -> TopicListing:
    print('Computing top tf-idf words per topic...')

    result = []
    for topic_i in range(vectors.shape[0]):
        rank = np.argsort(vectors[topic_i].todense())
        result.append([
            (token_lookup[rank[0, -(token_i + 1)]], vectors[topic_i, rank[0, -(token_i + 1)]])
            for token_i in range(n_tokens)
        ])
    return result


class FrankenTopic:
    def __init__(self,
                 cluster_args: Union[HDBSCANArgs, KMeansArgs] = None,
                 n_words_per_topic: int = 20,
                 n_candidates: int = 40,
                 mmr_diversity: float = 0.8,
                 emb_model: str = 'minilm',
                 model_cache_location: str = 'data/models/',
                 dr_args: Union[UMAPArgs, TSNEArgs] = None,
                 cache_layout: str = None,
                 vectorizer_args: VectorizerArgs = None):
        if vectorizer_args is None:
            vectorizer_args = VectorizerArgs()
        if dr_args is None:
            dr_args = TSNEArgs()
        if cluster_args is None:
            cluster_args = KMeansArgs()

        self.vectorizer_args = vectorizer_args
        self.dr_args = dr_args
        self.cluster_args = cluster_args
        self.n_candidates = n_candidates
        self.n_words_per_topic = n_words_per_topic
        self.mmr_diversity = mmr_diversity
        self.layout_cache_file = cache_layout

        self.emb_model = emb_model
        self.model_cache_location = model_cache_location

        self.clusterer = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.vocab: Optional[dict[int, str]] = None
        self.tf_idf_vecs: Optional[csr_matrix] = None
        self.layout: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self._is_fit = False

    def _run_kmeans(self, args: KMeansArgs):
        clustering = KMeans(n_clusters=args.max_n_topics)
        self.labels = clustering.fit_predict(self.layout)

        if args.min_docs_per_topic is not None:
            print('Dumping small clusters...')
            labels = np.copy(self.labels)  # shift all by one, so cluster 0 is outlier cluster
            i = 1
            for cluster, count in zip(*np.unique(self.labels + 1, return_counts=True)):
                if count > args.min_docs_per_topic:
                    labels[self.labels == (cluster - 1)] = i
                    i += 1
                else:
                    labels[self.labels == (cluster - 1)] = 0

            # in case no clusters were dropped, shift all back again
            if np.max(labels) == len(np.unique(labels)):
                labels = labels - 1

            self.labels = labels
            print(f'Now left with {len(np.unique(labels))} clusters.')

    def fit(self, tweets: list[str], embeddings: np.ndarray):
        if self.layout_cache_file is not None and os.path.isfile(self.layout_cache_file):
            print('Loading cached layout...')
            self.layout = np.load(self.layout_cache_file)
        else:
            print(self.dr_args)
            if self.dr_args.__class__ == UMAPArgs:
                print('Fitting UMAP...')
                mapper = UMAP(**asdict(self.dr_args))
                self.layout = mapper.fit_transform(embeddings)
            else:
                print('Fitting tSNE...')
                mapper = openTSNE.TSNE(**asdict(self.dr_args))
                self.layout = mapper.fit(embeddings)
            if self.layout_cache_file is not None:
                print('Storing layout...')
                np.save(self.layout_cache_file, self.layout)

        print(self.cluster_args)
        if self.cluster_args.__class__ == KMeansArgs:
            print('Clustering with k-means...')
            self._run_kmeans(self.cluster_args)
        else:
            print('Clustering with HDBSCAN')
            self.clusterer = hdbscan.HDBSCAN(**asdict(self.cluster_args))
            self.clusterer.fit(self.layout)
            self.labels = self.clusterer.labels_ + 1  # increment by one, so -1 (outlier) cluster becomes 0

        print('Grouping tweets...')
        grouped_texts = [
            [tweets[i] for i in np.argwhere(self.labels == label).reshape(-1, )]
            for label in np.unique(self.labels)
        ]

        # note, that BERTopic does something slightly different:
        # https://github.com/MaartenGr/BERTopic/blob/15ea0cd804d35c1f11c6692f33c3666b648dd6c8/bertopic/_ctfidf.py
        print('Vectorising groups...')
        self.vectorizer = TfidfVectorizer(**asdict(self.vectorizer_args))
        self.tf_idf_vecs = self.vectorizer.fit_transform([' '.join(g) for g in grouped_texts])
        self.vocab = {v: k for k, v in self.vectorizer.vocabulary_.items()}

        self._is_fit = True

    def get_top_n_mmr(self, n_tokens: int = None) -> list[list[tuple[str, float]]]:
        assert self._is_fit
        tfidf_candidates = self.get_top_n_tfidf(self.n_candidates)
        return get_top_mmr(topics_tfidf=tfidf_candidates,
                           n_tokens=n_tokens or self.n_words_per_topic,
                           model_cache_location=self.model_cache_location,
                           model=self.emb_model,
                           mmr_diversity=self.mmr_diversity)

    def get_top_n_tfidf(self, n_tokens: int = 20) -> list[list[tuple[str, float]]]:
        assert self._is_fit
        return get_top_tfidf(self.tf_idf_vecs, token_lookup=self.vocab, n_tokens=n_tokens)

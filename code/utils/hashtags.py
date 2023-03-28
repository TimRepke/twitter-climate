import re
from datetime import datetime
from dataclasses import dataclass
import sqlite3
from collections import defaultdict, Counter
import scipy.spatial.distance
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import Union, Literal
import pandas as pd
import numpy as np


class HashtagSimilarities:
    def __init__(self):
        pass


class GroupedHashtags:
    def __init__(self,
                 groups, fake_docs,
                 vectoriser: Union[TfidfVectorizer, CountVectorizer]):
        self.fake_docs = fake_docs
        self.groups = groups

        self.vectoriser = vectoriser
        self.vectors = self.vectoriser.fit_transform(self.fake_docs)
        self.vocab = {v: k for k, v in self.vectoriser.vocabulary_.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def most_common(self, top_n=5, include_count=True, include_hashtag=True, least_common=False) -> \
            list[tuple[str, Union[str, int, float, tuple[str, int], tuple[str, float]]]]:
        assert include_hashtag or include_count

        if least_common:
            indices = np.asarray(np.argsort(self.vectors.todense(), axis=1)[:, :top_n])
        else:
            indices = np.flip(np.asarray(np.argsort(self.vectors.todense(), axis=1)[:, -top_n:]), axis=1)

        token_value_pairs = [
            [(self.vocab[ind], self.vectors[row_i, ind]) for ind in row]
            for row_i, row in enumerate(indices)
        ]
        if include_count and include_hashtag:
            return [(group, tvps) for group, tvps in zip(self.groups, token_value_pairs)]
        if include_hashtag:
            return [(group, [tvp[0] for tvp in tvps]) for group, tvps in zip(self.groups, token_value_pairs)]
        if include_count:
            return [(group, [tvp[1] for tvp in tvps]) for group, tvps in zip(self.groups, token_value_pairs)]

    def pairwise_similarities(self, metric: Literal['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                                                    'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
                                                    'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis',
                                                    'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
                                                    'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
                                                    'wminkowski', 'yule']) -> np.ndarray:
        return scipy.spatial.distance.cdist(self.vectors.todense(), self.vectors.todense(), metric=metric)

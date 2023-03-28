from pathlib import Path
import re
import os
from sentence_transformers import SentenceTransformer
from transformers import (AutoModel,
                          AutoModelForSequenceClassification,
                          AutoTokenizer, TextClassificationPipeline)
import torch
from scipy.special import softmax
from typing import Literal, Optional, Union
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod


class Classifier:
    def __init__(self, hf_name: str, labels: list[str]):
        self.hf_name = hf_name
        self.labels = labels
        self._classifier: Optional[TextClassificationPipeline] = None

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @property
    def label2id(self) -> dict[str, int]:
        return {k: i for i, k in enumerate(self.labels)}

    @property
    def id2label(self) -> dict[int, str]:
        return {i: k for i, k in enumerate(self.labels)}

    def store(self, target_dir: Path):
        target = str(target_dir)
        os.makedirs(target, exist_ok=True)

        pretrained_model = AutoModelForSequenceClassification.from_pretrained(self.hf_name)
        pretrained_model.save_pretrained(target)

        tokenizer = AutoTokenizer.from_pretrained(self.hf_name)
        tokenizer.save_pretrained(target)

    def load(self, target_dir: Path, device: int):
        if self._classifier is None:
            target = str(target_dir)
            tokenizer = AutoTokenizer.from_pretrained(target, use_fast=True)
            pretrained_model = AutoModelForSequenceClassification.from_pretrained(target,
                                                                                  num_labels=self.num_labels,
                                                                                  label2id=self.label2id,
                                                                                  id2label=self.id2label)
            self._classifier = TextClassificationPipeline(model=pretrained_model, tokenizer=tokenizer, device=device)

    def classify(self, texts: list[str], return_all_scores: bool = False):
        scores = self._classifier(texts, return_all_scores=return_all_scores)
        if return_all_scores:
            return [{score['label']: score['score'] for score in scores_i} for scores_i in scores]
        return [{score['label']: score['score']} for score in scores]


class CARDSClassifier(Classifier):
    URL = 'https://socialanalytics.ex.ac.uk/cards/models.zip'

    def store(self, target_dir: Path):
        import requests
        import zipfile
        target = str(target_dir)
        zip_file = os.path.join(target, 'cards_model.zip')
        os.makedirs(target, exist_ok=True)

        # wget -nc --no-check-certificate https://socialanalytics.ex.ac.uk/cards/models.zip
        r = requests.get(self.URL)
        with open(zip_file, 'wb') as model_file:
            model_file.write(r.content)

        # unzip -n models.zip
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(target)

        # unzip -n models/CARDS_RoBERTa_Classifier.zip -d models
        with zipfile.ZipFile(f'{target}/models/CARDS_RoBERTa_Classifier.zip', 'r') as zip_ref:
            zip_ref.extractall(f'{target}/models/')

        os.remove(zip_file)

    def load(self, target_dir: Path, device: int):
        if self._classifier is None:
            from simpletransformers.classification import ClassificationModel

            model = ClassificationModel(
                model_type='roberta',
                model_name=f'{target_dir}/models/CARDS_RoBERTa_Classifier',
                use_cuda=device == 0
            )
            self._classifier = model

    def classify(self, texts: list[str], return_all_scores: bool = False):
        labels, scores = self._classifier.predict(texts)
        scores = [softmax(score[0]) for score in scores]
        # scores = [score[0] for score in scores]
        if return_all_scores:
            return [{label: scores_i[label_i] for label_i, label in self.id2label.items()} for scores_i in scores]
        return [{self.id2label[label_i]: scores_i[label_i]} for label_i, scores_i in zip(labels, scores)]


@dataclass
class Embedder:
    hf_name: str
    kind: Literal['transformer', 'auto']


ClassifierLiteral = Literal['cardiff-sentiment', 'cardiff-emotion', 'cardiff-offensive', 'cardiff-stance-climate',
                            'geomotions-orig', 'geomotions-ekman', 'nlptown-sentiment', 'bertweet-sentiment',
                            'bertweet-emotions', 'bert-sst2', 'cards']

# to find more models, browse this page:
# https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads
# Hint: the search function doesn't really work...
CLASSIFIERS = {
    # https://github.com/cardiffnlp/tweeteval/blob/main/datasets/sentiment/mapping.txt
    'cardiff-sentiment': Classifier(hf_name='cardiffnlp/twitter-roberta-base-sentiment',
                                    labels=['negative', 'neutral', 'positive']),

    # https://github.com/cardiffnlp/tweeteval/blob/main/datasets/emotion/mapping.txt
    'cardiff-emotion': Classifier(hf_name='cardiffnlp/twitter-roberta-base-emotion',
                                  labels=['anger', 'joy', 'optimism', 'sadness']),

    # https://github.com/cardiffnlp/tweeteval/blob/main/datasets/offensive/mapping.txt
    'cardiff-offensive': Classifier(hf_name='cardiffnlp/twitter-roberta-base-offensive',
                                    labels=['not-offensive', 'offensive']),

    # https://github.com/cardiffnlp/tweeteval/blob/main/datasets/stance/mapping.txt
    'cardiff-stance-climate': Classifier(hf_name='cardiffnlp/twitter-roberta-base-stance-climate',
                                         labels=['none', 'against', 'favor']),

    # https://huggingface.co/monologg/bert-base-cased-goemotions-original/blob/main/config.json
    'geomotions-orig': Classifier(hf_name='monologg/bert-base-cased-goemotions-original',
                                  labels=[
                                      'admiration',
                                      'amusement',
                                      'anger',
                                      'annoyance',
                                      'approval',
                                      'caring',
                                      'confusion',
                                      'curiosity',
                                      'desire',
                                      'disappointment',
                                      'disapproval',
                                      'disgust',
                                      'embarrassment',
                                      'excitement',
                                      'fear',
                                      'gratitude',
                                      'grief',
                                      'joy',
                                      'love',
                                      'nervousness',
                                      'neutral',
                                      'optimism',
                                      'pride',
                                      'realization',
                                      'relief',
                                      'remorse',
                                      'sadness',
                                      'surprise',
                                  ]),

    # https://huggingface.co/monologg/bert-base-cased-goemotions-ekman/blob/main/config.json
    'geomotions-ekman': Classifier(hf_name='monologg/bert-base-cased-goemotions-ekman',
                                   labels=['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']),

    # https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment/blob/main/config.json
    'nlptown-sentiment': Classifier(hf_name='nlptown/bert-base-multilingual-uncased-sentiment',
                                    labels=['1 star', '2 stars', '3 stars', '4 stars', '5 stars']),

    # https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis
    'bertweet-sentiment': Classifier(hf_name='finiteautomata/bertweet-base-sentiment-analysis',
                                     labels=['negative', 'neutral', 'positive']),

    # https://huggingface.co/finiteautomata/bertweet-base-emotion-analysis
    'bertweet-emotions': Classifier(hf_name='finiteautomata/bertweet-base-emotion-analysis',
                                    labels=['others', 'joy', 'sadness', 'anger', 'surprise', 'disgust', 'fear']),
    # https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/blob/main/config.json
    'bert-sst2': Classifier(hf_name='distilbert-base-uncased-finetuned-sst-2-english',
                            labels=['negative', 'positive']),
    'cards': CARDSClassifier(hf_name='CARDS',
                             labels=['0_0',  # 0 - no claim
                                     # global warming is not happening
                                     '1_1',  # 1 - ice isn't melting
                                     '1_2',  # 2 - heading into ice age
                                     '1_3',  # 3 - weather is cold
                                     '1_4',  # 4 - hiatus in warming
                                     '1_5',  # 5 - oceans are cooling
                                     '1_6',  # 6 - sea level rise is exaggerated
                                     '1_7',  # 7 - extremes aren't increasing
                                     # '1_8',  # 8 - changed the name
                                     # human greenhouse gases are not causing global warming
                                     '2_1',  # 9 - it's natural cycles
                                     # '2_2',  # 10 - non-ghg forcings
                                     # '2_3',  # 11 - no evidence of greenhouse effect
                                     # '2_4',  # 12 - co2 not rising
                                     # '2_5',  # 13 - emissions not raising co2 levels
                                     # climate impacts are not bad
                                     '3_1',  # 14 - sensitivity is low
                                     '3_2',  # 15 - no species impact
                                     '3_3',  # 16 - not a pollutant
                                     # '3_4',  # 17 - only a few degrees
                                     # '3_5',  # 18 - no link to conflict
                                     # '3_6',  # 19 - no health impacts
                                     # climate solutions won't work
                                     '4_1',  # 20 - policies are harmful
                                     '4_2',  # 21 - policies are ineffective
                                     # '4_3',  # 22 - too hard
                                     '4_4',  # 23 - clean energy won't work
                                     '4_5',  # 24 - we need energy
                                     # climate movement / science is unreliable
                                     '5_1',  # 25 - science is unreliable
                                     '5_2',  # 26 - movement is unreliable
                                     # '5_3'  # 27 - climate is conspiracy
                                     ])
}
EMBEDDERS = {
    'bertweet': Embedder(hf_name='vinai/bertweet-large', kind='transformer'),
    'minilm': Embedder(hf_name='paraphrase-multilingual-MiniLM-L12-v2', kind='transformer')
}


class BaseEmbedder(ABC):
    def __init__(self, embedding_model: str = None):
        self.embedding_model = embedding_model

    @abstractmethod
    def embed(self,
              documents: list[str],
              verbose: bool = False) -> np.ndarray:
        raise NotImplementedError()

    def embed_words(self,
                    words: list[str],
                    verbose: bool = False) -> np.ndarray:
        return self.embed(words, verbose)

    def embed_documents(self,
                        documents: list[str],
                        verbose: bool = False) -> np.ndarray:
        return self.embed(documents, verbose)


class AutoModelBackend(BaseEmbedder):
    def __init__(self, embedding_model: str, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model, use_fast=False)
        self.embedding_model = AutoModel.from_pretrained(embedding_model)

    def embed(self,
              documents: list[str],
              verbose: bool = False) -> np.ndarray:
        tokenised = torch.tensor([self.tokenizer.encode(d) for d in documents])
        embeddings = self.embedding_model.encode(tokenised, show_progress_bar=verbose)
        return embeddings


class SentenceTransformerBackend(BaseEmbedder):
    def __init__(self, embedding_model: str, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(embedding_model)

    def embed(self,
              documents: list[str],
              verbose: bool = False) -> np.ndarray:
        embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
        return embeddings


class ModelCache:
    """Class that controls the model caching process.

    Args:
        cache_dir (Path): The directory where the models should be cached to
    """

    cache_dir: Path

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir)

    def get_model_path(self, model_name: str) -> Path:
        return self.cache_dir / self.to_safe_name(model_name)

    def is_cached(self, model_name: str) -> bool:
        return self.get_model_path(model_name).exists()

    @staticmethod
    def to_safe_name(name: str):
        return re.sub(r'[^A-Za-z0-9]', '_', name)

    def cache_model(self, model_name: str):
        if model_name in CLASSIFIERS:
            self.cache_classifier(model_name)
        elif model_name in EMBEDDERS:
            self.cache_embedding_model(model_name)
        else:
            raise KeyError(f'Unknown model: {model_name}')

    def cache_embedding_model(self, model_name: str):
        print(f'Checking for {model_name} in {self.cache_dir}')
        if not self.is_cached(model_name):
            real_model_name = EMBEDDERS[model_name].hf_name
            model_cache_path = str(self.get_model_path(model_name))
            print(f'Downloading and caching {model_name} ({real_model_name} at {model_cache_path})')
            os.makedirs(model_cache_path, exist_ok=True)
            if EMBEDDERS[model_name].kind == 'transformer':
                pretrained_model = SentenceTransformer(real_model_name)
                pretrained_model.save(model_cache_path)
            else:
                pretrained_model = AutoModel.from_pretrained(real_model_name)
                pretrained_model.save_pretrained(model_cache_path)

                tokenizer = AutoTokenizer.from_pretrained(real_model_name)
                tokenizer.save_pretrained(model_cache_path)
        else:
            print(f'Already cached {model_name}')

    def cache_classifier(self, model_name: str):
        print(f'Checking for {model_name} in {self.cache_dir}')
        if not self.is_cached(model_name):
            model_cache_path = self.get_model_path(model_name)
            print(f'Downloading and caching {model_name} ({CLASSIFIERS[model_name].hf_name} at {model_cache_path})')
            CLASSIFIERS[model_name].store(model_cache_path)
        else:
            print(f'Already cached {model_name}')

    def cache_all_classifiers(self):
        for model_name in CLASSIFIERS.keys():
            self.cache_classifier(model_name)

    def cache_all_embeddings(self):
        for model_name in EMBEDDERS.keys():
            self.cache_embedding_model(model_name)

    def cache_all_models(self):
        self.cache_all_classifiers()
        self.cache_all_embeddings()

    @staticmethod
    def _set_cuda_settings():
        if torch.cuda.is_available():
            # Tell PyTorch to use the GPU.
            torch.device("cuda")
            use_cuda = True
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print(f'We will use GPU {torch.cuda.current_device()}: '
                  f'{torch.cuda.get_device_name(torch.cuda.current_device())}')
        else:
            print('No GPU available, using the CPU instead.')
            torch.device("cpu")
            use_cuda = False
        return use_cuda

    @staticmethod
    def _get_classifier_instance(model_name: str) -> Classifier:
        if model_name not in CLASSIFIERS:
            raise KeyError(f'Classifier {model_name} unknown.')
        return CLASSIFIERS[model_name]

    def get_classifier(self, model_name: str) -> Classifier:
        # ensure the model is downloaded
        self.cache_classifier(model_name)

        classifier = self._get_classifier_instance(model_name)
        model_cache_path = self.get_model_path(model_name)
        use_cuda = self._set_cuda_settings()
        device = 0 if use_cuda else -1
        classifier.load(model_cache_path, device=device)
        return classifier

    def get_embedder(self, model_name: str) -> Union[SentenceTransformerBackend, AutoModelBackend]:
        self.cache_embedding_model(model_name)
        model_cache_path = str(self.get_model_path(model_name))
        if EMBEDDERS[model_name].kind == 'transformer':
            return SentenceTransformerBackend(model_cache_path, model_name)
        else:
            return AutoModelBackend(model_cache_path, model_name)

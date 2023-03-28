import json
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
from utils.tweets import get_hashtags, get_urls, clean_tweet
from dataclasses import dataclass, field
import plotly.graph_objects as go
from colorcet import glasbey
import os

MODELS = {
    'cardiff-sentiment': ['negative', 'neutral', 'positive'],
    'cardiff-emotion': ['anger', 'joy', 'optimism', 'sadness'],
    'cardiff-offensive': ['not-offensive', 'offensive'],
    'cardiff-stance-climate': ['none', 'against', 'favor'],
    'geomotions-orig': [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire',
        'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'neutral', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise',
    ],
    'geomotions-ekman': ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'],
    'nlptown-sentiment': ['1 star', '2 stars', '3 stars', '4 stars', '5 stars'],
    'bertweet-sentiment': ['negative', 'neutral', 'positive'],
    'bertweet-emotions': ['others', 'joy', 'sadness', 'anger', 'surprise', 'disgust', 'fear'],
    'bert-sst2': ['negative', 'positive']
}


def get_empty_stats() -> dict[str, dict[str, int]]:
    return {
        model: {label: 0 for label in labels}
        for model, labels in MODELS.items()
    }


@dataclass
class Group:
    n_hashtags: list[int] = field(default_factory=list)
    n_urls: list[int] = field(default_factory=list)
    n_tokens: list[int] = field(default_factory=list)
    n_chars_clean: list[int] = field(default_factory=list)
    n_chars: list[int] = field(default_factory=list)
    stats: dict[str, dict[str, int]] = field(default_factory=get_empty_stats)


def plot_stacked_area(groups_: dict[str, Group], model):
    x = list(groups_.keys())
    fig = go.Figure()
    for i, label in enumerate(MODELS[model]):
        y = [g.stats[model][label] / (sum(g.stats[model].values()) + 0.00000001) for g in groups_.values()]

        fig.add_trace(go.Scatter(
            x=x, y=y,
            hoverinfo='x+y',
            mode='lines',
            name=label,
            line=dict(width=0.5, color=glasbey[i]),
            stackgroup='one'  # define stack group
        ))

    fig.update_layout(yaxis_range=(0, 1))
    # fig.show()
    return fig


FORMATS = {'yearly': '%Y', 'monthly': '%Y-%m', 'weekly': '%Y-%W', 'daily': '%Y-%m-%d'}
SELECTED_FORMAT = 'monthly'
FORMAT = FORMATS[SELECTED_FORMAT]

# DATASET = 'geoengineering'
DATASET = 'climate'
LIMIT = 10000
SOURCE_FILE = f'data/{DATASET}/tweets_sentiment_{LIMIT}.jsonl'
TARGET_FOLDER = f'data/{DATASET}/sentiments'

with open(SOURCE_FILE) as f:
    groups = {}

    for line in tqdm(f):
        tweet = json.loads(line)
        timestamp = datetime.strptime(tweet['created_at'][:19], '%Y-%m-%dT%H:%M:%S')
        group = timestamp.strftime(FORMAT)
        clean_txt = clean_tweet(tweet['text'],
                                remove_hashtags=True, remove_urls=True,
                                remove_nonals=True, remove_mentions=True)

        if group not in groups:
            groups[group] = Group()

        groups[group].n_tokens.append(len(clean_txt.split(' ')))
        groups[group].n_chars_clean.append(len(clean_txt))
        groups[group].n_chars.append(len(tweet['text']))
        groups[group].n_urls.append(len(get_urls(tweet['text'])))
        groups[group].n_hashtags.append(len(get_hashtags(tweet['text'])))

        for k, v in tweet['sentiments'].items():
            groups[group].stats[k][v[0][0]] += 1

    for model in MODELS.keys():
        fig_ = plot_stacked_area(groups, model)
        filename = f'{TARGET_FOLDER}/{SELECTED_FORMAT}_{model}.png'
        os.makedirs(TARGET_FOLDER, exist_ok=True)
        fig_.write_image(filename)

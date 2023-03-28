import numpy as np
from .frankentopic import FrankenTopic, TopicListing
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from colorcet import glasbey
from datetime import datetime
from typing import Literal, List

FORMATS = {'yearly': '%Y', 'monthly': '%Y-%m', 'weekly': '%Y-%W', 'daily': '%Y-%m-%d'}


def date2group(dt, fmt):
    timestamp = datetime.strptime(dt[:19], '%Y-%m-%dT%H:%M:%S')
    return timestamp.strftime(FORMATS[fmt])


def get_tokens(listing: TopicListing, topic: int, n_tokens: int):
    return [w[0] for w in listing[topic]][:n_tokens]


def simple_topic_listing(topics_tfidf: TopicListing, topics_mmr: TopicListing, n_tokens: int, labels: np.ndarray):
    topic_ids, topic_sizes = np.unique(labels, return_counts=True)
    for i, (c_i, c_cnt) in enumerate(zip(topic_ids, topic_sizes)):
        print(f'>> Topic {c_i} with {c_cnt} Tweets:')
        print(f' - tfidf: {",".join(get_tokens(topics_tfidf, i, n_tokens))}')
        print(f' - mmr:   {",".join(get_tokens(topics_mmr, i, n_tokens))}')
        print('---')


def get_temporal_distribution(tweets: List[dict],
                              labels: np.ndarray,
                              date_format: Literal['monthly', 'yearly', 'weekly', 'daily'],
                              boost: list[Literal['retweets', 'likes', 'replies']] = None,
                              skip_topic_zero=False):
    topic_ids = np.unique(labels, return_counts=False)
    groups = {}
    for tw, to in zip(tweets, labels):
        if (not skip_topic_zero) or (skip_topic_zero and to != 0):
            group = date2group(tw['created_at'], date_format)

            if group not in groups:
                groups[group] = np.zeros_like(topic_ids)

            groups[group][to] += 1
            if boost is not None:
                boost_map = {
                    'retweets': 'retweets_count',
                    'likes': 'favorites_count',
                    'replies': 'replies_count'
                }
                for b in boost:
                    groups[group][to] += tw.get(boost_map[b], 0)

    return groups


def write_normed_topic_distribution(temporal_topics,
                                    target_html_hist: str,
                                    target_html_heat: str,
                                    target_json: str,
                                    n_tokens: int,
                                    topic_listing: TopicListing):
    time_groups = sorted(temporal_topics.keys())

    topic_dist = np.array([temporal_topics[tg].tolist() for tg in time_groups])

    fig = go.Figure([go.Bar(x=[f'd:{d}' for d in time_groups],
                            y=topic_dist.sum(axis=0))])
    fig.write_html(f'{TARGET_DIR}/histogram_{LIMIT}_{DATE_FORMAT}_{"_".join(boost)}.html')

    if norm == 'row':
        topic_dist = topic_dist / (topic_dist.sum(axis=0) + 0.0000001)
    elif norm == 'col':
        topic_dist = (topic_dist.T / (topic_dist.sum(axis=1) + 0.0000001)).T
    # elif norm == 'abs':
    #     pass

    fig = go.Figure(data=go.Heatmap(
        z=topic_dist.T,
        x=[f'd:{d}' for d in time_groups],
        y=[', '.join(topic_model_utils.get_keywords(t, keyword_source='mmr', n_keywords=4))
           for t in topic_model_utils.topic_ids[1:]],
        hoverongaps=False))
    fig.write_html(f'{TARGET_DIR}/temporal_{LIMIT}_{DATE_FORMAT}_{norm}_{"_".join(boost)}.html')

    with open(f'{TARGET_DIR}/temporal/tt_{LIMIT}_{DATE_FORMAT}_{norm}_{"_".join(boost)}.json', 'w') as f:
        f.write(json.dumps({
            'z': topic_dist.T.tolist(),
            'x': time_groups,
            'y': [', '.join(topic_model_utils.get_keywords(t, keyword_source='mmr', n_keywords=4))
                  for t in topic_model_utils.topic_ids[1:]],
        }))

    for ti, vals in enumerate(topic_dist.T):
        output['topics'][ti][f'{norm}_{"_".join(boost or ["raw"])}'] = vals.tolist()


class FrankenTopicUtils:
    def __init__(self,
                 tweets: list[dict],
                 topic_model: FrankenTopic,
                 n_tokens_per_topic: int = 20):
        self.tweets = tweets
        self.topic_model = topic_model
        self.n_tokens_per_topic = n_tokens_per_topic

        self.tfidf_topics_ = None
        self.mmr_topics_ = None

        self.topic_ids, self.topic_sizes = np.unique(topic_model.labels, return_counts=True)
        self.num_topics = len(self.topic_ids)

    @property
    def tfidf_topics(self):
        if self.tfidf_topics_ is None:
            self.tfidf_topics_ = self.topic_model.get_top_n_tfidf(self.n_tokens_per_topic)
        return self.tfidf_topics_

    @property
    def mmr_topics(self):
        if self.mmr_topics_ is None:
            self.mmr_topics_ = self.topic_model.get_top_n_mmr(self.n_tokens_per_topic)
        return self.mmr_topics_

    def list_topics(self, emotions_model: str = None, emotions_keys: list[str] = None, include_mmr=True):
        for i, (c_i, c_cnt) in enumerate(zip(self.topic_ids, self.topic_sizes)):
            print(f'>> Topic {c_i} with {c_cnt} Tweets:')
            if emotions_model is not None and emotions_keys is not None:
                topic_sentiments = [self.tweets[idx]['sentiments'][emotions_model][0][0]
                                    for idx in np.where(self.topic_model.labels == c_i)[0]]
                topic_sentiments_cnt = Counter(topic_sentiments)
                print('sentiment:', ' '.join([
                    f'{l}: {topic_sentiments_cnt.get(l, 0)} '
                    f'({(topic_sentiments_cnt.get(l, 0) / sum(topic_sentiments_cnt.values())) * 100:.1f}%)'
                    for l in emotions_keys]))

            print('tfidf:', [w[0] for w in self.tfidf_topics[i]])
            if include_mmr:
                print('mmr:', [w[0] for w in self.mmr_topics[i]])
            print('---')

    def get_keywords(self,
                     topic: int,
                     n_keywords: int,
                     keyword_source: Literal['mmr', 'tfidf'] = 'mmr'):
        if keyword_source == 'mmr':
            return get_tokens(self.mmr_topics, topic, n_keywords)
        return get_tokens(self.tfidf_topics, topic, n_keywords)

    def landscape(self,
                  emotions_model: str = None,
                  emotions_keys: list[str] = None,
                  emotions_colours: list[str] = None,
                  keyword_source: Literal['mmr', 'tfidf'] = 'mmr',
                  n_keywords_per_topic_legend: int = 3,
                  n_keywords_per_topic_map: int = 6,
                  include_text: bool = True,
                  colormap=glasbey):

        fig = go.Figure()

        # draw emotions heatmap if required
        if emotions_model is not None and emotions_keys is not None and emotions_colours is not None:
            for emo_key, emo_col in zip(emotions_keys, emotions_colours):
                relevant = [i for i, t in enumerate(self.tweets) if t['sentiments'][emotions_model][0][0] == emo_key]
                pts = self.topic_model.layout[relevant]

                # Type: enumerated , one of ( "fill" | "heatmap" | "lines" | "none" )
                # Determines the coloring method showing the contour values. If "fill", coloring is done evenly between each
                # contour level If "heatmap", a heatmap gradient coloring is applied between each contour level. If "lines",
                # coloring is done on the contour lines. If "none", no coloring is applied on this trace.
                fig.add_trace(go.Histogram2dContour(x=pts[:, 0], y=pts[:, 1], colorscale=emo_col,
                                                    contours={'coloring': 'heatmap', 'start': 10},
                                                    opacity=0.6,
                                                    showscale=False))

        for topic in self.topic_ids:
            pts = self.topic_model.layout[self.topic_model.labels == topic]

            texts = [
                self.tweets[idx]['text']
                for idx in np.argwhere(self.topic_model.labels == topic).reshape(-1, )
            ]
            legend_keywords = self.get_keywords(topic, n_keywords_per_topic_legend, keyword_source=keyword_source)
            fig.add_trace(go.Scatter(
                x=pts[:, 0], y=pts[:, 1],
                name=f'Topic {topic}:<br> {", ".join(legend_keywords)}',
                marker_color=colormap[topic],
                opacity=0.5,
                text=texts
            ))

            if include_text:
                positions = pts[np.random.choice(pts.shape[0],
                                                 size=min(n_keywords_per_topic_map, pts.shape[0]),
                                                 replace=False)]
                keywords_map = self.get_keywords(topic, n_keywords_per_topic_map, keyword_source)
                for tw, pos in zip(keywords_map[:n_keywords_per_topic_map], positions):
                    fig.add_annotation(x=pos[0], y=pos[1], text=tw, showarrow=False,
                                       font={
                                           'family': "sans-serif",
                                           'color': "#000000",
                                           'size': 14
                                       })

        fig.update_traces(mode='markers', marker_line_width=1, marker_size=3, selector={'type': 'scatter'})
        fig.update_layout(title='Styled Scatter',
                          yaxis_zeroline=False, xaxis_zeroline=False)

        return fig

    def get_temporal_distribution(self,
                                  date_format: Literal['monthly', 'yearly', 'weekly', 'daily'],
                                  boost: list[Literal['retweets', 'likes', 'replies']] = None,
                                  skip_topic_zero=False):

        return get_temporal_distribution(tweets=self.tweets, labels=self.topic_model.labels, date_format=date_format,
                                         boost=boost, skip_topic_zero=skip_topic_zero)

    def temporal_stacked_fig(self,
                             date_format: Literal['monthly', 'yearly', 'weekly', 'daily'],
                             n_keywords_per_topic: int = 5,
                             keyword_source: Literal['mmr', 'tfidf'] = 'mmr',
                             colorscheme=glasbey):

        groups = self.get_temporal_distribution(date_format=date_format, boost=None, skip_topic_zero=True)

        fig = go.Figure()
        x = list(groups.keys())
        for topic in np.unique(self.topic_model.labels):
            top_words = self.get_keywords(topic, keyword_source=keyword_source, n_keywords=n_keywords_per_topic)
            y = [g[topic] / g.sum() for g in groups.values()]

            fig.add_trace(go.Scatter(
                x=x, y=y,
                hoverinfo='x+y',
                mode='lines',
                name=f'Topic {topic}:<br> {", ".join(top_words)}',
                line={'width': 0.5, 'color': colorscheme[topic]},
                stackgroup='one'  # define stack group
            ))

        fig.update_layout(yaxis_range=(0, 1))
        return fig

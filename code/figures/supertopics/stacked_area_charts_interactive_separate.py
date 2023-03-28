import os

from scripts.util import read_supertopics, SuperTopic, get_spottopics, DateFormat, read_temp_dist, smooth
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go

BOOST = ['raw',  # 0
         'retweets',  # 1
         'replies',  # 2
         'likes',  # 3
         'retweets_likes',  # 4
         'replies_likes',  # 5
         'retweets_replies',  # 6
         'retweets_likes_replies'  # 7
         ][0]

FILE_SUPERTOPICS = f'data/climate2/topics_big2/supertopics.csv'

FILES_TEMP_DIST = {
    'keep (majority)': f'data/climate2/topics_big2/temporal_keep_majority/daily/temporal_daily_{BOOST}_abs.json',
    'fresh (majority)': f'data/climate2/topics_big2/temporal_fresh_majority/daily/temporal_daily_{BOOST}_abs.json'
}
FILE_TEMP_DIST = FILES_TEMP_DIST[['keep (majority)', 'fresh (majority)'][0]]
BOUNDARY = '2020-03-01'
SMOOTHING = 90
EPS = 1e-12
annotations = read_supertopics(FILE_SUPERTOPICS)

td_groups, td_topics, td_counts = read_temp_dist(FILE_TEMP_DIST)
supertopic_counts = []
st_summed_counts = []
st_topic_counts = []
for st in SuperTopic:
    t_counts = td_counts.T[annotations[:, st] > 0].sum(axis=0)
    supertopic_counts.append(t_counts)
    print(st.name, f'{t_counts.sum():,}')
    st_summed_counts.append(t_counts.sum())
    st_topic_counts.append(sum(annotations[:, st] > 0))

supertopic_counts = np.array(supertopic_counts)
BOUND = td_groups.index(BOUNDARY)
sts_plot = [SuperTopic.COVID, SuperTopic.Causes, SuperTopic.Impacts, SuperTopic.Solutions,
            SuperTopic.POLITICS, SuperTopic.Movements, SuperTopic.Contrarian,
            # SuperTopic.Other,  # SuperTopic.Interesting, SuperTopic.NotRelevant
            ]

tweets_per_day = np.sum(td_counts, axis=1)
tweets_per_topic = np.sum(td_counts, axis=0)
st_plot_counts = supertopic_counts[sts_plot]
st_plot_shares = st_plot_counts / tweets_per_day
st_plot_shares_smooth = smooth(st_plot_shares, kernel_size=SMOOTHING)

subplot_titles = [
    f'{st.name}: {sum(annotations[:, st] > 0):,} topics with {int(st_summed_counts[sti]):,} tweets'
    for sti, st in enumerate(sts_plot)
]
os.makedirs('data/climate2/figures/supertopic_shares_split/', exist_ok=True)
for i, st in enumerate(sts_plot, start=1):
    fig = go.Figure(layout={'title': {'text': subplot_titles[i - 1]}})
    n_st_tweets = td_counts.T[annotations[:, st] > 0].T
    n_st_tweets_per_day = n_st_tweets.sum(axis=1)
    subfig = []
    subfig_y = smooth(n_st_tweets.T / (n_st_tweets_per_day + EPS), kernel_size=SMOOTHING)

    topic_nums = np.arange(annotations.shape[0])[annotations[:, st] > 0]

    for ti, (y_, yt) in enumerate(zip(subfig_y, n_st_tweets.T)):
        fig.add_trace(go.Scatter(x=td_groups,
                                 y=y_,
                                 mode='lines',
                                 stackgroup='one',
                                 name=f'Topic {topic_nums[ti]} ({int(yt.sum()):,} tweets)'))

    fig.update_layout(height=1000, width=1000)
    fig.write_html(f'data/climate2/figures/supertopic_shares_split/supertopic_{st.name}.html')

os.makedirs('data/climate2/figures/supertopic_abs_split/', exist_ok=True)
for i, st in enumerate(sts_plot, start=1):
    fig = go.Figure(layout={'title': {'text': subplot_titles[i - 1]}})
    n_st_tweets = td_counts.T[annotations[:, st] > 0].T
    n_st_tweets_per_day = n_st_tweets.sum(axis=1)

    subfig_y = smooth(n_st_tweets.T, kernel_size=SMOOTHING)

    topic_nums = np.arange(annotations.shape[0])[annotations[:, st] > 0]

    for ti, (y_, yt) in enumerate(zip(subfig_y, n_st_tweets.T)):
        fig.add_trace(go.Scatter(x=td_groups,
                                 y=y_,
                                 mode='lines',
                                 stackgroup='one',
                                 name=f'Topic {topic_nums[ti]} ({int(yt.sum()):,} tweets)'))

    fig.update_layout(height=1000, width=1000)
    fig.write_html(f'data/climate2/figures/supertopic_abs_split/supertopic_{st.name}.html')

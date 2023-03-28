from scripts.util import read_supertopics, SuperTopic, get_spottopics, DateFormat, read_temp_dist
from typing import Literal, Optional
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
import seaborn as sns
import csv
from itertools import chain, repeat

DATASET = 'climate2'
LIMIT = 7000000
DATE_FORMAT: DateFormat = 'daily'
NORM: Literal['abs', 'col', 'row'] = 'abs'
BOOST = ['raw',  # 0
         # 'retweets',  # 1
         # 'replies',  # 2
         # 'likes',  # 3
         # 'retweets_likes',  # 4
         # 'replies_likes',  # 5
         # 'retweets_replies',  # 6
         'retweets_likes_replies'  # 7
         ][0]

FILE_CC_NEWS = 'data/world_dataset.xlsx'
FILE_SUPERTOPICS = f'data/{DATASET}/topics_big2/supertopics.csv'
FILES_COUNTS = {
    'en_tweets': ('data/climate2/english_tweet_counts_daily_2006-2021-rt.csv', 0, 1),
    'cc_tweets': ('data/climate2/cc_tweet_counts_daily_2006-2021-rt.csv', 4, 3),
    'cov_tweets': ('data/climate2/covid_tweet_counts_daily_2006-2021-rt_specific2.csv', 4, 3)
}

FILES_TEMP_DIST = {
    'sampled': f'data/{DATASET}/topics_big2/temporal_sampled/{DATE_FORMAT}/temporal_{LIMIT}_{DATE_FORMAT}_{BOOST}_{NORM}.json',
    'keep (majority)': f'data/{DATASET}/topics_big2/temporal_keep_majority/{DATE_FORMAT}/temporal_{DATE_FORMAT}_{BOOST}_{NORM}.json',
    # 'keep (proximity)': f'data/{DATASET}/topics_big2/temporal_keep_proximity/{DATE_FORMAT}/temporal_{DATE_FORMAT}_{BOOST}_{NORM}.json',
    'fresh (majority)': f'data/{DATASET}/topics_big2/temporal_fresh_majority/{DATE_FORMAT}/temporal_{DATE_FORMAT}_{BOOST}_{NORM}.json',
    # 'fresh (proximity)': f'data/{DATASET}/topics_big2/temporal_fresh_proximity/{DATE_FORMAT}/temporal_{DATE_FORMAT}_{BOOST}_{NORM}.json',
}
FIRST = '2018-01-01'
LAST = '2021-12-15'
BOUNDARY = '2020-03-01'
SMOOTHING = 30

annotations = read_supertopics(FILE_SUPERTOPICS)


def smooth(array, kernel_size=SMOOTHING):
    kernel = np.ones(kernel_size) / kernel_size
    return np.array([np.convolve(row, kernel, mode='same') for row in array])


def read_csv(args):
    global GROUPS
    filename, col_groups, col_counts = args
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader, None)
        cnts = {
            r[col_groups]: r[col_counts] for r in reader
        }

        if GROUPS is None:
            tweet_groups = sorted(cnts.keys())

            s = tweet_groups.index(FIRST)
            e = tweet_groups.index(LAST)

            GROUPS = tweet_groups[s:e]
        return np.array([int(cnts[k]) for k in GROUPS])


def groups2ticks(grps=None):
    if grps is None:
        grps = GROUPS
    xticks_ = []
    xticklabels_ = []
    for i, g in enumerate(grps):
        s = re.search(r'\d{4}-(\d{2})-(\d{2})', g)
        if int(s.group(1)) % 3 == 0 and int(s.group(2)) == 1:
            xticks_.append(i)
            xticklabels_.append(g)
    return np.arange(len(grps)), xticks_, xticklabels_


def read_cc_news():
    df = pd.read_excel(FILE_CC_NEWS)
    ret = {}
    for name, row in {'Total': 122, 'Oceania': 117, 'Africa': 108, 'CS America': 93,
                      'N America': 77, 'Europe': 68, 'Middle East': 34, 'Asia': 27}.items():
        ret[name] = df.iloc[row].to_list()[170: 218]
    grps = [
        f'{y}-{m:02d}-01'
        for y in [2018, 2019, 2020, 2021]
        for m in range(1, 13)
    ]
    return grps, ret


def read_st_counts(filename):
    td_groups, td_topics, td_counts = read_temp_dist(filename)
    supertopic_counts = []
    for st in SuperTopic:
        # number of tweets per day (only including topics belonging to supertopic)
        t_counts = td_counts.T[annotations[:, st] > 0].sum(axis=0)
        supertopic_counts.append(t_counts)
        print(st, f'{t_counts.sum():,}')
    supertopic_counts = np.array(supertopic_counts)
    mapped = {g: c for g, c in zip(td_groups, supertopic_counts.T)}
    if DATE_FORMAT == 'monthly':
        return np.array([mapped.get(g[:7], np.zeros(len(supertopic_counts))) for g in GROUPS]).T
    else:
        return np.array([mapped.get(g, np.zeros(len(supertopic_counts))) for g in GROUPS]).T


def draw_news(axis, region: Literal['Total', 'Oceania', 'Africa', 'CS America',
                                    'N America', 'Europe', 'Middle East', 'Asia']):
    groups, cnts = read_cc_news()
    xa, xt, xtl = groups2ticks(groups)
    bound = groups.index(BOUNDARY)
    axis.set_title('CC and GW News')

    y = np.array(cnts[region])
    threshold = y.mean()
    axis.axhline(threshold, color='black', lw=2, alpha=0.5)
    # fill the areas under the curve
    axis.fill_between(xa, threshold, y, where=y > threshold, color='green', alpha=0.5)
    axis.fill_between(xa, y, threshold, where=y < threshold, color='red', alpha=0.5)
    axis.plot(xa, y, color='black')
    axis.axvline(bound, color='black', lw=2, alpha=0.5)

    axis.set_xticks(xt)
    axis.set_xticklabels([tl[:7] for tl in xtl], rotation=45, fontsize=8)
    sns.regplot(x=xa[:bound], y=y[:bound], ax=axis, scatter=False)
    sns.regplot(x=xa[bound:], y=y[bound:], ax=axis, scatter=False)


def draw_data(axis, y):
    y_smooth = smooth([y], kernel_size=7).reshape(-1, )
    threshold = y.mean()
    axis.axhline(threshold, color='black', lw=2, alpha=0.5)
    # fill the areas under the curve
    axis.fill_between(x, threshold, y_smooth, where=y_smooth > threshold, color='green', alpha=0.5)
    axis.fill_between(x, y_smooth, threshold, where=y_smooth < threshold, color='red', alpha=0.5)

    axis.plot(x, y_smooth, color='black')
    axis.axvline(BOUND, color='black', lw=2, alpha=0.5)

    axis.set_xticks(xticks)
    axis.set_xticklabels([tl[:7] for tl in xticklabels], rotation=45, fontsize=8)

    axis.scatter(x, y, c='orange', s=0.4, marker=',')
    # plot regression line for period before and after pandemic
    sns.regplot(x=x[:BOUND], y=y[:BOUND], ax=axis, scatter=False)
    sns.regplot(x=x[BOUND:], y=y[BOUND:], ax=axis, scatter=False)

    # set the opacity of confidence interval
    plt.setp(axis.collections[3], alpha=0.5)
    plt.setp(axis.collections[4], alpha=0.5)

    # exclude extreme values by setting ylim
    # axis.set_ylim(np.percentile(y, q=1), np.percentile(y, q=99))


GROUPS: Optional[list[str]] = None
COUNTS = {k: read_csv(v) for k, v in FILES_COUNTS.items()}
TEMP_DISTS = {k: read_st_counts(v) for k, v in FILES_TEMP_DIST.items()}

x, xticks, xticklabels = groups2ticks()
BOUND = GROUPS.index(BOUNDARY)

REL = ['raw', 'mean', 'sum'][0]
sts_plot = [SuperTopic.COVID, SuperTopic.Causes, SuperTopic.Impacts, SuperTopic.Solutions,
            SuperTopic.POLITICS, SuperTopic.Movements, SuperTopic.Contrarian,
            # SuperTopic.Other,  # SuperTopic.Interesting, SuperTopic.NotRelevant
            ]
for d_title, distributions in TEMP_DISTS.items():
    fig = plt.figure(figsize=(10, 30), dpi=150)
    fig.suptitle(f'{d_title} | {BOOST} | {REL}', y=1)

    n_subplots = len(COUNTS) + len(sts_plot) + 1

    ax = plt.subplot(n_subplots, 1, 1)
    draw_news(ax, region='Total')

    for i, (title, counts) in enumerate(chain(COUNTS.items(),
                                              zip(sts_plot, repeat(distributions))), start=2):
        ax = plt.subplot(n_subplots, 1, i)
        ax.set_title(title.name if hasattr(title, 'name') else title)

        # plot the data
        if title not in COUNTS:
            ys = counts[title]
        else:
            ys = counts
        if title != 'en_tweets':
            smooth([ys], kernel_size=30).reshape(-1, )

        draw_data(axis=ax, y=ys)

    fig.tight_layout()
    plt.show()

#
# for base in FILE_TEMP_DIST_BASE:
#
#     groups, topics, counts = read_temp_dist(FILE_TEMP_DIST)
#     annotations = read_supertopics(FILE_SUPERTOPICS)
#
#     # Set up an array where each column is a supertopic and each row is a daily tweet count
#     supertopic_counts = []
#     for st in SuperTopic:
#         # number of tweets per day (only including topics belonging to supertopic)
#         t_counts = counts.T[annotations[:, st] > 0].sum(axis=0)
#         supertopic_counts.append(t_counts)
#         print(st, f'{t_counts.sum():,}')
#     supertopic_counts = np.array(supertopic_counts)
#     supertopic_counts_smooth = smooth(supertopic_counts)
#     totals_daily = supertopic_counts.sum(axis=0)
#     totals_daily_smooth = smooth(totals_daily)
#     totals_topics = supertopic_counts.sum(axis=1)
#
#     print('total (incl boost):', supertopic_counts.sum())
#     print('counts', counts.shape)
#     print('annos', annotations.shape)
#     print('st counts', supertopic_counts.shape)
#
#     sts_plot = [st for st in SuperTopic if
#                 st not in [SuperTopic.Interesting, SuperTopic.NotRelevant, SuperTopic.Other]]
#     labels = [st.name for st in sts_plot]
#     xticks = []
#     xticklabels = []
#     for i, x in enumerate(groups):
#         s = re.search(r'\d{4}-(\d{2})-(\d{2})', x)
#         if int(s.group(1)) % 3 == 0 and int(s.group(2)) == 1:
#             xticks.append(i)
#             xticklabels.append(x)
#
#     bound = groups.index('2020-01-01')
#     fig = plt.figure(figsize=(10, 20), dpi=150)
#     for i, st in enumerate(sts_plot, start=1):
#         ax = plt.subplot(len(sts_plot), 1, i)
#         ax.set_title(st.name)
#         ax.axvline(bound, color='black', lw=2, alpha=0.5)
#         # plot the data
#         x = np.arange(0, len(groups))
#         y = smooth([supertopic_counts[st]], kernel_size=7).reshape(-1, )
#
#         # ax.set_ylabel('Proportion of topic tweets')
#         # y = smooth([supertopic_counts[st] / supertopic_counts[st].sum()], kernel_size=30).reshape(-1, )
#
#         # ax.set_ylabel('Proportion from mean')
#         # y = smooth([supertopic_counts[st] / supertopic_counts[st].mean()], kernel_size=7).reshape(-1, )
#
#         ax.plot(x, y, color='black')
#
#         # plot the mean line (=1)
#         threshold = y.mean()
#         ax.axhline(threshold, color='black', lw=2, alpha=0.5)
#         # fill the areas under the curve
#         ax.fill_between(x, threshold, y, where=y > threshold, color='green', alpha=0.5)
#         ax.fill_between(x, y, threshold, where=y < threshold, color='red', alpha=0.5)
#         # show the mean of tweets before and after the pandemic
#         # ax.axhline(y[:bound].mean(), color='black', ls=':', lw=2, alpha=0.5, xmax=bound / len(groups))
#         # ax.axhline(y[bound:].mean(), color='black', ls=':', lw=2, alpha=0.5, xmin=bound / len(groups))
#         ax.axvline(bound)
#         ax.set_xticks(xticks)
#         ax.set_xticklabels([tl[:7] for tl in xticklabels], rotation=45, fontsize=8)
#         # ax.set_yscale('symlog')
#
#         ax2 = ax.twinx()
#         y = supertopic_counts[st] / supertopic_counts.sum(axis=0)
#         ax2.scatter(x, y, c='orange', s=0.4, marker=',')
#         # plot regression line for period before and after pandemic
#         sns.regplot(x=x[:bound], y=y[:bound], ax=ax2, scatter=False)
#         sns.regplot(x=x[bound:], y=y[bound:], ax=ax2, scatter=False)
#         # set the opacity of confidence interval
#         plt.setp(ax2.collections[1], alpha=0.3)
#         plt.setp(ax2.collections[2], alpha=0.3)
#         # exclude extreme values by setting ylim
#         ax2.set_ylim(np.percentile(y, q=1), np.percentile(y, q=99))
#         ax2.set_ylabel('Share of tweets')
#
#     fig.suptitle(base[35:50] + ' | ' + boost, y=1)

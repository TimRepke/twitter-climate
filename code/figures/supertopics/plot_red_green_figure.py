from scripts.util import read_supertopics, SuperTopic, get_spottopics, DateFormat, read_temp_dist
from typing import Literal
import numpy as np
from matplotlib import pyplot as plt
import re
import seaborn as sns

DATASET = 'climate2'
LIMIT = 7000000
DATE_FORMAT: DateFormat = 'daily'
NORM: Literal['abs', 'col', 'row'] = 'abs'
REGRESSION: Literal['abs', 'rel'] = 'rel'
BOOST = ['raw',  # 0
         # 'retweets',  # 1
         # 'replies',  # 2
         # 'likes',  # 3
         # 'retweets_likes',  # 4
         # 'replies_likes',  # 5
         # 'retweets_replies',  # 6
         # 'retweets_likes_replies'  # 7
         ]
SMOOTHING = 60
FILE_SUPERTOPICS = f'data/{DATASET}/topics_big2/supertopics.csv'
FILE_TEMP_DIST_BASE = [# f'data/{DATASET}/topics_big2/temporal_sampled/{DATE_FORMAT}/temporal_{LIMIT}',  # 0
                       # f'data/{DATASET}/topics_big2/temporal_fresh_majority/{DATE_FORMAT}/temporal',  # 1
                       # f'data/{DATASET}/topics_big2/temporal_fresh_proximity/{DATE_FORMAT}/temporal',  # 2
                       f'data/{DATASET}/topics_big2/temporal_keep_majority/{DATE_FORMAT}/temporal',  # 3
                       # f'data/{DATASET}/topics_big2/temporal_keep_proximity/{DATE_FORMAT}/temporal'  # 4
                       ]


def smooth(array, kernel_size=SMOOTHING):
    kernel = np.ones(kernel_size) / kernel_size
    return np.array([np.convolve(row, kernel, mode='same') for row in array])

for boost in BOOST:
    for base in FILE_TEMP_DIST_BASE:
        print(f'=== {boost} -> {base} ===')
        FILE_TEMP_DIST = f'{base}_{DATE_FORMAT}_{boost}_{NORM}.json'

        groups, topics, counts = read_temp_dist(FILE_TEMP_DIST)
        annotations = read_supertopics(FILE_SUPERTOPICS)

        # Set up an array where each column is a supertopic and each row is a daily tweet count
        supertopic_counts = []
        for st in SuperTopic:
            # number of tweets per day (only including topics belonging to supertopic)
            t_counts = counts.T[annotations[:, st] > 0].sum(axis=0)
            supertopic_counts.append(t_counts)
            print(st, f'{t_counts.sum():,}')
        supertopic_counts = np.array(supertopic_counts)
        supertopic_counts_smooth = smooth(supertopic_counts)
        totals_daily = supertopic_counts.sum(axis=0)
        totals_daily_smooth = smooth(totals_daily)
        totals_topics = supertopic_counts.sum(axis=1)

        print('total (incl boost):', supertopic_counts.sum())
        print('counts', counts.shape)
        print('annos', annotations.shape)
        print('st counts', supertopic_counts.shape)

        sts_plot = [st for st in SuperTopic if st not in [SuperTopic.Interesting, SuperTopic.NotRelevant, SuperTopic.Other]]
        labels = [st.name for st in sts_plot]
        xticks = []
        xticklabels = []
        for i, x in enumerate(groups):
            s = re.search(r'\d{4}-(\d{2})-(\d{2})', x)
            if int(s.group(1)) % 3 == 0 and int(s.group(2)) == 1:
                xticks.append(i)
                xticklabels.append(x)

        bound = groups.index('2020-03-01')
        fig = plt.figure(figsize=(8, 20), dpi=150)
        for i, st in enumerate(sts_plot, start=1):
            ax = plt.subplot(len(sts_plot), 1, i)
            ax.set_title(st.name)
            ax.axvline(bound, color='black', lw=2, alpha=0.5)
            # plot the data
            x = np.arange(0, len(groups))
            y = smooth([supertopic_counts[st]], kernel_size=SMOOTHING).reshape(-1, )

            # ax.set_ylabel('Proportion of topic tweets')
            # y = smooth([supertopic_counts[st] / supertopic_counts[st].sum()], kernel_size=30).reshape(-1, )

            # ax.set_ylabel('Proportion from mean')
            # y = smooth([supertopic_counts[st] / supertopic_counts[st].mean()], kernel_size=7).reshape(-1, )

            ax.plot(x, y, color='black')

            # plot the mean line (=1)
            threshold = y.mean()
            ax.axhline(threshold, color='black', lw=2, alpha=0.5)
            # fill the areas under the curve
            ax.fill_between(x, threshold, y, where=y > threshold, color='green', alpha=0.5)
            ax.fill_between(x, y, threshold, where=y < threshold, color='red', alpha=0.5)
            # show the mean of tweets before and after the pandemic
            # ax.axhline(y[:bound].mean(), color='black', ls=':', lw=2, alpha=0.5, xmax=bound / len(groups))
            # ax.axhline(y[bound:].mean(), color='black', ls=':', lw=2, alpha=0.5, xmin=bound / len(groups))
            ax.axvline(bound)
            ax.set_xticks(xticks)
            ax.set_xticklabels([tl[:7] for tl in xticklabels], rotation=45, fontsize=8)
            # ax.set_yscale('symlog')

            ax2 = ax.twinx()
            y_abs = supertopic_counts[st] / supertopic_counts.sum(axis=0)
            ax2.scatter(x, y_abs, c='orange', s=0.4, marker=',')
            # exclude extreme values by setting ylim
            ax2.set_ylim(np.percentile(y_abs, q=1), np.percentile(y_abs, q=99))
            ax2.set_ylabel('Share of tweets')

            if REGRESSION == 'abs':
                # plot regression line for period before and after pandemic
                sns.regplot(x=x[:bound], y=y_abs[:bound], ax=ax2, scatter=False)
                sns.regplot(x=x[bound:], y=y_abs[bound:], ax=ax2, scatter=False)
                # set the opacity of confidence interval
                plt.setp(ax2.collections[1], alpha=0.3)
                plt.setp(ax2.collections[2], alpha=0.3)
            else:
                # plot regression line for period before and after pandemic
                sns.regplot(x=x[:bound], y=supertopic_counts[st][:bound], ax=ax, scatter=False)
                sns.regplot(x=x[bound:], y=supertopic_counts[st][bound:], ax=ax, scatter=False)
                # set the opacity of confidence interval
                plt.setp(ax.collections[1], alpha=0.3)
                plt.setp(ax.collections[2], alpha=0.3)

        fig.suptitle(base[35:50] + ' | ' + boost, y=1)
        fig.tight_layout()
        plt.show()

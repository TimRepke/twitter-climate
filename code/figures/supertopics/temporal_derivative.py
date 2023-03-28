from scripts.util import read_supertopics, SuperTopic, get_spottopics, DateFormat, read_temp_dist
from typing import Literal
import numpy as np
from matplotlib import pyplot as plt
import json
from prettytable import PrettyTable
import re
import seaborn as sns

DATASET = 'climate2'
LIMIT = 7000000
DATE_FORMAT: DateFormat = 'daily'
NORM: Literal['abs', 'col', 'row'] = 'abs'
BOOST = ['raw',  # 0
         'retweets',  # 1
         'replies',  # 2
         'likes',  # 3
         'retweets_likes',  # 4
         'replies_likes',  # 5
         'retweets_replies',  # 6
         'retweets_likes_replies'  # 7
         ][0]
SMOOTHING = 30
FILE_SUPERTOPICS = f'data/{DATASET}/topics_big2/supertopics.csv'
FILE_TEMP_DIST_BASE = [f'data/{DATASET}/topics_big2/temporal_sampled/{DATE_FORMAT}/temporal_{LIMIT}',
                       f'data/{DATASET}/topics_big2/temporal_keep_majority/{DATE_FORMAT}/temporal'][1]
FILE_TEMP_DIST = f'{FILE_TEMP_DIST_BASE}_{DATE_FORMAT}_{BOOST}_{NORM}.json'

groups, topics, counts = read_temp_dist(FILE_TEMP_DIST)
annotations = read_supertopics(FILE_SUPERTOPICS)
fi = 0


def smooth(array, kernel_size=SMOOTHING):
    kernel = np.ones(kernel_size) / kernel_size
    return np.array([np.convolve(row, kernel, mode='same') for row in array])


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

fig, ax = plt.subplots(dpi=150)
ax.plot(supertopic_counts[sts_plot].T, label=labels)
ax.set_title('Number of tweets (abs, raw)')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
ax.legend(bbox_to_anchor=(1.3, 1.05), fontsize=8)
fig.tight_layout()
plt.savefig(f'data/climate2/figures/temporal_derivative/fig_{fi}.png')
plt.show()
fi += 1

fig, ax = plt.subplots(dpi=150)
ax.plot(smooth(supertopic_counts[sts_plot]).T, label=labels)
ax.set_title('Number of tweets (abs, smoothed)')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
ax.legend(bbox_to_anchor=(1.3, 1.05), fontsize=8)
fig.tight_layout()
plt.savefig(f'data/climate2/figures/temporal_derivative/fig_{fi}.png')
plt.show()
fi += 1

fig, ax = plt.subplots(dpi=150)
ax.plot(smooth(supertopic_counts / supertopic_counts.sum(axis=0))[sts_plot].T, label=labels)
ax.set_title('Number of tweets (normed per day, all)')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
ax.legend(bbox_to_anchor=(1.3, 1.05), fontsize=8)
fig.tight_layout()
plt.savefig(f'data/climate2/figures/temporal_derivative/fig_{fi}.png')
plt.show()
fi += 1

fig, ax = plt.subplots(dpi=150)
ax.plot(smooth(supertopic_counts / supertopic_counts[sts_plot].sum(axis=0))[sts_plot].T, label=labels)
ax.set_title('Number of tweets (normed per day; excl other, non-relevant, interesting)')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
ax.legend(bbox_to_anchor=(1.3, 1.05), fontsize=8)
fig.tight_layout()
plt.savefig(f'data/climate2/figures/temporal_derivative/fig_{fi}.png')
plt.show()
fi += 1

fig, ax = plt.subplots(dpi=150)
ax.plot(smooth(supertopic_counts / supertopic_counts.sum(axis=1)[:, None])[sts_plot].T, label=labels)
ax.set_title('Number of tweets (normed per supertopic)')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
ax.legend(bbox_to_anchor=(1.3, 1.05), fontsize=8)
fig.tight_layout()
plt.savefig(f'data/climate2/figures/temporal_derivative/fig_{fi}.png')
plt.show()
fi += 1

fig = plt.figure(figsize=(10, 20), dpi=150)
for i, st in enumerate(sts_plot, start=1):
    ax = plt.subplot(len(sts_plot), 1, i)
    ax.set_title(st.name)
    x = np.arange(0, len(groups))
    y = smooth(supertopic_counts / supertopic_counts[sts_plot].sum(axis=0))[st].T
    ax.plot(x, y, color='black')
    threshold = y.mean()
    ax.axhline(threshold, color='green', lw=2, alpha=0.5)
    ax.fill_between(x, 0, 1, where=y > threshold, color='green', alpha=0.5, transform=ax.get_xaxis_transform())
    ax.fill_between(x, 0, 1, where=y < threshold, color='red', alpha=0.5, transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 0.4)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
fig.tight_layout()
plt.savefig(f'data/climate2/figures/temporal_derivative/fig_{fi}.png')
plt.show()
fi += 1

means = np.mean(supertopic_counts, axis=1)
pre_post_curves_cum_smooth = []
pre_post_curves_cum_raw = []
pre_post_curves_auc_ratio = []
pre_post_curves_auc_sum = []
pre_post_curves_auc_cum = []
pre_post_curves_mean = []
for threshold in range(len(groups)):
    pre_post_curves_cum_smooth.append(
        supertopic_counts_smooth[:, :threshold].sum(axis=1) /
        supertopic_counts_smooth.sum(axis=1))
    pre_post_curves_cum_raw.append(
        supertopic_counts[:, :threshold].sum(axis=1) /
        supertopic_counts.sum(axis=1))
    pre_post_curves_auc_ratio.append(
        np.abs(supertopic_counts - means[:, None])[:, :threshold].sum(axis=1) /
        np.abs(supertopic_counts - means[:, None]).sum(axis=1)
    )
    pre_post_curves_auc_sum.append((supertopic_counts - means[:, None])[:, :threshold].sum(axis=1))
    pre_post_curves_auc_cum.append(np.abs(supertopic_counts - means[:, None])[:, :threshold].sum(axis=1))

pre_post_curves_cum_raw = np.array(pre_post_curves_cum_raw)
pre_post_curves_cum_smooth = np.array(pre_post_curves_cum_smooth)
pre_post_curves_auc_ratio = np.array(pre_post_curves_auc_ratio)
pre_post_curves_auc_sum = np.array(pre_post_curves_auc_sum)
pre_post_curves_auc_cum = np.array(pre_post_curves_auc_cum)

fig, ax = plt.subplots(dpi=150)
ax.plot(pre_post_curves_auc_ratio[:, sts_plot], label=labels)
ax.set_title('Ratio of abs AUC')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
ax.legend(bbox_to_anchor=(1.3, 1.05), fontsize=8)
fig.tight_layout()
plt.savefig(f'data/climate2/figures/temporal_derivative/fig_{fi}.png')
plt.show()
fi += 1

fig, ax = plt.subplots(dpi=150)
ax.plot(pre_post_curves_auc_sum[:, sts_plot], label=labels)
ax.set_title('Sum of AUC up to a date')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
ax.legend(bbox_to_anchor=(1.3, 1.05), fontsize=8)
fig.tight_layout()
plt.savefig(f'data/climate2/figures/temporal_derivative/fig_{fi}.png')
plt.show()
fi += 1

fig, ax = plt.subplots(dpi=150)
ax.plot(pre_post_curves_auc_cum[:, sts_plot], label=labels)
ax.set_title('Sum of abs. AUC up to a date')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
ax.legend(bbox_to_anchor=(1.3, 1.05), fontsize=8)
fig.tight_layout()
plt.savefig(f'data/climate2/figures/temporal_derivative/fig_{fi}.png')
plt.show()
fi += 1

fig, ax = plt.subplots(dpi=150)
ax.plot((pre_post_curves_auc_cum / pre_post_curves_auc_cum[-1])[:, sts_plot], label=labels)
ax.set_title('Sum of abs. AUC up to a date (normed by ST)')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
ax.legend(bbox_to_anchor=(1.3, 1.05), fontsize=8)
fig.tight_layout()
plt.savefig(f'data/climate2/figures/temporal_derivative/fig_{fi}.png')
plt.show()
fi += 1

fig, ax = plt.subplots(dpi=150)
ax.plot((pre_post_curves_auc_sum / (pre_post_curves_auc_sum.sum(axis=1) + 1e-12)[:, None])[:, sts_plot], label=labels)
ax.set_title('Sum of AUC up to a date (normed by day)')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
plt.hlines(0, 0, len(groups), colors='black', linestyles='dashed')
ax.legend(bbox_to_anchor=(1.3, 1.05), fontsize=8)
fig.tight_layout()
plt.savefig(f'data/climate2/figures/temporal_derivative/fig_{fi}.png')
plt.show()
fi += 1

fig, ax = plt.subplots(dpi=150)
ax.plot(smooth(supertopic_counts - means[:, None]).T[:, sts_plot], label=labels)
ax.set_title('AUC per day (smoothed)')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
plt.hlines(0, 0, len(groups), colors='black', linestyles='dashed')
ax.legend(bbox_to_anchor=(1.3, 1.05), fontsize=8)
fig.tight_layout()
plt.savefig(f'data/climate2/figures/temporal_derivative/fig_{fi}.png')
plt.show()
fi += 1

fig, ax = plt.subplots(dpi=150)
ax.plot(((supertopic_counts_smooth - means[:, None]) /
         np.abs(supertopic_counts - means[:, None]).sum(axis=1)[:, None]).T[:, sts_plot], label=labels)
ax.set_title(' AUC per day (smoothed, normed)')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
ax.legend(bbox_to_anchor=(1.3, 1.05), fontsize=8)
plt.hlines(0, 0, len(groups), colors='black', linestyles='dashed')
fig.tight_layout()
plt.savefig(f'data/climate2/figures/temporal_derivative/fig_{fi}.png')
plt.show()
fi += 1

fig, ax = plt.subplots(dpi=150)
ax.plot(((supertopic_counts_smooth - means[:, None]) /
         np.abs(supertopic_counts - means[:, None]).sum(axis=1)[:, None]).T[:, sts_plot], label=labels)
ax.set_title(' AUC per day (smoothed, normed)')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
ax.legend(bbox_to_anchor=(1.3, 1.05), fontsize=8)
plt.hlines(0, 0, len(groups), colors='black', linestyles='dashed')
fig.tight_layout()
plt.savefig(f'data/climate2/figures/temporal_derivative/fig_{fi}.png')
plt.show()
fi += 1

fig = plt.figure(figsize=(10, 20), dpi=150)
for i, st in enumerate(sts_plot, start=1):
    ax = plt.subplot(len(sts_plot), 1, i)
    ax.set_title(st.name)
    x = np.arange(0, len(groups))
    y = smooth(supertopic_counts / supertopic_counts[sts_plot].sum(axis=0), kernel_size=30)[st].T
    ax.plot(x, y, color='black')
    threshold = y.mean()
    ax.axhline(threshold, color='green', lw=2, alpha=0.5)
    ax.fill_between(x, threshold, y, where=y > threshold, color='green', alpha=0.5)
    ax.fill_between(x, y, threshold, where=y < threshold, color='red', alpha=0.5)
    ax.set_ylim(0, 0.5)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
fig.tight_layout()
plt.savefig(f'data/climate2/figures/temporal_derivative/fig_{fi}.png')
plt.show()
fi += 1

bound = groups.index('2020-02-01')
fig = plt.figure(figsize=(10, 20), dpi=150)
fig.suptitle('Mean normalised (each plot with own yscale)', y=1)
for i, st in enumerate(sts_plot, start=1):
    ax = plt.subplot(len(sts_plot), 1, i)
    ax.set_title(st.name)
    x = np.arange(0, len(groups))
    y = smooth([supertopic_counts[st] / supertopic_counts[st].mean()], kernel_size=30).reshape(-1, )
    y_pre = y[:bound].mean()
    ax.plot(x, y, color='black')
    threshold = y.mean()
    ax.axhline(threshold, color='green', lw=2, alpha=0.5)
    ax.axhline(y[:bound].mean(), color='black', ls=':', lw=2, alpha=0.5, xmax=bound / len(groups))
    ax.axhline(y[bound:].mean(), color='black', ls=':', lw=2, alpha=0.5, xmin=bound / len(groups))
    ax.fill_between(x, threshold, y, where=y > threshold, color='green', alpha=0.5)
    ax.fill_between(x, y, threshold, where=y < threshold, color='red', alpha=0.5)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
fig.tight_layout()
plt.savefig(f'data/climate2/figures/temporal_derivative/fig_{fi}.png')
plt.show()
fi += 1

bound = groups.index('2020-02-01')
fig = plt.figure(figsize=(10, 20), dpi=150)
fig.suptitle('Mean normalised (shared yscale)', y=1)
for i, st in enumerate(sts_plot, start=1):
    ax = plt.subplot(len(sts_plot), 1, i)
    ax.set_title(st.name)
    x = np.arange(0, len(groups))
    y = smooth([supertopic_counts[st] / supertopic_counts[st].mean()], kernel_size=30).reshape(-1, )
    y_pre = y[:bound].mean()
    ax.plot(x, y, color='black')
    threshold = y.mean()
    ax.axhline(threshold, color='green', lw=2, alpha=0.5)
    ax.axhline(y[:bound].mean(), color='black', ls=':', lw=2, alpha=0.5, xmax=bound / len(groups))
    ax.axhline(y[bound:].mean(), color='black', ls=':', lw=2, alpha=0.5, xmin=bound / len(groups))
    ax.fill_between(x, threshold, y, where=y > threshold, color='green', alpha=0.5)
    ax.fill_between(x, y, threshold, where=y < threshold, color='red', alpha=0.5)
    ax.set_ylim(0, 8.5)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
fig.tight_layout()
plt.savefig(f'data/climate2/figures/temporal_derivative/fig_{fi}.png')
plt.show()
fi += 1

bound = groups.index('2020-02-01')
fig = plt.figure(figsize=(10, 20), dpi=150)
fig.suptitle('Mean normalised (symlog yscale)', y=1)
for i, st in enumerate(sts_plot, start=1):
    ax = plt.subplot(len(sts_plot), 1, i)
    ax.set_title(st.name)
    x = np.arange(0, len(groups))
    y = smooth([supertopic_counts[st] / supertopic_counts[st].mean()], kernel_size=30).reshape(-1, )
    y_pre = y[:bound].mean()
    ax.plot(x, y, color='black')
    threshold = y.mean()
    ax.axhline(threshold, color='green', lw=2, alpha=0.5)
    ax.axhline(y[:bound].mean(), color='black', ls=':', lw=2, alpha=0.5, xmax=bound / len(groups))
    ax.axhline(y[bound:].mean(), color='black', ls=':', lw=2, alpha=0.5, xmin=bound / len(groups))
    ax.fill_between(x, threshold, y, where=y > threshold, color='green', alpha=0.5)
    ax.fill_between(x, y, threshold, where=y < threshold, color='red', alpha=0.5)
    ax.set_yscale('symlog')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
fig.tight_layout()
plt.savefig(f'data/climate2/figures/temporal_derivative/fig_{fi}.png')
plt.show()
fi += 1

bound = groups.index('2020-02-01')
fig = plt.figure(figsize=(10, 20), dpi=150)
fig.suptitle('Mean normalised (each plot with own yscale)', y=1)
for i, st in enumerate(sts_plot, start=1):
    ax = plt.subplot(len(sts_plot), 1, i)
    ax.set_title(st.name)
    x = np.arange(0, len(groups))
    y = smooth([supertopic_counts[st] / supertopic_counts[st].mean()], kernel_size=30).reshape(-1, )
    y_pre = y[:bound].mean()
    ax.plot(x, y, color='black')
    threshold = y.mean()
    ax.axhline(threshold, color='green', lw=2, alpha=0.5)
    ax.axhline(y[:bound].mean(), color='black', ls=':', lw=2, alpha=0.5, xmax=bound / len(groups))
    ax.axhline(y[bound:].mean(), color='black', ls=':', lw=2, alpha=0.5, xmin=bound / len(groups))
    ax.fill_between(x, threshold, y, where=y > threshold, color='green', alpha=0.5)
    ax.fill_between(x, y, threshold, where=y < threshold, color='red', alpha=0.5)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
fig.tight_layout()
plt.savefig(f'data/climate2/figures/temporal_derivative/fig_{fi}.png')
plt.show()
fi += 1

bound = groups.index('2020-01-01')
fig = plt.figure(figsize=(10, 20), dpi=150)
for i, st in enumerate(sts_plot, start=1):
    ax = plt.subplot(len(sts_plot), 1, i)
    ax.set_title(st.name)
    ax.axvline(bound, color='black', lw=2, alpha=0.5)
    # plot the data
    x = np.arange(0, len(groups))
    y = smooth([supertopic_counts[st] / supertopic_counts[st].mean()], kernel_size=30).reshape(-1, )
    ax.plot(x, y, color='black')
    ax.set_ylabel('Proportion from mean')
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

    ax2 = ax.twinx()
    y = supertopic_counts[st] / supertopic_counts.sum(axis=0)
    ax2.scatter(x, y, c='orange', s=0.4, marker=',')
    # plot regression line for period before and after pandemic
    sns.regplot(x=x[:bound], y=y[:bound], ax=ax2, scatter=False)
    sns.regplot(x=x[bound:], y=y[bound:], ax=ax2, scatter=False)
    # set the opacity of confidence interval
    plt.setp(ax2.collections[1], alpha=0.3)
    plt.setp(ax2.collections[2], alpha=0.3)
    # exclude extreme values by setting ylim
    ax2.set_ylim(np.percentile(y, q=1), np.percentile(y, q=99))
    ax2.set_ylabel('Share of tweets')

fig.tight_layout()
plt.savefig(f'data/climate2/figures/temporal_derivative/fig_{fi}.png')
plt.show()
fi += 1

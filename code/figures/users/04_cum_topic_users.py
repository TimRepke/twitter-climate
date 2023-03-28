from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import date2num
from datetime import datetime
from tqdm import tqdm
from scripts.util import smooth

# 0 created_at
# 1 id
# 2 author_id
# 3 t_km
# 4 st_cov
# 5 st_pol
# 6 st_mov
# 7 st_imp
# 8 st_cau
# 9 st_sol
# 10 st_con
# 11 st_oth

topics = {
    'Covid': 4,
    'Politics': 5,
    'Movements': 6,
    'Impacts': 7,
    'Causes': 8,
    'Solutions': 9,
    'Contrarian': 10,
    'Other': 11
}

topic_authors = defaultdict(lambda: defaultdict(list))
with open('data/climate2/meta_filtered_annotated_all.csv', 'r') as f_in:
    for line in tqdm(f_in):
        s = line.split(',')
        for topic, topic_i in topics.items():
            if int(s[topic_i]) > 0:
                topic_authors[s[0][:10]][topic].append(s[2])

keys = list(sorted(topic_authors.keys()))
bound = keys.index('2020-01-01')
keys_dt = [datetime.strptime(key, '%Y-%m-%d') for key in keys]
keys_ts = date2num(keys_dt)

topic_nums = {}
for topic in topics.keys():
    print(f'Processing {topic}')
    num_tweets = [len(topic_authors[key][topic]) for key in keys]
    num_authors = [len(set(topic_authors[key][topic])) for key in keys]

    cum_num_tweets = np.cumsum(num_tweets)
    cum_num_authors = []
    authors_cum = set()
    for key in tqdm(keys):
        authors_cum = authors_cum.union(set(topic_authors[key][topic]))
        cum_num_authors.append(len(authors_cum))

    topic_nums[topic] = {
        'num_tweets': num_tweets,
        'num_authors': num_authors,
        'cum_num_tweets': cum_num_tweets,
        'cum_num_authors': np.array(cum_num_authors)
    }

fig, axes = plt.subplots(nrows=1, ncols=len(topics) - 1, figsize=(30, 10))
for topic, ax in zip(topics.keys(), axes):
    ax.set_title(topic)
    ax.axvline(date2num(datetime(year=2020, month=1, day=1)))
    if topic == 'Other':
        pass
    elif topic == 'Covid':
        ax.plot_date(keys_ts, np.hstack([[0] * bound, topic_nums[topic]['cum_num_authors'][bound:]]),
                     '-', xdate=True, ydate=False, label='users (cum)')
        ax.plot_date(keys_ts, np.hstack([[0] * bound, topic_nums[topic]['cum_num_tweets'][bound:]]),
                     '-', xdate=True, ydate=False, label='tweets (cum)')
    else:
        ax.plot_date(keys_ts, topic_nums[topic]['cum_num_authors'], '-', xdate=True, ydate=False, label='users (cum)')
        ax.plot_date(keys_ts, topic_nums[topic]['cum_num_tweets'], '-', xdate=True, ydate=False, label='tweets (cum)')
fig.autofmt_xdate()
plt.legend()
plt.tight_layout()
plt.savefig(f'data/climate2/figures/users/topics_tweets_users_cum.png')
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=len(topics)-1, figsize=(30, 10))
for topic, ax in zip(topics.keys(), axes):
    ax.set_title(topic)
    ax.axvline(date2num(datetime(year=2020, month=1, day=1)), color='gray', ls='--')
    y = topic_nums[topic]['cum_num_tweets'] / topic_nums[topic]['cum_num_authors']
    if topic == 'Other':
        pass
    elif topic == 'Covid':
        ax.plot_date(keys_ts, np.hstack([[0] * bound, y[bound:]]), '-',
                     xdate=True, ydate=False, label='tweets/users (cum)')
    else:
        ax.plot_date(keys_ts, y, '-',
                     xdate=True, ydate=False, label='tweets/users (cum)')
fig.autofmt_xdate()
plt.legend()
plt.tight_layout()
plt.savefig(f'data/climate2/figures/users/topics_tweets_div_users_cum.png')
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=len(topics)-1, figsize=(30, 10), sharey='all')
for topic, ax in zip(topics.keys(), axes):
    ax.set_title(topic)
    ax.axvline(date2num(datetime(year=2020, month=1, day=1)))
    if topic == 'Other':
        pass
    elif topic == 'Covid':
        ax.plot_date(keys_ts, np.hstack([[0] * bound, topic_nums[topic]['cum_num_authors'][bound:]]),
                     '-', xdate=True, ydate=False, label='users (cum)')
        ax.plot_date(keys_ts, np.hstack([[0] * bound, topic_nums[topic]['cum_num_tweets'][bound:]]),
                     '-', xdate=True, ydate=False, label='tweets (cum)')
    else:
        ax.plot_date(keys_ts, topic_nums[topic]['cum_num_authors'], '-', xdate=True, ydate=False, label='users (cum)')
        ax.plot_date(keys_ts, topic_nums[topic]['cum_num_tweets'], '-', xdate=True, ydate=False, label='tweets (cum)')
fig.autofmt_xdate()
plt.legend()
plt.tight_layout()
plt.savefig(f'data/climate2/figures/users/topics_tweets_users_cum_sharey.png')
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=len(topics)-1, figsize=(30, 10), sharey='all')
for topic, ax in zip(topics.keys(), axes):
    ax.set_title(topic)
    ax.axvline(date2num(datetime(year=2020, month=1, day=1)), color='gray', ls='--')
    y = topic_nums[topic]['cum_num_tweets'] / topic_nums[topic]['cum_num_authors']
    if topic == 'Other':
        pass
    elif topic == 'Covid':
        ax.plot_date(keys_ts, np.hstack([[0] * bound, y[bound:]]), '-',
                     xdate=True, ydate=False, label='tweets/users (cum)')
    else:
        ax.plot_date(keys_ts, y, '-',
                     xdate=True, ydate=False, label='tweets/users (cum)')
fig.autofmt_xdate()
plt.legend()
plt.tight_layout()
plt.savefig(f'data/climate2/figures/users/topics_tweets_div_users_cum_sharey.png')
plt.show()

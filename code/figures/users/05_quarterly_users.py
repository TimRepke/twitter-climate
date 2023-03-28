from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import date2num
from datetime import datetime
from tqdm import tqdm
from scripts.util import smooth
import pandas as pd

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
topic_list = list(topics.keys())

timeframes = [
    ('2018', '2018-01-01', '2018-12-31'),
    ('Q1 2018', '2018-01-01', '2018-03-31'),
    ('Q2 2018', '2018-04-01', '2018-06-30'),
    ('Q3 2018', '2018-07-01', '2018-09-30'),
    ('Q4 2018', '2018-10-01', '2018-12-31'),
    ('2019', '2019-01-01', '2019-12-31'),
    ('Q1 2019', '2019-01-01', '2019-03-31'),
    ('Q2 2019', '2019-04-01', '2019-06-30'),
    ('Q3 2019', '2019-07-01', '2019-09-30'),
    ('Q4 2019', '2019-10-01', '2019-12-31'),
    ('2020', '2020-01-01', '2020-12-31'),
    ('Q1 2020', '2020-01-01', '2020-03-31'),
    ('Q2 2020', '2020-04-01', '2020-06-30'),
    ('Q3 2020', '2020-07-01', '2020-09-30'),
    ('Q4 2020', '2020-10-01', '2020-12-31'),
    ('2021', '2021-01-01', '2021-12-31'),
    ('Q1 2021', '2021-01-01', '2021-03-31'),
    ('Q2 2021', '2021-04-01', '2021-06-30'),
    ('Q3 2021', '2021-07-01', '2021-09-30'),
    ('Q4 2021', '2021-10-01', '2021-12-31'),
    ('Pre', '2018-01-01', '2019-12-31'),
    ('Post', '2020-01-01', '2021-12-31'),
    ('All', '2018-01-01', '2021-12-31'),
]

authors = defaultdict(lambda: defaultdict(list))
with open('data/climate2/meta_filtered_annotated_all.csv', 'r') as f_in:
    for line in tqdm(f_in):
        s = line.split(',')
        for topic, topic_i in topics.items():
            if int(s[topic_i]) > 0:
                authors[topic][s[0][:10]].append(s[2])

rows = []
dict_raw = {}
dict_uids = {}
for timeframe, start, end in timeframes:
    row = {
        'Interval': timeframe
    }
    f_uids = []
    f_uids_noother = []

    dict_uids[timeframe] = {}
    dict_raw[timeframe] = {}

    for topic in topic_list:
        uids = [
            aid
            for day, auth_ids in authors[topic].items()
            if start <= day <= end
            for aid in auth_ids
        ]

        f_uids += uids
        if topic != 'Other':
            f_uids_noother += uids

        n_users = len(set(uids))
        n_tweets = len(uids)

        row[topic] = f'{n_tweets / n_users:.2f} = {n_tweets:,} / {n_users:,}'
        dict_raw[timeframe][topic] = (n_tweets / n_users, n_tweets, n_users)
        dict_uids[timeframe][topic] = uids

    n_users = len(set(f_uids))
    n_tweets = len(f_uids)
    row['All'] = f'{n_tweets / n_users:.2f} = {n_tweets:,} / {n_users:,}'
    dict_raw[timeframe]['All'] = (n_tweets / n_users, n_tweets, n_users)
    dict_uids[timeframe]['All'] = f_uids

    n_users = len(set(f_uids_noother))
    n_tweets = len(f_uids_noother)
    row['All (w/o Other)'] = f'{n_tweets / n_users:.2f} = {n_tweets:,} / {n_users:,}'
    dict_raw[timeframe]['All (w/o Other)'] = (n_tweets / n_users, n_tweets, n_users)
    dict_uids[timeframe]['All (w/o Other)'] = f_uids_noother

    rows.append(row)

df = pd.DataFrame(rows)
print(df.to_markdown())

x = []
y = []
for row in dict_raw.values():
    for topic, values in row.items():
        if type(values) == tuple:
            x.append(values[1])
            y.append(values[2])
plt.scatter(x, y)
plt.show()

# plt.figure(dpi=300)
# for topic in rows[0].keys():
#     x = []
#     y = []
#     for row in dict_raw.values():
#         if row['Interval'] not in {'2018', '2019', '2020', '2021'}:
#             continue
#         if type(row[topic]) == tuple:
#             x.append(row[topic][1])
#             y.append(row[topic][2])
#         plt.scatter(x, y, label=f'{topic}|{row["Interval"]}',
#                     marker={'2018': 'o', '2019': '^', '2020': 'x', '2021': '1'}[row['Interval']])
# plt.yscale('log')
# plt.xscale('log')
# plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, ls='--')
# plt.legend(fontsize=5, bbox_to_anchor=(1.3, 1.))
# plt.tight_layout()
# plt.show()

xy_labels = ['2018', '2019', '2020', '2021']
keys = ['Covid', 'Politics', 'Movements', 'Impacts', 'Causes', 'Solutions', 'Contrarian', 'All (w/o Other)']
fig = plt.figure(figsize=(8, 40))
n_rows = len(keys)
n_cols = 2
for ti, topic in enumerate(keys, start=1):
    print(ti, topic)
    ax = plt.subplot(n_rows, n_cols, (ti * 2) - 1)
    ax.set_title(f'{topic}', fontdict={'fontsize': 22})

    matrix = np.zeros((len(xy_labels), len(xy_labels)))
    for xi, x_label in enumerate(xy_labels):
        for yi, y_label in enumerate(xy_labels):
            if xi != yi:
                matrix[xi][yi] = len(set(dict_uids[x_label][topic]).intersection(set(dict_uids[y_label][topic])))

    plt.imshow(matrix, interpolation='none', aspect='equal', origin='lower', cmap='Greens')
    for (j, i), label in np.ndenumerate(matrix):
        plt.text(i, j, f'{int(label):,}', ha='center', va='center')
    plt.xticks(np.arange(len(xy_labels)), xy_labels)
    plt.yticks(np.arange(len(xy_labels)), xy_labels)

    ax = plt.subplot(n_rows, n_cols, ti * 2)
    ax.set_title(f'{topic}', fontdict={'fontsize': 22})

    matrix = np.zeros((len(xy_labels), len(xy_labels)))
    for xi, x_label in enumerate(xy_labels):
        for yi, y_label in enumerate(xy_labels):
            if xi != yi:
                matrix[xi][yi] = len(set(dict_uids[x_label][topic]).intersection(set(dict_uids[y_label][topic]))) / \
                                 len(set(dict_uids[x_label][topic]).union(set(dict_uids[y_label][topic])))

    plt.imshow(matrix, interpolation='none', aspect='equal', origin='lower', cmap='Greens')
    for (j, i), label in np.ndenumerate(matrix):
        plt.text(i, j, f'{label:.2f}', ha='center', va='center')
    plt.xticks(np.arange(len(xy_labels)), xy_labels)
    plt.yticks(np.arange(len(xy_labels)), xy_labels)

plt.tight_layout()
plt.show()

xy_labels = ['Q1 2018', 'Q2 2018', 'Q3 2018', 'Q4 2018',
             'Q1 2019', 'Q2 2019', 'Q3 2019', 'Q4 2019',
             'Q1 2020', 'Q2 2020', 'Q3 2020', 'Q4 2020',
             'Q1 2021', 'Q2 2021', 'Q3 2021', 'Q4 2021']
keys = ['Covid', 'Politics', 'Movements', 'Impacts', 'Causes', 'Solutions', 'Contrarian', 'All (w/o Other)']
fig = plt.figure(figsize=(16, len(keys) * 8), dpi=200)
n_rows = len(keys)
n_cols = 2
for ti, topic in enumerate(keys, start=1):
    print(ti, topic)
    ax = plt.subplot(n_rows, n_cols, (ti * 2) - 1)
    ax.set_title(f'{topic}', fontdict={'fontsize': 22})

    matrix = np.zeros((len(xy_labels), len(xy_labels)))
    for xi, x_label in enumerate(xy_labels):
        for yi, y_label in enumerate(xy_labels):
            if xi != yi:
                matrix[xi][yi] = len(set(dict_uids[x_label][topic]).intersection(set(dict_uids[y_label][topic])))

    plt.imshow(matrix, interpolation='none', aspect='equal', origin='lower', cmap='Greens')
    # for (j, i), label in np.ndenumerate(matrix):
    #     plt.text(i, j, f'{int(label):,}', ha='center', va='center')
    plt.xticks(np.arange(len(xy_labels)), xy_labels)
    plt.yticks(np.arange(len(xy_labels)), xy_labels)

    ax = plt.subplot(n_rows, n_cols, ti * 2)
    ax.set_title(f'{topic}', fontdict={'fontsize': 22})

    matrix = np.zeros((len(xy_labels), len(xy_labels)))
    for xi, x_label in enumerate(xy_labels):
        for yi, y_label in enumerate(xy_labels):
            if xi != yi:
                matrix[xi][yi] = len(set(dict_uids[x_label][topic]).intersection(set(dict_uids[y_label][topic]))) / \
                                 len(set(dict_uids[x_label][topic]).union(set(dict_uids[y_label][topic])))

    plt.imshow(matrix, interpolation='none', aspect='equal', origin='lower', cmap='Greens', vmin=0, vmax=0.18)
    for (j, i), label in np.ndenumerate(matrix):
        plt.text(i, j, f'{label:.2f}', ha='center', va='center', fontsize=8)
    plt.xticks(np.arange(len(xy_labels)), xy_labels)
    plt.yticks(np.arange(len(xy_labels)), xy_labels)
plt.tight_layout()
plt.show()

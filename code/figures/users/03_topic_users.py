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
topic_list = list(topics.keys())

topic_authors = defaultdict(list)
with open('data/climate2/meta_filtered_annotated_all.csv', 'r') as f_in:
    for line in tqdm(f_in):
        s = line.split(',')
        for topic, topic_i in topics.items():
            if int(s[topic_i]) > 0:
                topic_authors[topic].append(s[2])

matrix_d0 = np.zeros((8, 8))
for i, (topic, authors) in enumerate(topic_authors.items()):
    print(f'Topic {topic} with {len(authors):,} tweets and {len(set(authors)):,} users '
          f'({len(authors) / len(set(authors)):.2f} tweets/user)')
    for j, (topic_b, authors_b) in enumerate(topic_authors.items()):
        if topic_b != topic:
            matrix_d0[i][j] = len(set(authors).intersection(set(authors_b)))

matrix = matrix_d0.copy()
for i in range(len(topics)):
    matrix[i][i] = len(set(topic_authors[topic_list[len(topics) - i - 1]]))

ticks = list(topic_authors.keys())

fig = plt.figure(figsize=(10, 10))
plt.title('Number of overlapping users')
plt.imshow(matrix_d0, interpolation='none', aspect='equal', origin='lower')
for (j, i), label in np.ndenumerate(matrix):
    plt.text(i, j, f'{int(label):,}', ha='center', va='center')
plt.xticks(np.arange(len(ticks)), ticks)
plt.yticks(np.arange(len(ticks)), ticks)
plt.tight_layout()
plt.savefig(f'data/climate2/figures/users/topic_users.png')
plt.show()

fig = plt.figure(figsize=(10, 10))
plt.title('Number of overlapping users')
plt.imshow(matrix, interpolation='none', aspect='equal', origin='lower')
for (j, i), label in np.ndenumerate(matrix):
    plt.text(i, j, f'{int(label):,}', ha='center', va='center')
plt.xticks(np.arange(len(ticks)), ticks)
plt.yticks(np.arange(len(ticks)), ticks)
plt.tight_layout()
plt.savefig(f'data/climate2/figures/users/topic_users_diag.png')
plt.show()

# ##################
# without "Others"
ticks = [t for t in ticks if t != 'Other']
fig = plt.figure(figsize=(10, 10))
plt.title('Number of overlapping users')
plt.imshow(matrix_d0[1:, 1:], interpolation='none', aspect='equal', origin='lower')
for (j, i), label in np.ndenumerate(matrix[1:, 1:]):
    plt.text(i, j, f'{int(label):,}', ha='center', va='center')
plt.xticks(np.arange(len(ticks)), ticks)
plt.yticks(np.arange(len(ticks)), ticks)
plt.tight_layout()
plt.savefig(f'data/climate2/figures/users/topic_users_noother.png')
plt.show()

fig = plt.figure(figsize=(10, 10))
plt.title('Number of overlapping users')
plt.imshow(matrix[1:, 1:], interpolation='none', aspect='equal', origin='lower')
for (j, i), label in np.ndenumerate(matrix[1:, 1:]):
    plt.text(i, j, f'{int(label):,}', ha='center', va='center')
plt.xticks(np.arange(len(ticks)), ticks)
plt.yticks(np.arange(len(ticks)), ticks)
plt.tight_layout()
plt.savefig(f'data/climate2/figures/users/topic_users_diag_noother.png')
plt.show()

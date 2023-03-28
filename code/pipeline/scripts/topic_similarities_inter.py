from typing import Literal
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
import json

file_labels = 'data/climate2/topics_big2/labels_7000000_tsne.npy'

file_embeddings = 'data/climate2/topics_big2/tweets_embeddings_7000000_True_minilm.npy'
file_layout = 'data/climate2/topics_big2/layout_7000000_tsne.npy'

print('loading...')
labels = np.load(file_labels)
vectors_2d = np.load(file_layout, mmap_mode='r')
vectors_hd = np.load(file_embeddings, mmap_mode='r')

print('centroids...')
centroids_2d = np.array([np.mean(vectors_2d[labels == topic_i], axis=0) for topic_i in np.unique(labels)])
centroids_hd = np.array([np.mean(vectors_hd[labels == topic_i], axis=0) for topic_i in np.unique(labels)])

print('distances...')
distances_2d = pairwise_distances(centroids_2d, centroids_2d, metric='euclidean')
distances_hd = pairwise_distances(centroids_hd, centroids_hd, metric='cosine')

print('sorting...')
neighbours_2d = np.argsort(distances_2d, axis=1)
neighbours_hd = np.argsort(distances_hd, axis=1)

print('prepping')
k = 10
results = []
for topic_i, cnt in zip(*np.unique(labels, return_counts=True)):
    top_2d = neighbours_2d[topic_i][1:k+1]
    top_hd = neighbours_hd[topic_i][1:k+1]
    results.append({
        'ld': list(zip(top_2d.tolist(), distances_2d[topic_i][top_2d].tolist())),
        'hd': list(zip(top_hd.tolist(), distances_hd[topic_i][top_hd].tolist()))
    })

print('merging...')
dumps = [
    'viewer/dump_7000000_monthly_downsampled2.json',
    'viewer/dump_7000000_monthly_downsampled3.json'
]
for dump in dumps:
    with open(dump, 'r') as f_in:
        dump_o = json.load(f_in)
    dump_o['neighbours'] = results
    with open(dump, 'w') as f_out:
        json.dump(dump_o, f_out)

plt.imshow(distances_2d, interpolation='none', aspect='equal')
plt.title('2D')
plt.colorbar()
plt.show()

plt.imshow(distances_hd, interpolation='none', aspect='equal')
plt.title('embedding')
plt.colorbar()
plt.show()

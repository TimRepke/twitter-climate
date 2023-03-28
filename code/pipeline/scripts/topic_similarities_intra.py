from typing import Literal
from matplotlib import pyplot as plt
import numpy as np
import hnswlib

file_labels = 'data/climate2/topics_big2/labels_7000000_tsne.npy'

file_embeddings = 'data/climate2/topics_big2/tweets_embeddings_7000000_True_minilm.npy'
file_layout = 'data/climate2/topics_big2/layout_7000000_tsne.npy'

space: Literal['2d', 'hd'] = '2d'
k = 10

labels = np.load(file_labels)

if space == '2d':  # 2D layout
    vectors = np.load(file_layout, mmap_mode='r')
    metric = 'l2'
    dim = 2
else:  # embedding space
    vectors = np.load(file_embeddings, mmap_mode='r')
    metric = 'cosine'
    dim = vectors.shape[1]

# span_min = vectors.min(axis=1)
# span_max = vectors.max(axis=1)

results = []

for topic_i, cnt in zip(*np.unique(labels, return_counts=True)):
    if cnt > 200000:
        continue
    vectors_topic = vectors[labels == topic_i]
    index = hnswlib.Index(space=metric, dim=dim)
    index.init_index(max_elements=cnt, ef_construction=200, M=16)
    index.add_items(vectors_topic, np.arange(cnt))
    index.set_ef(k * 3)

    _, distances = index.knn_query(vectors_topic, k=k + 1)

    result = {
        'topic': topic_i,
        'size': cnt,
        'min': distances.min(),
        'max': distances.max(),
        'mean': np.mean(distances),
        'median': np.median(distances),
        'std': np.std(distances)
    }

    print(f'Topic {topic_i} ({cnt} tweets) | '
          f'min: {distances.min():.7f}, '
          f'max: {distances.max():.7f}, '
          f'mean: {np.mean(distances):.7f}, '
          f'median: {np.median(distances):.7f}, '
          f'std: {np.std(distances):.7f}')

    results.append(result)

y = [r['size'] for r in results]
for m in ['max', 'mean', 'median', 'std']:
    plt.scatter([r[m] for r in results], y, s=0.05)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('topic size')
    plt.xlabel(m)
    plt.show()

print('max,mean,median,std')
i = 0
for r in results:
    if i < r['topic']:
        for _ in range(r['topic'] - i):
            print('skip')
            i += 1
    print(f"{r['max']},{r['mean']},{r['median']},{r['std']}")
    i += 1


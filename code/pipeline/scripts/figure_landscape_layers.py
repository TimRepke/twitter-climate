from matplotlib import pyplot as plt
import numpy as np

print('loading labels')
labels = np.load('data/climate2/topics_big2/labels_7000000_tsne.npy')
print('loading layout')
layout = np.load('data/climate2/topics_big2/layout_7000000_tsne.npy')

lim_min = np.min(layout, axis=0)
lim_max = np.max(layout, axis=0)
print(lim_max, lim_min)
for topic in np.unique(labels):
    print(f'plotting topic {topic}')
    fig = plt.figure(figsize=(20, 20), dpi=120)

    topic_points = layout[labels == topic]
    print(f' - num points: {topic_points.shape}; min: {topic_points.min(axis=0)}; max: {topic_points.max(axis=0)}')
    plt.scatter(topic_points[:, 0], topic_points[:, 1], alpha=0.5, s=0.03, marker='*')

    plt.xlim(lim_min[0], lim_max[0])
    plt.ylim(lim_min[1], lim_max[1])
    plt.axis('off')

    plt.savefig(f'data/climate2/topics_big2/layout_layers/topic_{topic}.png', transparent=True)
    plt.clf()
    plt.close(fig)

# img{max-width:100%;vertical-align:top;}
# .blend-overlay{ mix-blend-mode: overlay; }
# <div style="background:#822;">
#    <img class="blend-overlay" src="https://i.imgur.com/StEW4JD.jpg">
# </div>

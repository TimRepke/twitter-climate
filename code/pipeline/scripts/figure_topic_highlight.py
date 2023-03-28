import numpy as np
from matplotlib import pyplot as plt

print('loading labels')
labels = np.load('data/climate2/topics_big2/labels_7000000_tsne.npy')
print('loading layout')
layout = np.load('data/climate2/topics_big2/layout_7000000_tsne.npy')

highlight_groups = {
    'covid (215, 316, 712, 713, 714, 715)': ([702, 703, 366, 75, 203, 198], 0.4, 0.01),
    'topic 983': ([982], 0.1, 0.01),
    # 'topic 846': ([846], 0.02, 0.005),
    'outliers (0)': ([0], 0.05, 0.01)
}
highlight_topics = [t for g in highlight_groups.values() for t in g[0]]

fig = plt.figure(figsize=(20, 20), dpi=150)

print('plot baselayer')
layout_base = layout[np.isin(labels, highlight_topics, invert=True)]
plt.scatter(layout_base[:, 0], layout_base[:, 1], alpha=0.3, s=0.03, marker='*')

for name, (hl_group, alpha, s) in highlight_groups.items():
    print(f'adding "{name}"')
    layout_group = layout[np.isin(labels, hl_group)]
    plt.scatter(layout_group[:, 0], layout_group[:, 1], alpha=alpha, s=s, marker='.', label=name)

legend = plt.legend(loc='upper left', markerscale=300, prop={'size': 20})
# print('fix legend')
# for l in legend.get_lines():
#     l.set_alpha(10)
#     l.set_markersize(2)
print('show')
plt.show()

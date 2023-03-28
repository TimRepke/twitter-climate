import json
import numpy as np
from matplotlib import pyplot as plt
EPS = 1e-16
themes = {'COVID-19': [198, 702, 203],
          'People': [463, 352, 282, 289, 98, 251, 217],
          'Geographies': [756, 311, 225, 130, 253, 865, 839, 763, 158, 403, 764, 39, 355,
                          449, 838, 437, 169, 655, 645, 177],
          '"Old" economy': [693, 90, 687, 395],
          'Environment & Sustainability': [625, 577, 96],
          'Impacts': [324, 311, 903, 629, 329, 638, 207, 625, 577, 901, 96, 877, 7,
                      136, 437, 38, 655, 104],
          'Solutions': [667, 658, 597, 689, 827, 90, 851, 403, 765, 859, 355, 249, 787,
                        784],
          'social aspects, justice & human rights': [726, 741, 299, 140],
          'Contrarian': [982, 486, 715],
          'Human activites': [667, 371],
          'General CC discourse': [595, 770, 815, 895, 148, 520, 835, 805, 738, 511, 107],
          'Institutions': [850, 542],
          'Movements': [561, 463, 352, 265, 813, 507, 421]}

with open('data/climate2/topics_big2/temporal/temporal_7000000_monthly_raw_abs.json') as f:
    dump = json.load(f)

dists = np.array(dump['z'])
groups = dump['x']
topics = dump['y']

spot_topics = (np.max(dists, axis=1) / (np.sum(dists, axis=1)+EPS)) > 0.9
print(sum(spot_topics))
spot_groups = np.argmax(dists[spot_topics], axis=1)

print(spot_topics.shape)
print(spot_groups.shape)

spots, counts = np.unique(spot_groups, return_counts=True)
print(spots.shape)
print(counts.shape, counts.sum())
x = groups
y = np.zeros((len(x),))
for s, c in zip(spots, counts):
    y[s] = c

plt.bar(x=np.arange(0, len(x)), height=y)
ticks = np.arange(0, len(x), 3)
plt.xticks(ticks, [x[i] for i in ticks])
plt.tick_params(labelrotation=90)
plt.title('Histogram of spot topics per month')
plt.show()

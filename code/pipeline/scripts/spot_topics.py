from matplotlib import pyplot as plt
import numpy as np
import json

with open('data/climate2/topics_big2/temporal/temporal_7000000_monthly_raw_abs.json', 'r') as f:
    dump = json.load(f)

dist = np.array(dump['z'])
rels = dist.max(axis=1) / (dist.sum(axis=1)+1e-16)
print(rels.shape)
lst = []
for frac in range(1000):
    frac_ = 1 - (frac / 1000)
    lst.append(np.sum(dist.sum(axis=1)[rels > frac_]))

plt.plot(lst)
plt.show()

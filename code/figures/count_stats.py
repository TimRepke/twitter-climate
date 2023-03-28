import json
from collections import defaultdict
from datetime import datetime, timedelta
from scripts.util import DateFormat, smooth
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# NAME = 'cc'
# START_TIME = datetime(year=2007, month=1, day=1)

NAME = 'cc_Jan2007_Sept2022'
START_TIME = datetime(year=2007, month=1, day=1)
SMOOTHING = 90
SHOW_EN_STAT = False

counts = {}
with open(f'data/counts/{NAME}.jsonl', 'r') as f_in:
    for line in f_in:
        page = json.loads(line)
        for cnt in page['data']:
            counts[cnt['start'][:10]] = cnt['tweet_count']

time_series = []
dates = []

time_cursor = START_TIME
for day, count in sorted(counts.items(), key=lambda e: e[0]):
    while time_cursor.strftime('%Y-%m-%d') < day:
        time_series.append(0)
        dates.append(time_cursor.strftime('%Y-%m-%d'))
        time_cursor += timedelta(days=1)

    time_series.append(count)
    dates.append(time_cursor.strftime('%Y-%m-%d'))
    time_cursor += timedelta(days=1)

# for d, c in zip(dates, time_series):
#     print(f'{d}   {c:,}')

# yearly mean/min/max/std of tweets per day
yearly_agg = defaultdict(list)
for day, count in zip(dates, time_series):
    yearly_agg[day[:4]].append(count)
print('Tweets matching \'"climate change" lang:en -is:retweet -is:quote\'')
growths = []
prev = None
for yr, counts in sorted(yearly_agg.items(), key=lambda e: e[0]):
    counts = np.array(counts)
    growth = 0
    if prev is not None:
        growth = (counts.sum() - prev) / prev
        if yr >= '2011':
            growths.append(growth)
    print(f'{yr}: total = {counts.sum():,} '
          f'| perc5 = {np.percentile(counts, 5):,.0f} '
          f'| perc95 = {np.percentile(counts, 95):,.0f} '
          f'| mean = {counts.mean():,.2f} '
          f'| max = {counts.max():,} '
          f'| std = {counts.std():,.2f} '
          f'| growth = {growth:.2%}')
    prev = counts.sum()
print()
print(f'Average growth {START_TIME.year}-{yr}: {np.mean(growths):.2%}')

if SHOW_EN_STAT:
    print('\n----------------\n')
    print('Total English tweets')

    yearly_agg = defaultdict(list)
    with open('data/climate2/english_tweet_counts_daily_2006-2021-rt.csv', 'r') as f_in:
        next(f_in)
        for line in f_in:
            day, count = line.split(',')
            count = int(count)
            if day[:4] >= '2007':
                yearly_agg[day[:4]].append(count)

    growths = []
    prev = None
    for yr, counts in sorted(yearly_agg.items(), key=lambda e: e[0]):
        counts = np.array(counts)
        growth = 0
        if prev is not None:
            growth = (counts.sum() - prev) / prev
            if yr >= '2011':
                growths.append(growth)
        print(f'{yr}: total = {counts.sum():,} '
              f'| perc5 = {np.percentile(counts, 5):,.0f} '
              f'| perc95 = {np.percentile(counts, 95):,.0f} '
              f'| mean = {counts.mean():,.2f} '
              f'| max = {counts.max():,} '
              f'| std = {counts.std():,.2f} '
              f'| growth = {growth:.2%}')
        prev = counts.sum()
    print()
    print(f'Average growth 2011-2021: {np.mean(growths):.2%}')

SMOOTHING = 60
START_YEAR = 2007

x = np.arange(len(dates))
xticks = []
xticklabels = []
for i, g in enumerate(dates):
    s = g.split('-')
    if int(s[1]) % 3 == 0 and int(s[2]) == 1:
        xticks.append(i)
        xticklabels.append(g)
y = time_series
y_smooth = smooth([y], kernel_size=SMOOTHING, with_pad=True)[0]

fig = plt.figure(figsize=(15, 5))
fig.suptitle('Tweets matching \'"climate change" lang:en -is:retweet -is:quote\' (resolution: daily)')
ax = plt.subplot()

ax.scatter(x, y, color='orange', alpha=0.9, s=0.1, marker='o')
ax.plot(x, y_smooth, color='black')

SHADE_BOUNDS = [dates.index('2019-12-18'), dates.index('2020-03-11')]
plt.axvspan(SHADE_BOUNDS[0], SHADE_BOUNDS[1], color='black', alpha=0.1, lw=0)
ax.axvline(SHADE_BOUNDS[0], color='black', lw=1, alpha=0.2)
ax.axvline(SHADE_BOUNDS[1], color='black', lw=1, alpha=0.2)

for yr in range(2007, 2023, 1):
    if yr < START_YEAR:
        continue
    s = dates.index(f'{yr}-01-01')
    try:
        e = dates.index(f'{yr}-12-31')
        ax.axvline(e, color='black', ls='--', lw=1, alpha=0.5)
    except ValueError:
        e = len(dates) - 1
    avg = np.array(time_series[s:e + 1]).mean()
    ax.hlines(avg, xmin=s, xmax=e, color='black', ls='--', lw=1, alpha=0.5)
    ax.text(s + 5, avg + 300, f'∅{avg:,.0f}/day')

ax.set_xticks(xticks)
ax.set_xticklabels([tl[:7] for tl in xticklabels], rotation=45, fontsize=8)
ax.set_ylim(0, 30000)

ax.set_xlim(dates.index(f'{START_YEAR}-01-01'), len(time_series))

plt.tight_layout()
plt.show()

import json
from collections import defaultdict
from datetime import datetime, timedelta
from scripts.util import DateFormat, smooth
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

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

FONT_SIZE_1 = 18
FONT_SIZE_2 = 20
FONT_SIZE_3 = 22

SMOOTHING = 14
START_YEAR = 2018  # inclusive
END_YEAR = 2021  # inclusive
SHOW_XLABELS = True
MAX_VAL = 35000

x = np.arange(len(dates))
y = time_series
y_smooth = smooth([y], kernel_size=SMOOTHING, with_pad=True)[0]

y = np.array(y)
y_smooth = y_smooth

fig = plt.figure(figsize=(20, 4), dpi=300)
# fig.suptitle('Tweets matching \'"climate change" lang:en -is:retweet -is:quote\' (resolution: daily)')
ax = plt.subplot()

ax.scatter(x, y, color='#ffb93a', s=0.2, marker='o')
ax.plot(x, y_smooth, color='black')

SHADE_BOUNDS = [dates.index('2019-12-18'), dates.index('2020-03-11')]
plt.axvspan(SHADE_BOUNDS[0], SHADE_BOUNDS[1], color='#ffa90e', alpha=0.2, lw=0)  # no alpha #ffedce
ax.axvline(SHADE_BOUNDS[0], color='#ffa90e', lw=1, alpha=0.7)  # no alpha #f3b442
ax.axvline(SHADE_BOUNDS[1], color='#ffa90e', lw=1, alpha=0.7)  # no alpha #f3b442

for yr in range(START_YEAR, END_YEAR + 1, 1):
    s = dates.index(f'{yr}-01-01')
    try:
        e = dates.index(f'{yr}-12-31')
        ax.axvline(s, color='black', ls='-', lw=1, alpha=0.5)
    except ValueError:
        e = len(dates) - 1
    avg = np.array(time_series[s:e + 1]).mean()
    tot = np.array(time_series[s:e + 1]).sum()
    ax.hlines(avg, xmin=s, xmax=e, color='black', ls='--', lw=1, alpha=0.5)
    # ax.text(s + 5, avg + 300, f'∅{avg:,.0f}/day')

    # ax.text(s + 175, 27000, f'∅{avg:,.0f}/day', fontdict={'fontsize': FONT_SIZE_2, 'ha': 'center'})
    ax.text(s + 175, 27000, f'AVG {avg:,.0f}/day', fontdict={'fontsize': FONT_SIZE_2, 'ha': 'center'})
    ax.text(s + 175, 31000, f'{tot:,.0f}/yr', fontdict={'fontsize': FONT_SIZE_2, 'ha': 'center'})
    for ss, se, plot in [(f'{yr}-01-01', f'{yr}-03-31', True),
                         (f'{yr}-04-01', f'{yr}-06-30', True),
                         (f'{yr}-07-01', f'{yr}-09-30', True),
                         (f'{yr}-10-01', f'{yr}-12-31', True),
                         (f'{yr}-01-01', f'{yr}-12-31', False)]:
        s = dates.index(ss)
        e = dates.index(se)
        avg = np.array(time_series[s:e + 1]).mean()
        print(f'{ss} to {se} | daily avg: {avg:,.0f}; sum: {np.array(time_series[s:e + 1]).sum():,.0f}')
        if plot:
            avg = avg
            # ax.hlines(avg, xmin=s, xmax=e, color='black', ls='--', lw=1, alpha=0.3)
            ax.axvspan(s, e, 0, avg / MAX_VAL, color='black', alpha=0.15, lw=0)
    print('--')

xticks_ = [dates.index(f'{yr}-{quarter_start}')
           for yr in range(START_YEAR, END_YEAR + 1, 1)
           for quarter_start in ['01-01', '04-01', '07-01', '10-01']
           ]
xticks_.append(dates.index(f'{END_YEAR}-12-31'))
xticklabels_ = []
for yr in range(START_YEAR, END_YEAR + 1, 1):
    xticklabels_.append('')
    xticklabels_.append('')
    if SHOW_XLABELS:
        xticklabels_.append(f'{yr}')
    else:
        xticklabels_.append('')
    xticklabels_.append('')
xticklabels_.append('')
ax.set_xticks(xticks_)
ax.set_xticklabels(xticklabels_, fontsize=FONT_SIZE_3)
ax.set_ylim(0, MAX_VAL)

ax.set_xlim(dates.index(f'{START_YEAR}-01-01'), dates.index('2021-12-31'))
ax.set_ylabel('Tweets per day\non "climate change"', fontdict={'fontsize': FONT_SIZE_2})
ax.tick_params(axis='y', right=True, left=True, labelsize=FONT_SIZE_1)
ax.get_yaxis().set_major_formatter(FuncFormatter(lambda v, p: f'{v / 1000:.0f}k'))
ax.text(4040, 31000, 'a', fontdict={'fontsize': 30, 'fontweight': 'bold', 'ha': 'center'})
plt.tight_layout()
plt.savefig('data/climate2/figures/counts/tweets_cc.pdf')
plt.show()

import json
from collections import defaultdict
from datetime import datetime, timedelta

from matplotlib.ticker import FuncFormatter

from scripts.util import DateFormat, smooth
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

time_series = []
dates = []

with open('data/climate2/english_tweet_counts_daily_2006-2021-rt.csv', 'r') as f_in:
    next(f_in)
    for line in f_in:
        day, count = line.split(',')
        time_series.append(int(count))
        dates.append(day)

FONT_SIZE_1 = 18
FONT_SIZE_2 = 20
FONT_SIZE_3 = 22

SMOOTHING = 14
START_YEAR = 2018  # inclusive
END_YEAR = 2021  # inclusive
SHOW_XLABELS = False
MAX_VAL = 50000000

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

y = np.array(y)
y_smooth = y_smooth

fig = plt.figure(figsize=(20, 4), dpi=300)
# fig.suptitle('Tweets matching \'"climate change" lang:en -is:retweet -is:quote\' (resolution: daily)')
ax = plt.subplot()

ax.scatter(x, y, color='orange', alpha=0.9, s=0.3, marker='o')
ax.plot(x, y_smooth, color='black')

SHADE_BOUNDS = [dates.index('2019-12-18'), dates.index('2020-03-11')]
plt.axvspan(SHADE_BOUNDS[0], SHADE_BOUNDS[1], color='#ffa90e', alpha=0.2, lw=0)
ax.axvline(SHADE_BOUNDS[0], color='#ffa90e', lw=1, alpha=0.7)
ax.axvline(SHADE_BOUNDS[1], color='#ffa90e', lw=1, alpha=0.7)

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
    ax.text(s + 175, 1000000, f'AVG {avg:,.0f}/day', fontdict={'fontsize': FONT_SIZE_2, 'ha': 'center'})
    ax.text(s + 175, 6500000, f'{tot:,.0f}/yr', fontdict={'fontsize': FONT_SIZE_2, 'ha': 'center'})

    # for ss, se in [(f'{yr}-01-01', f'{yr}-03-31'),
    #                (f'{yr}-04-01', f'{yr}-06-30'),
    #                (f'{yr}-07-01', f'{yr}-09-30'),
    #                (f'{yr}-10-01', f'{yr}-12-31')]:
    #     s = dates.index(ss)
    #     e = dates.index(se)
    #     avg = np.array(time_series[s:e + 1]).mean() / FACTOR
    #     # ax.hlines(avg, xmin=s, xmax=e, color='black', ls='--', lw=1, alpha=0.3)
    #     ax.axvspan(s, e, 0, avg / MAX_VAL, color='black', alpha=0.15, lw=0)

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

ax.set_xlim(dates.index(f'{START_YEAR}-01-01'), dates.index(f'{END_YEAR}-12-31'))
# ax.set_ylabel('Tweets per day (x1,000,000)', fontdict={'fontsize': 20})
ax.set_ylabel('est. English Tweets\nper day', fontdict={'fontsize': FONT_SIZE_2})
ax.tick_params(axis='y', right=True, left=True, labelsize=FONT_SIZE_1)
ax.get_yaxis().set_major_formatter(FuncFormatter(lambda v, p: f'{v / 1000000:.0f}M'))
ax.text(4320, 44200000, 'b', fontdict={'fontsize': 30, 'fontweight': 'bold', 'ha': 'center'})
plt.tight_layout()
plt.savefig('data/climate2/figures/counts/tweets_en.pdf')
plt.show()

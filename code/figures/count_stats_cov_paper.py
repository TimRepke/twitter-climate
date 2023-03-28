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

with open('data/climate2/covid_tweet_counts_daily_2006-2021-rt_specific2.csv', 'r') as f_in:
    next(f_in)
    for line in f_in:
        _, _, _, count, day = line.split(',')
        time_series.append(int(count))
        dates.append(day.strip())

# unsorted in the file, so fix that
tmp = sorted(zip(dates, time_series), key=lambda d: d[0])
dates = [d[0] for d in tmp]
time_series = [d[1] for d in tmp]

df = pd.read_csv('data/climate2/owid-covid-data.csv')
dff = df[df['iso_code'] == 'OWID_WRL']
lkp = {row[1]['date']: row[1]['new_cases_smoothed']
       for row in dff[['date', 'new_cases_smoothed']].fillna(0).iterrows()}
cov_cases = np.array([lkp.get(dt, 0) for dt in dates])

SMOOTHING = 14
START_YEAR = 2018
MAX_VAL = 1600000
MAX_VAL = MAX_VAL
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
y_cov = cov_cases
y_smooth = y_smooth

fig = plt.figure(figsize=(20, 4), dpi=300)
# fig.suptitle('Tweets matching \'"climate change" lang:en -is:retweet -is:quote\' (resolution: daily)')
ax = plt.subplot()

ax.scatter(x, y, color='orange', alpha=0.9, s=0.3, marker='o')
ax.plot(x, y_smooth, color='black', label='14-day moving average')
ax.plot(x, y_cov, color='red', label='COVID cases', ls='dashdot')

SHADE_BOUNDS = [dates.index('2019-12-18'), dates.index('2020-03-11')]
plt.axvspan(SHADE_BOUNDS[0], SHADE_BOUNDS[1], color='#ffa90e', alpha=0.2, lw=0)
ax.axvline(SHADE_BOUNDS[0], color='#ffa90e', lw=1, alpha=0.7)
ax.axvline(SHADE_BOUNDS[1], color='#ffa90e', lw=1, alpha=0.7)

for yr in range(2007, 2022, 1):
    if yr < START_YEAR:
        continue
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
    ax.text(s + 175, MAX_VAL - 200000, f'AVG {avg:,.0f}/day',
            fontdict={'fontsize': 20, 'ha': 'center', 'va': 'top'})
    ax.text(s + 175, MAX_VAL - 50000, f'{tot:,.0f}/yr',
            fontdict={'fontsize': 20, 'ha': 'center', 'va': 'top'})
    # for ss, se in [(f'{yr}-01-01', f'{yr}-03-31'),
    #                (f'{yr}-04-01', f'{yr}-06-30'),
    #                (f'{yr}-07-01', f'{yr}-09-30'),
    #                (f'{yr}-10-01', f'{yr}-12-31')]:
    #     s = dates.index(ss)
    #     e = dates.index(se)
    #     avg = np.array(time_series[s:e + 1]).mean() / FACTOR
    #     # ax.hlines(avg, xmin=s, xmax=e, color='black', ls='--', lw=1, alpha=0.3)
    #     ax.axvspan(s, e, 0, avg / MAX_VAL, color='black', alpha=0.15, lw=0)

ax.set_xticks([
    dates.index('2018-01-01'),
    dates.index('2018-04-01'),
    dates.index('2018-07-01'),
    dates.index('2018-10-01'),
    dates.index('2019-01-01'),
    dates.index('2019-04-01'),
    dates.index('2019-07-01'),
    dates.index('2019-10-01'),
    dates.index('2020-01-01'),
    dates.index('2020-04-01'),
    dates.index('2020-07-01'),
    dates.index('2020-10-01'),
    dates.index('2021-01-01'),
    dates.index('2021-04-01'),
    dates.index('2021-07-01'),
    dates.index('2021-10-01'),
    dates.index('2021-12-31')
])
ax.set_xticklabels([
    '', '', '2018', '',
    '', '', '2019', '',
    '', '', '2020', '',
    '', '', '2021', '', ''
], fontsize=22)
# ax.set_xticklabels([
#     '', '', '', '',
#     '', '', '', '',
#     '', '', '', '',
#     '', '', '', '', ''
# ], fontsize=22)
ax.set_ylim(0, MAX_VAL)
ax.set_yticks([0, 500000, 1000000, 1500000])

ax.set_xlim(dates.index(f'{START_YEAR}-01-01'), dates.index('2021-12-31'))
ax.set_ylabel('Tweets per day\non COVID', fontdict={'fontsize': 20})
ax.tick_params(axis='y', right=True, left=True, labelsize=18)
ax.get_yaxis().set_major_formatter(FuncFormatter(lambda v, p: f'{v / 1000000:.1f}M'))
ax.text(4332, 1400000, 'c', fontdict={'fontsize': 30, 'fontweight': 'bold', 'ha': 'center'})
plt.legend(loc='lower left', fontsize=20)
plt.tight_layout()
plt.savefig('data/climate2/figures/counts/tweets_cov.pdf')
plt.show()

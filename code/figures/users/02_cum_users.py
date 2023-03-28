from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import date2num
from datetime import datetime
from tqdm import tqdm
from scripts.util import smooth

# 0 created_at
# 1 id
# 2 author_id
# 3 t_km
# 4 st_cov
# 5 st_pol
# 6 st_mov
# 7 st_imp
# 8 st_cau
# 9 st_sol
# 10 st_con
# 11 st_oth

for cutoff, fmt, desc, smoothing in [(7, '%Y-%m', 'monthly', 0), (10, '%Y-%m-%d', 'daily', 30)]:
    authors = defaultdict(list)
    with open('data/climate2/meta_filtered_annotated_all.csv', 'r') as f_in:
        for line in tqdm(f_in):
            s = line.split(',')
            authors[s[0][:cutoff]].append(s[2])

    keys = list(sorted(authors.keys()))
    keys_dt = [datetime.strptime(key, fmt) for key in keys]
    keys_ts = date2num(keys_dt)
    num_tweets = [len(authors[key]) for key in keys]
    num_authors = [len(set(authors[key])) for key in keys]

    cum_num_tweets = np.cumsum(num_tweets)
    cum_num_authors = []
    authors_cum = set()
    for key in tqdm(keys):
        authors_cum = authors_cum.union(set(authors[key]))
        cum_num_authors.append(len(authors_cum))

    fig = plt.figure()
    if smoothing > 0:
        plt.plot_date(keys_ts, smooth([num_authors], smoothing)[0], '-', xdate=True, ydate=False, label='users')
        plt.plot_date(keys_ts, smooth([num_tweets], smoothing)[0], '-', xdate=True, ydate=False, label='tweets')
    else:
        plt.plot_date(keys_ts, num_authors, '-', xdate=True, ydate=False, label='users')
        plt.plot_date(keys_ts, num_tweets, '-', xdate=True, ydate=False, label='tweets')
    fig.autofmt_xdate()
    plt.legend()
    plt.savefig(f'data/climate2/figures/users/{desc}_tweets_users.png')
    plt.show()

    fig = plt.figure()
    if smoothing > 0:
        plt.plot_date(keys_ts, smooth([np.array(num_tweets) / np.array(num_authors)], smoothing)[0],
                      '-', xdate=True, ydate=False, label='tweets/user')
    else:
        plt.plot_date(keys_ts, np.array(num_tweets) / np.array(num_authors),
                      '-', xdate=True, ydate=False, label='tweets/user')
    fig.autofmt_xdate()
    plt.legend()
    plt.savefig(f'data/climate2/figures/users/{desc}_tweets_div_users.png')
    plt.show()

    fig = plt.figure()
    plt.plot_date(keys_ts, cum_num_authors, '-', xdate=True, ydate=False, label='users (cum)')
    plt.plot_date(keys_ts, cum_num_tweets, '-', xdate=True, ydate=False, label='tweets (cum)')
    fig.autofmt_xdate()
    plt.legend()
    plt.savefig(f'data/climate2/figures/users/{desc}_tweets_users_cum.png')
    plt.show()

    fig = plt.figure()
    plt.plot_date(keys_ts, cum_num_authors, '-', xdate=True, ydate=False, label='users (cum)')
    fig.autofmt_xdate()
    plt.legend()
    plt.savefig(f'data/climate2/figures/users/{desc}_users_cum.png')
    plt.show()

    fig = plt.figure()
    plt.plot_date(keys_ts, cum_num_tweets, '-', xdate=True, ydate=False, label='tweets (cum)')
    fig.autofmt_xdate()
    plt.legend()
    plt.savefig(f'data/climate2/figures/users/{desc}_tweets_cum.png')
    plt.show()

    fig = plt.figure()
    plt.plot_date(keys_ts, cum_num_tweets / np.array(cum_num_authors), '-',
                  xdate=True, ydate=False, label='tweets/users (cum)')
    fig.autofmt_xdate()
    plt.legend()
    plt.savefig(f'data/climate2/figures/users/{desc}_tweets_div_users_cum.png')
    plt.show()

    fig = plt.figure()
    plt.plot_date(keys_ts, np.array(cum_num_authors) / cum_num_tweets, '-',
                  xdate=True, ydate=False, label='users/tweets (cum)')
    fig.autofmt_xdate()
    plt.legend()
    plt.savefig(f'data/climate2/figures/users/{desc}_users_div_tweets_cum.png')
    plt.show()

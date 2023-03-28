import json
import re
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

models = ['cardiff-stance-climate', 'cardiff-offensive',  # 'cards',
          'cardiff-sentiment', 'bertweet-sentiment',
          'geomotions-orig', 'geomotions-ekman', 'cardiff-emotion', 'bertweet-emotions', 'nrc']

stats = {
    model: defaultdict(int)
    for model in models
}
cooc_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))

DATASET = 1
FILE_TWEETS = ['data/climate2/tweets_classified_7000000_False.jsonl',
               'data/geoengineering/tweets_classified2.jsonl'][DATASET]
FILE_OUT = ['data/climate2/tweets_classified_7000000_False.csv',
            'data/geoengineering/tweets_classified.csv'][DATASET]

with open(FILE_TWEETS, 'r') as f_in, \
        open(FILE_OUT, 'w') as f_out:
    f_out.write(','.join(models) + ',text\n')
    li = 0
    for line in tqdm(f_in):
        li += 1
        tweet = json.loads(line)
        for model in models:
            labels = list(tweet['classes'][model].keys())
            f_out.write('|'.join(labels) + ',')
            for label in labels:
                stats[model][label] += 1

            for model_cooc in models:
                if model_cooc != model:
                    labels_cooc = list(tweet['classes'][model_cooc].keys())
                    for label_cooc in labels_cooc:
                        for label in labels:
                            cooc_stats[model][label][model_cooc][label_cooc] += 1
        f_out.write(re.sub(r'(\s+|,)', ' ', tweet['text']) + '\n')
        # if li > 1000:
        #     break

print(stats)

# print(json.dumps(stats, indent=3))
# print(json.dumps(cooc_stats, indent=3))
print('plotting')
fig = plt.figure(figsize=(40, 40), dpi=120)
spi = 0
for i, model_i in enumerate(models):
    for j, model_j in enumerate(models):
        spi += 1
        if model_j != model_i:

            plt.subplot(len(models), len(models), spi, xmargin=10, ymargin=10)
            labels_i = sorted(list(cooc_stats[model_i].keys()), reverse=True)
            labels_j = sorted(list(cooc_stats[model_j].keys()), reverse=True)
            if model_i == 'cards':
                labels_i.remove('0_0')
            x = np.zeros((len(labels_i), len(labels_j)))
            for li, label_i in enumerate(labels_i):
                for lj, label_j in enumerate(labels_j):
                    x[li][lj] = cooc_stats[model_i][label_i][model_j][label_j]

            plt.imshow(x, interpolation='none')
            plt.ylabel(model_i, rotation=90)
            plt.xlabel(model_j)
            plt.xticks(np.arange(len(labels_j)), labels_j, rotation=90, fontsize=6)
            plt.yticks(np.arange(len(labels_i)), labels_i, fontsize=6)

print('layout+show')
fig.tight_layout()
plt.show()
#
# {'cardiff-stance-climate': {'none': 401220, 'favor': 1066763},
#  'cardiff-offensive': {'offensive': 44852, 'not-offensive': 1423131},
#  'cardiff-sentiment': {'negative': 338337, 'neutral': 931686, 'positive': 197960},
#  'bertweet-sentiment': {'negative': 376438, 'neutral': 896848, 'positive': 194697},
#  'geomotions-orig': {'surprise': 1123856, 'confusion': 25457, 'curiosity': 33793, 'fear': 6928, 'disgust': 1904, 'gratitude': 19005, 'excitement': 6193, 'approval': 64845, 'anger': 6447, 'admiration': 43558, 'disappointment': 8950, 'remorse': 2729, 'neutral': 26889, 'amusement': 4420, 'desire': 6284, 'caring': 11081, 'annoyance': 20757, 'disapproval': 21612, 'sadness': 7641, 'embarrassment': 610, 'love': 4617, 'nervousness': 716, 'joy': 2920, 'pride': 14195, 'relief': 1316, 'optimism': 1073, 'realization': 186, 'grief': 1},
#  'geomotions-ekman': {'neutral': 1174550, 'fear': 8547, 'joy': 185387, 'disgust': 1488, 'surprise': 56472, 'anger': 34105, 'sadness': 7434},
#  'cardiff-emotion': {'anger': 386030, 'sadness': 306153, 'optimism': 579827, 'joy': 195973},
#  'bertweet-emotions': {'anger': 24782, 'others': 1236294, 'disgust': 109933, 'fear': 39790, 'joy': 49429, 'sadness': 2202, 'surprise': 5553},
#  'nrc': {'fear': 327889, 'anger': 238695, 'negative': 562795, 'sadness': 259782, 'disgust': 185412, 'anticipation': 329305, 'positive': 593580, 'trust': 403964, 'joy': 222798, 'surprise': 161042}
#  }

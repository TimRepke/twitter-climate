import os
import json
import numpy as np
from typing import Literal
import plotly.graph_objects as go
from colorcet import glasbey
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import xlsxwriter

DATASET = 'climate2'
LIMIT = 7000000
SOURCE_DIR = f'data/{DATASET}/topics_big2'
TARGET_FILE = f'{SOURCE_DIR}/topic_stats_{LIMIT}.xlsx'

DATE_FORMAT: Literal['monthly', 'yearly', 'weekly', 'daily'] = 'monthly'

workbook = xlsxwriter.Workbook(TARGET_FILE)
for boost in [[], ['retweets'], ['replies'], ['likes'], ['retweets', 'likes']]:  # , ['retweets', 'likes', 'replies']]:
    for clip in [False, True]:
        worksheet = workbook.add_worksheet(f'{"_".join(boost or ["raw"])}{"_clipped" if clip else ""}')
        row = 0
        col = 0

        with open(f'{SOURCE_DIR}/temporal/temporal_{LIMIT}_{DATE_FORMAT}_{"_".join(boost or ["raw"])}_abs.json') as f:
            data = json.load(f)
            vectors = np.array(data['z'])[1:]
            topics = data['y']
            groups = data['x']
        # for i, d in enumerate(data['x']):
        #     print(i, d)
        if clip:
            for m, mv in [('median', np.median(vectors)),
                          ('mean', np.mean(vectors)),
                          ('max', np.max(vectors)),
                          ('perc 90', np.percentile(vectors, 90)),
                          ('perc 95', np.percentile(vectors, 95)),
                          ('perc 99', np.percentile(vectors, 99))]:
                worksheet.write(row, 0, m)
                worksheet.write(row, 1, mv)
                if m == 'perc 95':
                    worksheet.write(row, 2, 'used as maximum value')
                row += 1
            row += 2
            vectors = vectors.clip(max=np.percentile(vectors, 95))

        worksheet.write(row, 1, 'Topic')
        for i, t in enumerate(topics):
            worksheet.write(row + i + 1, 0, i)
            worksheet.write(row + i + 1, 1, t)

        worksheet.write(row, 2, 'Total')
        for i, v in enumerate(vectors.sum(axis=1)):
            worksheet.write(row + i + 1, 2, v)

        worksheet.write(row, 3, '2018-01 to 2019-12')
        for i, v in enumerate(vectors[:, :24].sum(axis=1)):
            worksheet.write(row + i + 1, 3, v)

        worksheet.write(row, 4, '2020-01 to 2021-12')
        for i, v in enumerate(vectors[:, 24:].sum(axis=1)):
            worksheet.write(row + i + 1, 4, v)

        # stages of the pandemic (based on our world in data)
        # stage 2: 01.03.2020 (pandemic emerges, stringent lockdowns)
        worksheet.write(row, 5, '2020-03 to 2021-09')
        for i, v in enumerate(vectors[:, 26:33].sum(axis=1)):
            worksheet.write(row + i + 1, 5, v)

        # stage 3: 01.10.2020 (vaccines, variants, shifting epicenter)
        worksheet.write(row, 6, '2020-03 to 2021-09')
        for i, v in enumerate(vectors[:, 33:43].sum(axis=1)):
            worksheet.write(row + i + 1, 6, v)

        # stage 4: 01.08.2021 (re-emergence)
        worksheet.write(row, 7, '2020-03 to 2021-09')
        for i, v in enumerate(vectors[:, 43:].sum(axis=1)):
            worksheet.write(row + i + 1, 7, v)

        worksheet.set_column(1, 1, max([len(t) for t in topics]))
        worksheet.set_column(3, 7, 19)

workbook.close()
#
# splits = [
#     0,  # 0 2018-01
#     6,  # 5 2018-07
#     12,  # 12 2019-01
#     18,  # 17 2019-07
#     24,  # 24 2020-01
#     30,  # 29 2020-07
#     36,  # 36 2021-01
#     42,  # 41 2021-07
#     47  # 47 2021-12
# ]
#
# slices = np.vstack([
#     vectors[:, i:j].sum(axis=1)
#     for i, j in zip(splits[:-1], splits[1:])
# ])
# print(slices.shape)
# labels = np.argmax(slices, axis=0)
# print(labels)
#
# # bc = vectors[:, :24].sum(axis=1)
# # ac = vectors[:, 24:].sum(axis=1)
# # labels = np.zeros((len(topics),), dtype=int)
# # labels[ac >= bc] = 1
#
# # print(vectors.sum(axis=1))
# fig = go.Figure(go.Heatmap(
#     # z=np.vstack([vectors[ac < bc], vectors[ac >= bc]]),
#     z=np.vstack([vectors[labels == c][vectors[labels == c].sum(axis=1).argsort()] for c in np.unique(labels)]),
#     x=[f'd:{d}' for d in data['x']],
#     y=[data['y'][di[0]] for c in np.unique(labels) for di in np.argwhere(labels == c)],
#     hoverongaps=False))
# print(np.unique(labels, return_counts=True))
# pos = 0
# for c, cnt in zip(*np.unique(labels, return_counts=True)):
#     fig.add_hline(y=pos,
#                   line_dash="dot",
#                   annotation_text=f'cluster {c}',
#                   annotation_position="bottom right",
#                   annotation_font_size=10,
#                   annotation_font_color="blue")
#     pos += cnt
# for s in splits[1:-1]:
#     fig.add_vline(x=s,
#                   line_dash="dot",
#                   annotation_text=f'split {s}',
#                   annotation_position="top right",
#                   annotation_font_size=10,
#                   annotation_font_color="blue")
#
# fig.update_layout(title=f'Norm: {norm}, boost: {"_".join(boost)}, '
#                         f'cluster dist: {np.unique(labels, return_counts=True)}')
# # fig.show()
# fig.write_html(f'{TARGET_DIR}/splits_{LIMIT}_{DATE_FORMAT}_{norm}_{"_".join(boost)}.html')
#
# x = np.arange(len(groups))
# for c in np.unique(labels):
#     y = vectors[labels == c].sum(axis=0)
#     plt.plot(x, y, label=f'cluster {c}')
# plt.title(f'Norm: {norm}, boost: {"_".join(boost)}')
# plt.legend()
# plt.show()

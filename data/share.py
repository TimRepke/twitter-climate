import json
from tqdm import tqdm
from typing import Optional, Union


def read_raw_line(lni: int, ln: str) -> dict[str, Optional[Union[str, float, int, bool]]]:
    obj = json.loads(ln)
    fil = irel.get(lni, None)

    return {
        'twitter_id': obj['id'],  # Tweet ID from Twitter
        'rel': lni in rel,  # True iff Tweet is contained in analysis
        'filters': {
            'dup': fil[1],  # 1 iff this is a duplicate (excl first)
            'lan': fil[2],  # 1 iff language is English (and not None)
            'txt': fil[3],  # 1 iff status text is not None
            'mit': fil[4],  # 1 iff text has minimum number of tokens (>=4)
            'mah': fil[5],  # 1 iff text has less than maximum number of hashtags (<=5),
            'pfd': fil[6],  # 1 iff tweet was posted after 01.01.2018
            'ptd': fil[7],  # 1 iff tweet was posted before 31.12.2021
            'cli': fil[8],  # 1 iff tweet actually contains "climate change" (API matches some false positives)
        } if fil is not None else None,
        'ann': None  # annotations
    }


def extend_annotation(ln: str) -> None:
    obj = json.loads(ln)
    tweets_l[obj['id']]['ann'] = {
        't_km': obj['t_km'],  # topic (based on "keep & majority vote" strategy)
        't_kp': obj['t_kp'],  # topic (based on "keep & closest topic centroid [proximity]" strategy)
        't_fm': obj['t_fm'],  # topic (based on "drop sample topic [fresh] & majority vote" strategy)
        't_fp': obj['t_fp'],  # topic (based on "drop sample topic [fresh] & closest topic centroid [proximity]")
        'st_int': obj['st_int'],  # theme annotation "Interesting"
        'st_nr': obj['st_nr'],  # theme annotation "Non-relevant / spam"
        'st_cov': obj['st_cov'],  # theme annotation "COVID"
        'st_pol': obj['st_pol'],  # theme annotation "Politics"
        'st_mov': obj['st_mov'],  # theme annotation "Movements"
        'st_imp': obj['st_imp'],  # theme annotation "Impacts"
        'st_cau': obj['st_cau'],  # theme annotation "Causes"
        'st_sol': obj['st_sol'],  # theme annotation "Solutions"
        'st_con': obj['st_con'],  # theme annotation "Contrarian"
        'st_oth': obj['st_oth'],  # theme annotation "Other"
        'x': obj['x'],  # x position in 2D representation
        'y': obj['y'],  # x position in 2D representation
        'sample': obj['sample']  # true iff this tweet was in the original topic model sample
    }


with (
    open('data/climate2/tweets_raw.jsonl', 'r') as fraw,
    open('data/climate2/tweets_relevant_True_True_4_5_2018-01_2022-01_True_False_False.txt', 'r') as frel,
    open('data/climate2/tweets_irrelevant_True_True_4_5_2018-01_2022-01_True_False_False.txt', 'r') as firel
):
    rel = set([int(line) for line in tqdm(frel, desc='Read relevance file...')])
    irel = {int(line.split('|')[0]): [int(pt) for pt in line.split('|')]
            for line in tqdm(firel, desc='Read irrelevance file...')}

    tweets = [read_raw_line(li, line) for li, line in tqdm(enumerate(fraw), desc='Read tweets...')]
    del rel
    del irel

    tweets_l = {t['twitter_id']: t for t in tqdm(tweets, desc='Create tweet lookup...')}
    del tweets

with open('data/climate2/tweets_filtered_annotated_all.jsonl', 'r') as fann:
    [extend_annotation(line) for line in tqdm(fann, desc='Extend tweet info...')]

with open('data/climate2/share.jsonl', 'w') as fout:
    [fout.write(json.dumps(t) + '\n') for t in tqdm(tweets_l.values(), desc='Writing data...')]

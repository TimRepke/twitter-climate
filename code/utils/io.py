import json
import math
import os
from tqdm import tqdm
from typing import Optional


def count_tweets(file_path: str, echo: bool = False):
    if echo:
        print(f'Counting tweets in {file_path}...')
    with open(file_path) as f:
        num_lines = sum(1 for _ in f)
        if echo:
            print(f'  - File contains {num_lines} tweets.')
        return num_lines


def exit_if_exists(file_path: str):
    if os.path.exists(file_path):
        print(f'The file {file_path} already exists. If you are sure you want to proceed, delete it first.')
        exit(1)


def ensure_folder(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def produce_batches(file_path: str, batch_size: int, init_skip: int = 0, limit: Optional[int] = None):
    num_lines = count_tweets(file_path, echo=True)

    if limit:
        if limit < num_lines:
            print(f"Limit is set, will only use the first {limit} lines!")
            num_lines = limit

    n_batches = math.ceil(num_lines / batch_size)

    with open(file_path, 'r') as f_in:
        line_num = 0
        for _ in range(init_skip):
            next(f_in)
            line_num += 1

        for batch_i in range(n_batches):
            tqdm.write(f'===== PROCESSING BATCH {batch_i + 1} ({(batch_i + 1) * batch_size:,}/{num_lines:,}) =====')

            tweets = []
            while len(tweets) < batch_size and line_num < num_lines:
                tweets.append(json.loads(next(f_in)))
                line_num += 1

            tqdm.write(f'Current file pos: {line_num}; '
                       f'Tweets from {tweets[0]["created_at"]} to {tweets[-1]["created_at"]}')
            yield tweets


def batched_lines(file_path: str, batch_size: int):
    with open(file_path, 'r') as f_in:
        lines_batch = []
        for line in f_in:
            lines_batch.append(line)
            if len(lines_batch) >= batch_size:
                yield lines_batch
                lines_batch = []
        yield lines_batch


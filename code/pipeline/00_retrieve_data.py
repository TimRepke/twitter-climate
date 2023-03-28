import json
import psycopg
from psycopg.rows import dict_row
from collections import defaultdict, Counter
from tqdm import tqdm
import os

DATASET = 'climate'
DB_ID = 83
BATCH_SIZE = 200000
TARGET_FILE_UNSRT = f'data/{DATASET}/tweets_raw_unsorted.jsonl'
TARGET_FILE_SRT = f'data/{DATASET}/tweets_raw.jsonl'

if os.path.exists(TARGET_FILE_UNSRT) or os.path.exists(TARGET_FILE_SRT):
    print(f'The files {TARGET_FILE_SRT} or {TARGET_FILE_UNSRT} already exist. '
          f'If you are sure you want to proceed, delete them first.')
    exit(1)

with psycopg.connect(f'postgresql://{os.environ["DB_USER"]}:{os.environ["DB_PASS"]}@'
                     f'{os.environ["DB_HOST"]}:{os.environ["DB_PORT"]}/{os.environ["DB_NAME"]}',
                     row_factory=dict_row) as conn:
    # Open a cursor to perform database operations
    with conn.cursor(row_factory=dict_row) as cur, open(TARGET_FILE_UNSRT, 'w') as f_out:
        batch_i = 0
        srt_groups = defaultdict(int)
        while True:
            print(f'Executing query for batch {batch_i} (LIMIT {BATCH_SIZE} OFFSET {batch_i * BATCH_SIZE})...')
            cur.execute(f"""
                SELECT ts.*, tt.*
                FROM (
                    SELECT *
                    FROM twitter_status_searches
                    WHERE twittersearch_id = {DB_ID}
                    ORDER BY id
                    LIMIT {BATCH_SIZE} OFFSET {batch_i * BATCH_SIZE}) tss
                LEFT JOIN twitter_status ts on tss.status_id = ts.twitterbasemodel_ptr_id
                LEFT JOIN twitter_twitterbasemodel tt on ts.twitterbasemodel_ptr_id = tt.id;""")

            # am I done?
            if cur.rowcount <= 0:
                break

            print('Writing unsorted records...')
            for record in cur:
                # convert dates
                if record['created_at'] is None:
                    continue
                record['created_at'] = record['created_at'].isoformat()
                # record['fetched'] = record['fetched'].isoformat()

                # save some space
                del record['source']
                del record['source_url']
                del record['fetched']
                del record['truncated']
                del record['favorited']
                del record['retweeted']
                del record['contributors']
                del record['coordinates']
                del record['geo']
                del record['api_got']
                del record['scrape_got']
                del record['entities']

                # prepare for sorting later
                srt_groups[record['created_at'][:7]] += 1

                # write out
                f_out.write(record['created_at'][:7] + json.dumps(record) + '\n')
            batch_i += 1


# with open(TARGET_FILE_UNSRT) as f_in:
#     srt_groups = Counter([l[:7] for l in tqdm(f_in)])


def get_group(fp, group):
    cnt = 0
    for l_ in tqdm(fp):
        if cnt >= srt_groups[group]:
            break
        if l_[:7] == group:
            cnt += 1
            yield l_[7:]


sort_group_keys = sorted(srt_groups.keys())
print(f'Sorting by date in {len(sort_group_keys)} groups from {sort_group_keys[0]} to {sort_group_keys[-1]}...')
with open(TARGET_FILE_SRT, 'w') as f_out:
    for sort_group in sort_group_keys:
        print(f'Sorting tweets with partial date {sort_group}')
        with open(TARGET_FILE_UNSRT) as f_in:
            tweets = [json.loads(l) for l in get_group(f_in, sort_group)]
            for tweet in sorted(tweets, key=lambda t: t['created_at']):
                f_out.write(json.dumps(tweet) + '\n')

print('Deleting unsorted data...')
os.remove(TARGET_FILE_UNSRT)

import json
from tqdm import tqdm

with open('data/climate2/tweets_filtered_annotated_all.jsonl', 'r') as f_in, \
        open('data/climate2/meta_filtered_annotated_all.csv', 'w') as f_out:
    for line in tqdm(f_in):
        tweet = json.loads(line)
        f_out.write(f'{tweet["created_at"][:19]},{tweet["id"]},{tweet["author_id"]},'
                    f'{tweet["t_km"]},{tweet["st_cov"]},{tweet["st_pol"]},{tweet["st_mov"]},'
                    f'{tweet["st_imp"]},{tweet["st_cau"]},{tweet["st_sol"]},{tweet["st_con"]},{tweet["st_oth"]}\n')

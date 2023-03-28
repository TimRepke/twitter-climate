# Pipeline for processing Tweets

### `00_retrieve_data.py`
This queries the database to fetch tweets associated with a `twittersearch_id` defined by `DB_ID`.
Since the query can be too slow to process for the database, this is done in batches.
The data is sorted by date afterwards.
You need to set environment variables for database credentials and connection info.

### `01_prepare_data.py`
This should always be run before any of the following steps.
It parses the tweets and populates the `meta` field for each object with hashtags etc. and cleans the tweet text.

### `02_filter_data.py`
This removes duplicates and tweets deemed as irrelevant using some basic rules.
Thus, it should always run before any of the following steps.
Furthermore, it can be used to downsample the dataset evenly across time. This is useful if downstream tasks 
can't handle the loads or you want to work with a representative smaller set of data.

The target size can be set using `LIMIT`

### `03_01_embed_data.py` / `03_02_classify_data.py`
The first one is required to run the landscape task.
It computes document embeddings using the selected `EMBEDDING_MODEL`.
Use `INCLUDE_HASHTAGS` if they should be part of the document embedding, otherwise it uses the cleaned tweet text.

The second one is required to run the sentiment tasks, and optional for the landscape task.
In both cases, you need to fit the `LIMIT` parameter to what you had before.

### Output tasks
They are independent of one another. All require 00, 01, 02 to run first. Some have 03 requirements as stated.

#### `04_01_landscape.py`
Requires 03_01, 03_02 is optional.
This produces BERTopic / FrankenTopic visualisations and dumps topic descriptions to console.

#### `04_02_sentiments_list.py`
Requires 03_02.
This dumps the sentiment classification per tweet to console.

#### `04_03_sentiments_plot.py`
Requires 03_02.
This writes plots of sentiments over time to a folder.

#### `04_04_hashtags.py`
This produces similarity matrices of daily, monthly, or yearly count or tf-idf representations.
It also writes the top hashtags or tokens per selected grouping to disk.

#### `04_05_similarities.py`

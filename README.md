# Twitter Analysis of "climate change" tweets

This repository hosts code and data for our publication "Attention to climate change only temporarily diverted by COVID-19".   
**Authors:** Tim Repke, Max Callaghan, William Lamb, Sarah Lück, Finn Müller-Hansen, Jan Minx (all @MCC_Berlin)   
**Keywords:** Climate Change, COVID-19 Pandemic, Public Attention, Social Media, Topic Model

---
> The COVID-19 pandemic disrupted peoples' daily lives and dominated the public discourse.
> We analyze 13.5 million tweets on climate change during 2018--2021 and show that attention to climate dropped substantially in 2020 with the onset of the pandemic.
> While research has helped to explain this drop in the context of issue attention theory, our analysis highlights a remarkable recovery in attention in 2021 towards pre-pandemic levels.
> Moreover, our large-scale, transformer-based text analysis reveals important thematic shifts during this period.
> In particular, we show a sustained drop in attention to activist movements and subsequently an increased focus on climate causes and climate solutions.
> This means, that while the climate change discourse in general recovered from the COVID-19 pandemic, activist movements such as the school protests that have mobilized millions around the globe in 2019 have measurably lost traction on Twitter.

## Structure of this repository

### Data
As per Twitter ToS, we cannot share the full dataset, only the Tweet IDs.

* `cc_news_coverage.xlsx`: Media coverage (world-wide) of climate change by Fernandez et al.
* `pwid-covid-data.csv`: Number of COVID-19 cases worldwide
* `english_tweet_counts_daily_2006_2021-rt.csv`: Our approximation of overall daily (English-language) tweet volume
* `topic_annotations_and_stats.xlsx`: Annotator labels for topic to theme mappings, topic keyword lists, and statistics
* `share.jsonl`: Tweet IDs and topic annotations; see https://zenodo.org/record/7778199

### Code

The main processing of the dataset is done by the "pipeline".
All other follow-up analyses and supplemental scripts are contained in that folder as well.
For example, scripts to estimate the number of clusters and heuristically determine hyper-parameters.

The `utils` folder contains code to deploy large computations to a SLURM cluster.
This might be overly specific for reproduction, but added for completeness anyway.

### Figures

* `pre_post`: Plots and statistics to explore how themes are more prominent before or after the pandemic after normalising overall tweet volume over time
* `counts`: High-res figures used in the paper
* `stacked_areas`: Topics per supertopic over time as stacked area charts
* `quarterly_overlap.png`: Semantic overlap of tweets per quarter
* `topics_tweets_*.png`/`(daily|monthly)_(tweets|users)*.png`: Number of tweets per user rates (cumulative or per time interval)
* `topic_users*.png`: Overlap of users posting in multiple themes


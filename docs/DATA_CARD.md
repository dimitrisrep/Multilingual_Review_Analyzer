# Data card

| Source | Snapshot | Reviews | License metadata |
| --- | --- | ---: | --- |
| [Nikos Fragkis, Skroutz shop reviews](https://www.kaggle.com/datasets/nikosfragkis/skroutz-shop-reviews-sentiment-analysis) | Kaggle version 1, `Skroutz_dataset.xlsx` | 6,552 | Other (specified in description) |
| [Dimitrios Kotzias, Sentiment Labelled Sentences](https://doi.org/10.24432/C57604) | UCI, Amazon cells file | 1,000 | CC BY 4.0 |

`review-analyzer download` retrieves the pinned archives from their sources and checks their SHA-256 digests. Source archives and trained models are stored locally under `data/raw/` and `artifacts/`.

## Preprocessing and split

Reviews are normalized with Unicode NFKC, case folding and whitespace compression to identify duplicates. Empty reviews and groups with conflicting sentiment labels are excluded. Deduplication leaves 7,532 unique reviews.

With seed 42, the split reserves 15% for validation and 15% for testing, stratified by language and sentiment. The training partition is sampled to 300 reviews per language and label, for a total of 1,200. Validation and test each contain 1,130 reviews, including 982 Greek and 148 English. [Split manifest](../reports/split_manifest.json)

The Greek data covers shop-service reviews and the English data covers product reviews. The split is at review level, so the results reflect both source and language differences.

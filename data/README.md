# Data

Datasets aren't included in this repository because of size (~2 GB) and third-party licensing.
`main.py` expects them under `data/processed/` (configurable with `--data_dir`):

```
data/
├── raw/                                   # raw Amazon category dumps (only needed to rebuild the subgroup datasets)
│   ├── AMAZON_FASHION.json
│   ├── All_Beauty.json
│   ├── Appliances.json
│   └── Magazine_Subscriptions.json
└── processed/
    ├── amazon-yelp-dataset/
    │   ├── english-amazon-review-sentiment/   # DatasetDict: train / validation / test  (source domain)
    │   └── yelp-reviews-split/                # DatasetDict: train / validation / test  (target domain)
    ├── final_dataset/                         # DatasetDict: Appliances / Beauty / Fashion / Magazine
    └── final_dataset_balanced/                # same, label-balanced and de-duplicated
```

All splits are HuggingFace `datasets` saved with `save_to_disk`. Labels are binary sentiment
(`1` = positive, `0` = negative).

## Amazon → Yelp (main experiment)

| Folder | Role | Columns used | Size used |
|---|---|---|---|
| `english-amazon-review-sentiment` | source | `text`, `label` | first 10,000 rows of `train` |
| `yelp-reviews-split` | target | `text`, `label` | 400 train (selection pool) / 200 validation / 250 test |

The Amazon data is the English subset of a processed multilingual Amazon review sentiment dataset,
with 3-star reviews removed (1–2 stars → negative, 4–5 stars → positive). The Yelp data is a binary
restaurant-review sentiment dataset. Any data with the same layout (a `text` column and a binary
`label` column) will work.

## Amazon category subgroups (leave-one-category-out)

Raw files come from the per-category review files of the
[Amazon Review Data (2018)](https://nijianmo.github.io/amazon/) (Ni, Li & McAuley, EMNLP 2019).
Download the four category files into `data/raw/`, then run from this folder:

```bash
cd data
python convert_to_hf_data.py        # -> processed/final_dataset           (first 1,000 reviews per category)
python create_balanced_dataset.py   # -> processed/final_dataset_balanced  (up to 500 pos + 500 neg, de-duplicated)
```

Reviews with `overall > 3` are labeled positive, all others negative.
Run the experiment with `python main.py --dataset amazon-subgroup --target_domain Appliances ...`.

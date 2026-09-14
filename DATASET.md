# DATASET.md — Twitter US Airline Sentiment

## Dataset Name
Twitter US Airline Sentiment

## Original Source
Originally collected and annotated by **Crowdflower** (later **Figure Eight**, now part of Appen) and released publicly on Kaggle.

- Kaggle listing: `crowdflower/twitter-airline-sentiment`
- Mirror used to obtain the file for this repository (real, unmodified data — only the hosting differs): `github.com/fastforwardlabs/airline-sentiment` (`data/Tweets.csv`)
- Also indexed at: `https://paperswithcode.com/dataset/twitter-us-airline-sentiment`

## License
Released by Figure Eight / Crowdflower as an open, publicly downloadable dataset for research and educational use (Kaggle's standard "CC0 / Public Domain-like" listing terms for this specific dataset — the original Crowdflower listing did not attach a restrictive commercial license). This is a **third-party dataset** — it is **not** covered by this repository's MIT license (see `LICENSE`). If you plan any commercial use, verify current terms directly on the Kaggle dataset page before doing so.

## Number of Samples
**14,640** tweets (raw). After the project's cleaning/deduplication step (`src/data/preprocessing.py`), **14,226** tweets remain, split:

| Split | Rows |
|---|---|
| Train | 9,958 (70%) |
| Validation | 2,134 (15%) |
| Test | 2,134 (15%) |

## Classes
3-way sentiment label (`airline_sentiment` column):

| Class | Count (raw) | Share |
|---|---|---|
| negative | 9,178 | 62.7% |
| neutral | 3,099 | 21.2% |
| positive | 2,363 | 16.1% |

Negative tweets are additionally tagged with a free-text `negativereason` (e.g. "Customer Service Issue", "Late Flight", "Can't Tell", "Cancelled Flight", "Lost Luggage") — not used as a model input in this project, but explored in `notebooks/01_exploratory_data_analysis.ipynb` and the Streamlit Dataset Explorer for business context.

## Airlines Covered

| Airline | Tweets |
|---|---|
| United | 3,822 |
| US Airways | 2,913 |
| American | 2,759 |
| Southwest | 2,420 |
| Delta | 2,222 |
| Virgin America | 504 |

## Collection Method
Tweets mentioning the six airlines above were scraped from Twitter between **2015-02-16 and 2015-02-24**, then sentiment-labeled by human contributors via Crowdflower's crowdsourced annotation platform. Contributors first classified each tweet as positive, negative, or neutral, then (for negative tweets only) categorized the complaint reason.

## Columns Used by This Project

| Column | Used as |
|---|---|
| `text` | model input (raw tweet text, before cleaning) |
| `airline_sentiment` | label |
| `airline` | exploratory analysis, Dataset Explorer breakdown (not a model feature) |
| `negativereason` | exploratory analysis only |
| `tweet_id`, `retweet_count`, `tweet_created` | validation / bookkeeping only |

Columns present in the raw file but unused: `airline_sentiment_confidence`, `negativereason_confidence`, `airline_sentiment_gold`, `name`, `negativereason_gold`, `tweet_coord`, `tweet_location`, `user_timezone`.

## Preprocessing Performed (see `src/data/preprocessing.py`, `src/data/text_cleaning.py`)
1. Schema and content validation (required columns present, label set matches the 3 expected classes, empty-text ratio below 1%).
2. Text cleaning: lowercase, strip URLs, strip `@mentions` (airline handles), drop the `#` symbol but keep hashtag words, collapse elongated characters ("soooo" → "soo"), strip remaining punctuation/emoji, collapse whitespace.
3. Drop rows that become empty after cleaning.
4. Deduplicate exact `(clean_text, label)` pairs (mostly retweets).
5. Stratified 70/15/15 train/validation/test split (fixed seed, preserves class balance in every split).
6. Vocabulary (LSTM) and TF-IDF vectorizer (baseline) are fit **only** on the training split.

## Known Limitations / Potential Bias
- **Temporal narrowness**: all tweets are from an 8-day window in February 2015. Language, slang, airline policies, and even airline names/mergers have changed materially since; the model's applicability to current airline social media is not validated.
- **Platform bias**: Twitter users skew toward a subset of the traveling public (younger, more digitally engaged, often more likely to complain publicly than to praise) — visible in the ~63% negative-tweet skew, which likely reflects who chooses to tweet at an airline, not the true sentiment distribution of all customers.
- **Label subjectivity**: sentiment labeling, especially the negative/neutral boundary, is inherently subjective; the dataset provides only a single majority label per tweet, not full annotator agreement scores for most rows.
- **No demographic data**: the dataset contains no protected attributes (race, gender, age, etc.), which limits fairness-auditing scope but also means such attributes cannot be a bias vector in the model's ability to have learned them, by construction.
- **Airline representation is uneven** (Virgin America has ~6x fewer tweets than United) — performance may vary by airline; not separately validated per-airline in this project.

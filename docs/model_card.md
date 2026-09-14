# Model Card — Airline Sentiment Bidirectional LSTM

## Model Purpose
Classify short customer-feedback text (originally: airline-related tweets) into **negative**, **neutral**, or **positive** sentiment, to support triage of customer-support/social-media queues.

## Intended Use
- **Intended**: decision-*support* aid — surfacing and prioritizing tweets for a human support team to review, and providing a rough real-time sentiment signal for dashboards.
- **Not intended**: fully automated customer response, moderation, or any decision with a material consequence for an individual made without human review. The model has not been validated for use outside short, informal, English-language, airline-related social-media text.

## Model Details
- Architecture: Embedding (128-dim) → Bidirectional LSTM (96 units/direction) → Dense(64, ReLU) → Dense(3, softmax)
- Framework: TensorFlow / Keras
- Parameters: 1,590,147
- Input: raw text, cleaned via `src/data/text_cleaning.py`, tokenized to a 40-token sequence via `src/features/vocabulary.py`
- Output: probability distribution over {negative, neutral, positive}

## Training Data
Twitter US Airline Sentiment dataset — see `DATASET.md` for full provenance. 9,958 training tweets after cleaning/deduplication and a stratified 70/15/15 split.

## Evaluation Results (held-out test set, n=2,134)
| Metric | Value |
|---|---|
| Accuracy | 79.4% |
| Macro F1 | 0.735 |
| Macro Precision | 0.722 |
| Macro Recall | 0.752 |

Full breakdown, confusion matrix, and comparison against a classical baseline: `docs/evaluation.md`.

**Important**: on this dataset size, a much simpler TF-IDF + Logistic Regression baseline performs comparably (79.9% accuracy, 0.740 macro F1) — see `docs/evaluation.md` for the full discussion. This model is not shown to be clearly superior to the classical alternative at current data volume.

## Limitations
- Trained on an 8-day window of February-2015 airline tweets; temporal and domain generalization is unvalidated.
- 3-way sentiment is coarse and does not capture sarcasm, mixed sentiment within one tweet, or urgency.
- Minority classes (`neutral`, `positive`) have lower per-class F1 (0.62 and 0.71 respectively) than the majority `negative` class (0.88) — expect more errors on non-negative feedback.
- Not evaluated for performance differences across the six airlines individually.

## Known Biases / Risks
- The training data's ~63% negative-tweet skew reflects who chooses to publicly tweet at an airline (more often complainers than satisfied customers), not necessarily the sentiment distribution of the full customer base. A downstream dashboard built on this model's outputs could overstate how negative overall customer sentiment is if this skew isn't accounted for.
- Airline `@mentions` are stripped from model input specifically to reduce the risk of the model learning "which airline is mentioned" as a shortcut proxy for sentiment, rather than genuine sentiment language.
- No demographic attributes are present in the training data, limiting both the risk and the auditability of demographic bias.

## Ethical Considerations
- This model should not be used as the sole basis for any action affecting an individual customer (e.g., account restrictions, prioritization that delays a legitimate complaint) without human review.
- If deployed on live social data, teams should monitor for distribution shift (new slang, new airlines, events like widespread flight disruptions that change the base rate and *type* of complaints) and re-evaluate/re-train periodically rather than assuming 2015-era performance holds indefinitely.

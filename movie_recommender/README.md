# Movie Recommendation System

A self-contained Python project implementing **content-based filtering**,
**collaborative filtering** (item-based + matrix factorization), and a
**hybrid** blend of both — plus a synthetic dataset generator so the whole
thing runs out of the box with no external downloads.

## Project structure

```
movie_recommender/
├── data/
│   ├── generate_data.py       # creates movies.csv + ratings.csv
│   ├── movies.csv              # generated: movie_id, title, genres, year
│   └── ratings.csv             # generated: user_id, movie_id, rating, timestamp
├── content_based.py            # TF-IDF over genres + cosine similarity
├── collaborative_filtering.py  # item-based CF + SVD matrix factorization
├── hybrid.py                   # blends content-based and collaborative scores
├── demo.py                     # CLI demo tying everything together
├── train_and_save.py           # trains all models once, pickles them
├── load_model.py               # loads the pickle, no retraining needed
├── trained_models.pkl          # pre-trained model bundle, ready to use
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python data/generate_data.py   # generates the dataset (only needed once)
```

## Quick start — use the pre-trained model (fastest)

A trained model bundle (`trained_models.pkl`) is already included, so you
can skip training entirely:

```bash
python load_model.py --user 7 --n 10 --method hybrid
# --method options: content | collaborative | hybrid
```

## Retrain from scratch

If you swap in your own data or change model parameters, regenerate the
pickle:

```bash
python train_and_save.py       # fits all models, writes trained_models.pkl
```

## Run the full demo (all methods side by side, no pickle needed)

```bash
python demo.py                 # random user
python demo.py --user 7 --n 10 # specific user, 10 recommendations
```

Sample output:

```
=== Content-Based Recommendations for User 1 ===
 movie_id             title           genres    score
        4  Beyond the Glass           Comedy 0.534230
       18 Journey to Empire    Action|Comedy 0.331847
       ...

=== Hybrid Recommendations for User 1 ===
 movie_id              title           genres  hybrid_score
       36   The Broken Storm   Comedy|Mystery      0.782627
       ...
```

## How each method works

### 1. Content-based filtering (`content_based.py`)
- Each movie's genre list is treated as a "document" and turned into a
  TF-IDF vector, so rarer genre combinations weigh more heavily than
  common ones.
- A user's taste profile is the rating-weighted average of the vectors
  of movies they've already rated (ratings are centered around 3, so
  disliked movies push the profile *away* from their genres).
- Recommendations are the unseen movies with the highest cosine
  similarity to that profile.
- Works even for users with very few ratings, and doesn't need any
  other users' data at all.

### 2. Collaborative filtering (`collaborative_filtering.py`)
Two variants, both using only the user-item ratings matrix (no genre data):

- **Item-based neighborhood method** — mean-centers each movie's ratings,
  computes cosine similarity between movies based on rating patterns, and
  predicts a user's rating for an unseen movie as a similarity-weighted
  average of their ratings on similar movies.
- **Matrix factorization (SVD)** — decomposes the mean-centered
  user-item matrix into low-rank latent factors via truncated SVD, then
  reconstructs predicted ratings from those factors. This captures
  latent "taste dimensions" that go beyond pairwise item similarity and
  tends to generalize better as the dataset grows.

### 3. Hybrid (`hybrid.py`)
Normalizes content-based similarity scores and SVD predicted ratings to
a common [0, 1] scale, then combines them with configurable weights
(default: 60% collaborative, 40% content). This mitigates the classic
cold-start weakness of pure collaborative filtering while still
benefiting from the crowd signal when it's available.

## Using your own data

Replace `data/movies.csv` and `data/ratings.csv` with real data (e.g. the
[MovieLens dataset](https://grouplens.org/datasets/movielens/)) as long as
the columns match:

- `movies.csv`: `movie_id, title, genres, year` (genres pipe-separated, e.g. `Action|Comedy`)
- `ratings.csv`: `user_id, movie_id, rating, timestamp`

Everything else works unchanged.

## Extending this project
- Swap the TF-IDF genre vectors for embeddings of plot summaries or
  cast/crew data for richer content-based similarity.
- Add implicit feedback (watches, clicks) alongside explicit ratings.
- Add evaluation: hold out a test set of ratings and compute RMSE/MAE
  or precision@k for each method.
- Wrap `HybridRecommender` in a small Flask/FastAPI service for a real
  API endpoint.

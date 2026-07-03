"""
train_and_save.py
------------------
Fits the content-based, collaborative-filtering, and hybrid recommenders
on the dataset and pickles them to disk, so they can be loaded instantly
elsewhere without retraining.

Usage:
    python train_and_save.py
    # writes: trained_models.pkl

Then, to load and use later:
    import pickle
    with open("trained_models.pkl", "rb") as f:
        bundle = pickle.load(f)

    hybrid_model = bundle["hybrid"]
    recs = hybrid_model.recommend(user_id=7, n=5)
"""

import pickle
from pathlib import Path

import pandas as pd

from content_based import ContentBasedRecommender
from collaborative_filtering import CollaborativeFilteringRecommender
from hybrid import HybridRecommender

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_PATH = Path(__file__).parent / "trained_models.pkl"


def load_data():
    movies_path = DATA_DIR / "movies.csv"
    ratings_path = DATA_DIR / "ratings.csv"

    if not movies_path.exists() or not ratings_path.exists():
        print("Dataset not found — generating synthetic data first...")
        from data.generate_data import main as generate
        generate()

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings


def main():
    movies, ratings = load_data()

    print("Training content-based model...")
    content_model = ContentBasedRecommender(movies, ratings)

    print("Training collaborative filtering model (item-based + SVD)...")
    collab_model = CollaborativeFilteringRecommender(movies, ratings)

    print("Training hybrid model...")
    hybrid_model = HybridRecommender(movies, ratings)

    bundle = {
        "content": content_model,
        "collaborative": collab_model,
        "hybrid": hybrid_model,
        "movies": movies,
        "ratings": ratings,
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nSaved trained models to: {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")

    # quick sanity check
    sample_user = int(ratings["user_id"].iloc[0])
    print(f"\nSanity check — hybrid recommendations for user {sample_user}:")
    print(hybrid_model.recommend(sample_user, n=5).to_string(index=False))


if __name__ == "__main__":
    main()

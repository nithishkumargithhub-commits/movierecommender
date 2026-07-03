"""
demo.py
-------
End-to-end demo of the movie recommendation system.

Usage:
    python demo.py                # runs on a random sample user
    python demo.py --user 7       # runs on a specific user_id
    python demo.py --n 10         # number of recommendations to show
"""

import argparse
import random
from pathlib import Path

import pandas as pd

from content_based import ContentBasedRecommender
from collaborative_filtering import CollaborativeFilteringRecommender
from hybrid import HybridRecommender

DATA_DIR = Path(__file__).parent / "data"


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


def show_user_profile(user_id, movies, ratings):
    user_ratings = ratings[ratings["user_id"] == user_id].merge(movies, on="movie_id")
    user_ratings = user_ratings.sort_values("rating", ascending=False)
    print(f"\n=== User {user_id}'s existing ratings (top 8) ===")
    print(user_ratings[["title", "genres", "rating"]].head(8).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Movie recommendation demo")
    parser.add_argument("--user", type=int, default=None, help="user_id to generate recommendations for")
    parser.add_argument("--n", type=int, default=5, help="number of recommendations")
    args = parser.parse_args()

    movies, ratings = load_data()

    user_id = args.user or int(random.choice(ratings["user_id"].unique()))
    n = args.n

    print(f"Loaded {len(movies)} movies and {len(ratings)} ratings across "
          f"{ratings['user_id'].nunique()} users.")

    show_user_profile(user_id, movies, ratings)

    content_model = ContentBasedRecommender(movies, ratings)
    print(f"\n=== Content-Based Recommendations for User {user_id} ===")
    print(content_model.recommend(user_id, n=n).to_string(index=False))

    collab_model = CollaborativeFilteringRecommender(movies, ratings)
    print(f"\n=== Collaborative Filtering (Item-Based) for User {user_id} ===")
    print(collab_model.recommend_item_based(user_id, n=n).to_string(index=False))

    print(f"\n=== Collaborative Filtering (Matrix Factorization/SVD) for User {user_id} ===")
    print(collab_model.recommend_svd(user_id, n=n).to_string(index=False))

    hybrid_model = HybridRecommender(movies, ratings)
    print(f"\n=== Hybrid Recommendations for User {user_id} ===")
    print(hybrid_model.recommend(user_id, n=n).to_string(index=False))

    # bonus: pure item-similarity demo ("more like this movie")
    sample_movie_id = int(movies["movie_id"].iloc[0])
    sample_title = movies.loc[movies["movie_id"] == sample_movie_id, "title"].values[0]
    print(f"\n=== Movies Similar to '{sample_title}' (content-based) ===")
    print(content_model.similar_movies(sample_movie_id, n=n).to_string(index=False))


if __name__ == "__main__":
    main()

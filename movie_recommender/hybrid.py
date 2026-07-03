"""
hybrid.py
---------
Simple hybrid recommender: blends content-based similarity scores with
collaborative-filtering (SVD) predicted ratings. Useful because:
  - Collaborative filtering alone struggles for users/movies with few
    ratings (cold start).
  - Content-based alone ignores the "wisdom of the crowd" signal.

Both score types are min-max normalized to [0, 1] before blending so
neither dominates just because of its native scale.
"""

import pandas as pd

from content_based import ContentBasedRecommender
from collaborative_filtering import CollaborativeFilteringRecommender


def _normalize(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    lo, hi = series.min(), series.max()
    if hi == lo:
        return series.apply(lambda _: 0.5)
    return (series - lo) / (hi - lo)


class HybridRecommender:
    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame,
                 content_weight: float = 0.4, collab_weight: float = 0.6):
        self.movies = movies_df
        self.ratings = ratings_df
        self.content_weight = content_weight
        self.collab_weight = collab_weight

        self.content_model = ContentBasedRecommender(movies_df, ratings_df)
        self.collab_model = CollaborativeFilteringRecommender(movies_df, ratings_df)

    def recommend(self, user_id: int, n: int = 5, pool_size: int = 20) -> pd.DataFrame:
        content_scores = self.content_model.recommend(user_id, n=pool_size)
        collab_scores = self.collab_model.recommend_svd(user_id, n=pool_size)

        if content_scores.empty and collab_scores.empty:
            return pd.DataFrame(columns=["movie_id", "title", "genres", "hybrid_score"])

        merged = pd.merge(
            content_scores[["movie_id", "title", "genres", "score"]].rename(
                columns={"score": "content_score"}),
            collab_scores[["movie_id", "predicted_rating"]].rename(
                columns={"predicted_rating": "collab_score"}),
            on="movie_id", how="outer"
        )

        # fill in missing title/genres from movies table if a row only came
        # from one of the two source frames
        merged = merged.merge(self.movies[["movie_id", "title", "genres"]],
                               on="movie_id", how="left", suffixes=("", "_full"))
        merged["title"] = merged["title"].fillna(merged["title_full"])
        merged["genres"] = merged["genres"].fillna(merged["genres_full"])
        merged = merged.drop(columns=["title_full", "genres_full"])

        merged["content_score"] = merged["content_score"].fillna(0)
        merged["collab_score"] = merged["collab_score"].fillna(merged["collab_score"].mean())

        merged["content_norm"] = _normalize(merged["content_score"])
        merged["collab_norm"] = _normalize(merged["collab_score"])

        merged["hybrid_score"] = (
            self.content_weight * merged["content_norm"] +
            self.collab_weight * merged["collab_norm"]
        )

        result = merged.sort_values("hybrid_score", ascending=False).head(n)
        return result[["movie_id", "title", "genres", "hybrid_score"]].reset_index(drop=True)

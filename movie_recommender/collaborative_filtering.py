"""
collaborative_filtering.py
---------------------------
Two collaborative filtering approaches, both built on the user-item
ratings matrix (no movie metadata used at all — recommendations come
purely from the pattern of who-rated-what):

1. Item-based CF (neighborhood method)
   - Build a user-item matrix.
   - Mean-center each item's ratings (removes "this movie is just
     generally liked/disliked" bias).
   - Compute cosine similarity between item vectors.
   - Predict a user's rating for an unseen movie as a similarity-
     weighted average of their ratings for similar movies.

2. Matrix factorization (SVD / latent factors)
   - Decompose the user-item matrix into low-rank user and item
     factor matrices via truncated SVD on the mean-centered matrix.
   - Predict ratings by reconstructing the matrix from the low-rank
     factors, which captures latent "taste dimensions" that go beyond
     simple item-to-item similarity.

Both methods return top-N unseen movies ranked by predicted rating.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds


class CollaborativeFilteringRecommender:
    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.movies = movies_df
        self.ratings = ratings_df

        self.user_item = ratings_df.pivot_table(
            index="user_id", columns="movie_id", values="rating"
        )
        self.user_ids = self.user_item.index.to_numpy()
        self.movie_ids = self.user_item.columns.to_numpy()

        # Fill missing with 0 for matrix math, but keep a mask of what's real
        self.filled = self.user_item.fillna(0).to_numpy()
        self.rated_mask = ~self.user_item.isna().to_numpy()

        self._fit_item_based()
        self._fit_svd()

    # ---------- Item-based CF ----------
    def _fit_item_based(self):
        # Mean-center each item (column) using only its actual ratings
        col_means = np.array([
            self.filled[:, j][self.rated_mask[:, j]].mean() if self.rated_mask[:, j].any() else 0
            for j in range(self.filled.shape[1])
        ])
        self.item_means = col_means

        centered = np.where(self.rated_mask, self.filled - col_means, 0)
        # Item-item similarity matrix
        self.item_sim = cosine_similarity(centered.T)
        np.fill_diagonal(self.item_sim, 0)  # a movie is not its own neighbor

    def predict_item_based(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.user_item.index or movie_id not in self.user_item.columns:
            return np.nan

        u_idx = self.user_item.index.get_loc(user_id)
        m_idx = self.user_item.columns.get_loc(movie_id)

        sims = self.item_sim[m_idx]
        user_ratings = self.filled[u_idx]
        user_rated = self.rated_mask[u_idx]

        num = np.sum(sims[user_rated] * (user_ratings[user_rated] - self.item_means[user_rated]))
        den = np.sum(np.abs(sims[user_rated]))

        if den == 0:
            return self.item_means[m_idx]
        return self.item_means[m_idx] + num / den

    def recommend_item_based(self, user_id: int, n: int = 5) -> pd.DataFrame:
        if user_id not in self.user_item.index:
            return pd.DataFrame(columns=["movie_id", "title", "genres", "predicted_rating"])

        u_idx = self.user_item.index.get_loc(user_id)
        unrated_movie_ids = self.movie_ids[~self.rated_mask[u_idx]]

        preds = [(mid, self.predict_item_based(user_id, mid)) for mid in unrated_movie_ids]
        preds.sort(key=lambda x: x[1], reverse=True)
        top = preds[:n]

        return self._format_predictions(top)

    # ---------- Matrix factorization (SVD) ----------
    def _fit_svd(self, k: int = 10):
        k = min(k, min(self.filled.shape) - 1)
        k = max(k, 1)

        global_mean = self.filled[self.rated_mask].mean() if self.rated_mask.any() else 0
        self.global_mean = global_mean

        centered = np.where(self.rated_mask, self.filled - global_mean, 0)

        U, sigma, Vt = svds(centered, k=k)
        sigma = np.diag(sigma)
        self.svd_predictions = U @ sigma @ Vt + global_mean

    def recommend_svd(self, user_id: int, n: int = 5) -> pd.DataFrame:
        if user_id not in self.user_item.index:
            return pd.DataFrame(columns=["movie_id", "title", "genres", "predicted_rating"])

        u_idx = self.user_item.index.get_loc(user_id)
        preds_row = self.svd_predictions[u_idx]
        unrated = ~self.rated_mask[u_idx]

        candidates = list(zip(self.movie_ids[unrated], preds_row[unrated]))
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:n]

        return self._format_predictions(top)

    # ---------- helpers ----------
    def _format_predictions(self, id_score_pairs):
        rows = []
        for movie_id, score in id_score_pairs:
            title_row = self.movies.loc[self.movies["movie_id"] == movie_id]
            if title_row.empty:
                continue
            rows.append({
                "movie_id": movie_id,
                "title": title_row["title"].values[0],
                "genres": title_row["genres"].values[0],
                "predicted_rating": round(float(score), 3)
            })
        return pd.DataFrame(rows)

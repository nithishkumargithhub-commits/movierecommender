"""
content_based.py
-----------------
Content-based filtering recommender.

Approach:
  1. Represent each movie as a TF-IDF vector over its genre tags
     (genres are treated as "words", so movies sharing rarer genre
     combinations are considered more similar).
  2. Build a user profile vector as the rating-weighted average of the
     TF-IDF vectors of movies the user has already rated.
  3. Recommend unseen movies whose vectors are most similar (cosine
     similarity) to the user's profile.

This does NOT require any other users' data — it works from a single
user's own ratings plus movie metadata, so it handles new users with
at least one rating and works well for the "cold start on collaborative
filtering" problem.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.movies = movies_df.reset_index(drop=True)
        self.ratings = ratings_df

        # Turn "Action|Sci-Fi|Thriller" into "Action Sci-Fi Thriller" so
        # TfidfVectorizer treats each genre as a token.
        genre_corpus = self.movies["genres"].str.replace("|", " ", regex=False)

        self.vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")
        self.movie_vectors = self.vectorizer.fit_transform(genre_corpus)

        self._movie_id_to_row = {
            mid: i for i, mid in enumerate(self.movies["movie_id"])
        }

    def _user_profile_vector(self, user_id: int):
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        if user_ratings.empty:
            return None

        rows, weights = [], []
        for _, r in user_ratings.iterrows():
            row = self._movie_id_to_row.get(r["movie_id"])
            if row is not None:
                rows.append(row)
                # center ratings around 3 so low ratings push the profile away
                weights.append(r["rating"] - 3.0)

        if not rows:
            return None

        weights = np.array(weights)
        vectors = self.movie_vectors[rows]

        # If every weight is <= 0 (user disliked everything they rated),
        # fall back to unweighted average to avoid a degenerate zero vector.
        if np.all(weights <= 0):
            weights = np.ones_like(weights)

        weighted = vectors.multiply(weights.reshape(-1, 1))
        profile = np.asarray(weighted.sum(axis=0))
        return profile

    def recommend(self, user_id: int, n: int = 5) -> pd.DataFrame:
        profile = self._user_profile_vector(user_id)
        if profile is None:
            return pd.DataFrame(columns=["movie_id", "title", "genres", "score"])

        sims = cosine_similarity(profile, self.movie_vectors).flatten()

        already_rated = set(
            self.ratings.loc[self.ratings["user_id"] == user_id, "movie_id"]
        )

        result = self.movies.copy()
        result["score"] = sims
        result = result[~result["movie_id"].isin(already_rated)]
        result = result.sort_values("score", ascending=False).head(n)
        return result[["movie_id", "title", "genres", "score"]].reset_index(drop=True)

    def similar_movies(self, movie_id: int, n: int = 5) -> pd.DataFrame:
        """Given a movie, find other movies with similar genre profiles."""
        row = self._movie_id_to_row.get(movie_id)
        if row is None:
            return pd.DataFrame(columns=["movie_id", "title", "genres", "score"])

        sims = cosine_similarity(self.movie_vectors[row], self.movie_vectors).flatten()
        result = self.movies.copy()
        result["score"] = sims
        result = result[result["movie_id"] != movie_id]
        result = result.sort_values("score", ascending=False).head(n)
        return result[["movie_id", "title", "genres", "score"]].reset_index(drop=True)

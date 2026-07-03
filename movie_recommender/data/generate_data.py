"""
generate_data.py
-----------------
Creates a synthetic-but-realistic MovieLens-style dataset:
  - movies.csv  : movie_id, title, genres, year
  - ratings.csv : user_id, movie_id, rating, timestamp

The data is generated with structure (users have genre preferences)
so that recommendations produced downstream are meaningful and not
just random noise.

Run directly to (re)generate the CSV files:
    python generate_data.py
"""

import csv
import random
import time
from pathlib import Path

random.seed(42)

GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Drama", "Fantasy", "Horror", "Mystery", "Romance",
    "Sci-Fi", "Thriller"
]

TITLE_WORDS_A = [
    "The Last", "Shadow of", "Rise of", "Beyond the", "Edge of",
    "City of", "Legend of", "Whispers in", "The Hidden", "Echoes of",
    "The Silent", "Age of", "The Broken", "Journey to", "The Forgotten"
]
TITLE_WORDS_B = [
    "Horizon", "Empire", "Stars", "Shadows", "Kingdom", "Dawn",
    "Storm", "Fire", "Ocean", "Night", "Silence", "Dreams",
    "Tomorrow", "Legacy", "Fury", "Winter", "Glass", "Steel"
]


def make_movies(n_movies=60):
    movies = []
    used_titles = set()
    for movie_id in range(1, n_movies + 1):
        # each movie gets 1-3 genres, weighted so genre combos feel plausible
        n_genres = random.choice([1, 1, 2, 2, 3])
        genres = random.sample(GENRES, n_genres)

        while True:
            title = f"{random.choice(TITLE_WORDS_A)} {random.choice(TITLE_WORDS_B)}"
            if title not in used_titles:
                used_titles.add(title)
                break

        year = random.randint(1985, 2025)
        movies.append({
            "movie_id": movie_id,
            "title": title,
            "genres": "|".join(genres),
            "year": year
        })
    return movies


def make_users_with_preferences(n_users=40):
    """Give each synthetic user 1-3 favorite genres that drive their ratings."""
    users = {}
    for user_id in range(1, n_users + 1):
        n_pref = random.choice([1, 2, 2, 3])
        users[user_id] = set(random.sample(GENRES, n_pref))
    return users


def make_ratings(movies, user_prefs, ratings_per_user=(15, 35)):
    ratings = []
    base_ts = int(time.time()) - 60 * 60 * 24 * 365  # ~1 year ago

    for user_id, prefs in user_prefs.items():
        n_ratings = random.randint(*ratings_per_user)
        rated_movies = random.sample(movies, min(n_ratings, len(movies)))

        for movie in rated_movies:
            movie_genres = set(movie["genres"].split("|"))
            overlap = len(movie_genres & prefs)

            # Rating distribution shaped by genre overlap with user preference
            if overlap >= 2:
                rating = random.choices([3, 4, 5], weights=[1, 3, 6])[0]
            elif overlap == 1:
                rating = random.choices([2, 3, 4, 5], weights=[1, 3, 4, 2])[0]
            else:
                rating = random.choices([1, 2, 3, 4], weights=[4, 4, 2, 1])[0]

            timestamp = base_ts + random.randint(0, 60 * 60 * 24 * 365)
            ratings.append({
                "user_id": user_id,
                "movie_id": movie["movie_id"],
                "rating": rating,
                "timestamp": timestamp
            })
    return ratings


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    here = Path(__file__).parent
    movies = make_movies(n_movies=60)
    user_prefs = make_users_with_preferences(n_users=40)
    ratings = make_ratings(movies, user_prefs)

    write_csv(here / "movies.csv", movies, ["movie_id", "title", "genres", "year"])
    write_csv(here / "ratings.csv", ratings, ["user_id", "movie_id", "rating", "timestamp"])

    print(f"Generated {len(movies)} movies and {len(ratings)} ratings.")
    print(f"Files written to: {here}")


if __name__ == "__main__":
    main()

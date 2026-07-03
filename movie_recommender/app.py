"""
Streamlit web app for the movie recommendation system.
A simple, user-friendly interface for discovering movies.
Run with: streamlit run app.py
"""

import random
from pathlib import Path

import pandas as pd
import streamlit as st

from content_based import ContentBasedRecommender
from hybrid import HybridRecommender

DATA_DIR = Path(__file__).parent / "data"


def load_data():
    """Load or generate movie and ratings data."""
    movies_path = DATA_DIR / "movies.csv"
    ratings_path = DATA_DIR / "ratings.csv"

    if not movies_path.exists() or not ratings_path.exists():
        from data.generate_data import main as generate
        generate()

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings


@st.cache_resource
def get_models(movies, ratings):
    """Cache the models to avoid retraining on every interaction."""
    return {
        "hybrid": HybridRecommender(movies, ratings),
        "content": ContentBasedRecommender(movies, ratings),
    }


def get_all_genres(movies_df):
    """Extract all unique genres from movies."""
    all_genres = set()
    for genres_str in movies_df["genres"].dropna():
        all_genres.update([g.strip() for g in str(genres_str).split("|")])
    return sorted(list(all_genres))


def search_movies_by_title(movies_df, query):
    """Search movies by title (case-insensitive)."""
    if not query:
        return pd.DataFrame()
    return movies_df[movies_df["title"].str.contains(query, case=False, na=False)]


def filter_by_genres(movies_df, selected_genres):
    """Filter movies by one or more selected genres."""
    if not selected_genres:
        return movies_df
    
    filtered = movies_df[movies_df["genres"].fillna("").apply(
        lambda x: any(genre in str(x).split("|") for genre in selected_genres)
    )]
    return filtered


def filter_by_rating(movies_df, ratings_df, min_rating):
    """Filter movies by minimum average rating from ratings data."""
    if min_rating == 0:
        return movies_df
    
    avg_ratings = ratings_df.groupby("movie_id")["rating"].mean()
    valid_movies = avg_ratings[avg_ratings >= min_rating].index.tolist()
    return movies_df[movies_df["movie_id"].isin(valid_movies)]


def main():
    st.set_page_config(page_title="🎬 Movie Recommender", layout="centered", initial_sidebar_state="collapsed")
    
    # Custom CSS for a cleaner, simpler design
    st.markdown("""
    <style>
        .main {
            padding: 2rem 1rem;
            max-width: 1000px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            text-align: center;
            color: #666;
            font-size: 1.1em;
            margin-bottom: 2rem;
        }
        .action-buttons {
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin: 2rem 0;
        }
        .movie-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .search-section {
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 1rem;
            margin: 2rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Load data
    movies, ratings = load_data()
    models = get_models(movies, ratings)

    # ===== HEADER =====
    st.title("🎬 Movie Recommender")
    st.markdown("### Find movies you'll love")
    
    # ===== SIMPLE MODE SELECTION =====
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        mode = st.radio(
            "What would you like to do?",
            ["Get Recommendations", "Search Movies"],
            label_visibility="collapsed"
        )
    
    with col2:
        st.metric("Total Movies", len(movies), label_visibility="collapsed")
        st.metric("Users", ratings["user_id"].nunique(), label_visibility="collapsed")
    
    st.markdown("---")
    
    # ===== MODE 1: GET RECOMMENDATIONS =====
    if mode == "Get Recommendations":
        st.header("🎯 Get Personalized Recommendations")
        
        # Simple user selection
        user_id = st.selectbox(
            "Select a user (or random):",
            options=sorted(ratings["user_id"].unique()),
            index=0
        )
        
        n_recs = st.select_slider("How many movies?", options=range(1, 11), value=5)
        
        # Filters
        st.subheader("🔎 Filter Options")
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            available_genres = get_all_genres(movies)
            selected_genres = st.multiselect("Filter by genres:", available_genres, key="rec_genres")
        
        with filter_col2:
            min_rating_rec = st.slider("Minimum rating:", min_value=0.0, max_value=5.0, step=0.5, value=0.0, key="rec_rating")
        
        # Show user's watch history
        user_ratings = ratings[ratings["user_id"] == user_id].merge(movies, on="movie_id")
        user_ratings = user_ratings.sort_values("rating", ascending=False)
        
        if len(user_ratings) > 0:
            st.subheader(f"📽️ Movies User {user_id} liked:")
            
            cols = st.columns(min(3, len(user_ratings)))
            for idx, (_, movie) in enumerate(user_ratings.head(3).iterrows()):
                with cols[idx % 3]:
                    st.write(f"⭐ **{movie['title']}**")
                    st.caption(f"{movie['genres']}")
        
        # Get recommendations and apply filters
        st.subheader("💡 We recommend:")
        
        try:
            recs = models["hybrid"].recommend(user_id, n=n_recs)
            
            if not recs.empty:
                # Apply genre filter
                if selected_genres:
                    recs = filter_by_genres(recs, selected_genres)
                
                # Apply rating filter
                if min_rating_rec > 0:
                    recs = filter_by_rating(recs, ratings, min_rating_rec)
                
                if not recs.empty:
                    for idx, (_, row) in enumerate(recs.iterrows(), 1):
                        col1, col2 = st.columns([0.15, 0.85])
                        with col1:
                            st.write(f"**#{idx}**")
                        with col2:
                            st.write(f"**{row['title']}**")
                            st.caption(f"Genres: {row['genres']} • Score: {row.get('hybrid_score', 0):.2f}")
                else:
                    st.info("No recommendations match your filters. Try adjusting them!")
            else:
                st.info("No recommendations available for this user yet.")
        except Exception as e:
            st.error(f"Error getting recommendations: {str(e)}")
    
    # ===== MODE 2: SEARCH MOVIES =====
    else:
        st.header("🔍 Find Movies")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input("Search by title:", placeholder="e.g., 'Shadow', 'Fire'")
        
        with col2:
            st.write("")  # Spacing
            show_all = st.checkbox("Show all movies")
        
        # Filters for search
        st.subheader("🔎 Filter Options")
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            available_genres = get_all_genres(movies)
            selected_genres_search = st.multiselect("Filter by genres:", available_genres, key="search_genres")
        
        with filter_col2:
            min_rating_search = st.slider("Minimum rating:", min_value=0.0, max_value=5.0, step=0.5, value=0.0, key="search_rating")
        
        if search_query or show_all:
            if show_all:
                results = movies.sort_values("title")
            else:
                results = search_movies_by_title(movies, search_query)
            
            # Apply genre filter
            if selected_genres_search:
                results = filter_by_genres(results, selected_genres_search)
            
            # Apply rating filter
            if min_rating_search > 0:
                results = filter_by_rating(results, ratings, min_rating_search)
            
            st.caption(f"Found {len(results)} movie(s)")
            
            if not results.empty:
                for _, movie in results.iterrows():
                    col1, col2 = st.columns([0.8, 0.2])
                    with col1:
                        st.write(f"**{movie['title']}**")
                        st.caption(f"Genres: {movie['genres']}")
                    with col2:
                        if st.button("Similar", key=f"sim_{movie['movie_id']}", help="Find similar movies"):
                            st.session_state.show_similar = movie['movie_id']
            else:
                st.info("No movies found. Try adjusting your filters!")
        
        # Show similar movies if one is selected
        if "show_similar" in st.session_state:
            st.divider()
            st.subheader("🔗 Similar Movies")
            
            selected_id = st.session_state.show_similar
            similar = models["content"].similar_movies(selected_id, n=5)
            
            if not similar.empty:
                for _, row in similar.iterrows():
                    st.write(f"**{row['title']}**")
                    st.caption(f"Genres: {row['genres']} • Similarity: {row['score']:.2f}")
    
    # ===== FOOTER =====
    st.divider()
    st.markdown("**About:** Uses hybrid AI recommendations combining content-based and collaborative filtering")


if __name__ == "__main__":
    main()

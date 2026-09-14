"""
Movie Recommendation System Using User Watch History
B.Tech CSE (Data Science) Project
Author: College Student Project
Technologies: Python, Streamlit, Pandas, Matplotlib
"""

import os
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Data Loading and Preprocessing
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    """
    Loads movie and watch history CSV files and applies standard,
    beginner-friendly data cleaning operations.
    """
    movies_file = "movies.csv"
    history_file = "watch_history.csv"

    if not os.path.exists(movies_file) or not os.path.exists(history_file):
        st.error("Dataset files missing. Please ensure movies.csv and watch_history.csv exist.")
        st.stop()

    # Load datasets
    movies_df = pd.read_csv(movies_file)
    history_df = pd.read_csv(history_file)

    # Simple Preprocessing & Cleaning:
    # 1. Remove duplicate records
    movies_df = movies_df.drop_duplicates(subset=["Movie_ID"])
    history_df = history_df.drop_duplicates(subset=["User_ID", "Movie_ID"])

    # 2. Drop rows with missing critical IDs
    movies_df = movies_df.dropna(subset=["Movie_ID", "Movie_Title"])
    history_df = history_df.dropna(subset=["User_ID", "Movie_ID"])

    # 3. Ensure proper data types
    movies_df["Movie_ID"] = pd.to_numeric(movies_df["Movie_ID"], errors="coerce").astype("Int64")
    movies_df["Rating"] = pd.to_numeric(movies_df["Rating"], errors="coerce").fillna(5.0)
    if "Year" in movies_df.columns:
        movies_df["Year"] = pd.to_numeric(movies_df["Year"], errors="coerce").fillna(2000).astype(int)
    else:
        movies_df["Year"] = 2000

    # Fill missing genre if any
    movies_df["Genre"] = movies_df["Genre"].fillna("General")

    history_df["User_ID"] = pd.to_numeric(history_df["User_ID"], errors="coerce").astype("Int64")
    history_df["Movie_ID"] = pd.to_numeric(history_df["Movie_ID"], errors="coerce").astype("Int64")

    # Drop any leftover invalid rows
    movies_df = movies_df.dropna(subset=["Movie_ID"]).reset_index(drop=True)
    history_df = history_df.dropna(subset=["User_ID", "Movie_ID"]).reset_index(drop=True)

    # Convert ID columns to standard int for seamless matching
    movies_df["Movie_ID"] = movies_df["Movie_ID"].astype(int)
    history_df["User_ID"] = history_df["User_ID"].astype(int)
    history_df["Movie_ID"] = history_df["Movie_ID"].astype(int)

    return movies_df, history_df

# Helper function to parse genre string into a clean list
def extract_genres(genre_str):
    if not isinstance(genre_str, str):
        return []
    # Support both pipe '|' and comma ',' separated genres
    genres = genre_str.replace("|", ",").split(",")
    return [g.strip() for g in genres if g.strip()]

# -----------------------------------------------------------------------------
# 3. Recommendation Logic (Transparent Content-Based Scoring)
# -----------------------------------------------------------------------------
def get_user_preferred_genres(watched_movies_df):
    """
    Counts frequencies of genres among all movies watched by the user.
    Returns:
      genre_counts: Counter object {genre: frequency}
      preferred_genres_list: sorted list of distinct genres by popularity
    """
    genre_list = []
    for _, row in watched_movies_df.iterrows():
        genre_list.extend(extract_genres(row["Genre"]))
    genre_counts = Counter(genre_list)
    preferred_genres_list = [genre for genre, _ in genre_counts.most_common()]
    return genre_counts, preferred_genres_list

def calculate_recommendations(user_id, movies_df, history_df, top_n=5):
    """
    Executes the content-based recommendation steps:
    1. Identify watched movies for the user.
    2. Extract genres & build user genre preference weights.
    3. Exclude already watched movies from candidate pool.
    4. Compute transparent similarity score = Genre Match Score + Rating Contribution.
    5. Return top_n recommendations with human-readable reasons.
    """
    # Step 1: Watched movies
    user_history = history_df[history_df["User_ID"] == user_id]
    watched_ids = set(user_history["Movie_ID"].tolist())
    watched_movies = movies_df[movies_df["Movie_ID"].isin(watched_ids)].copy()

    if watched_movies.empty:
        return watched_movies, pd.DataFrame(), [], Counter()

    # Step 2: Extract genres & weights
    genre_counts, preferred_genres = get_user_preferred_genres(watched_movies)

    # Maximum frequency for normalization
    max_freq = max(genre_counts.values()) if genre_counts else 1

    # Step 3: Exclude already watched movies
    candidate_movies = movies_df[~movies_df["Movie_ID"].isin(watched_ids)].copy()

    if candidate_movies.empty:
        return watched_movies, pd.DataFrame(), preferred_genres, genre_counts

    # Step 4: Calculate similarity score for each candidate movie
    scores = []
    reasons = []
    genre_scores = []
    rating_scores = []

    for _, movie in candidate_movies.iterrows():
        movie_genres = extract_genres(movie["Genre"])
        
        # Calculate genre overlap score weighted by user frequency
        matched_genres = [g for g in movie_genres if g in genre_counts]
        genre_score = sum(genre_counts[g] / max_freq for g in matched_genres)

        # Rating contribution: normalized rating (rating / 10) scaled by 0.5
        rating_val = float(movie["Rating"])
        rating_contribution = round((rating_val / 10.0) * 0.5, 3)

        total_score = round(genre_score + rating_contribution, 3)

        # Generate a beginner-friendly explanation reason
        if matched_genres:
            top_matches = matched_genres[:2]
            reason = f"Recommended because it matches your preferred {' & '.join(top_matches)} genre{'s' if len(top_matches) > 1 else ''}."
        else:
            reason = f"Recommended based on its high overall audience rating ({rating_val}/10)."

        scores.append(total_score)
        reasons.append(reason)
        genre_scores.append(round(genre_score, 3))
        rating_scores.append(rating_contribution)

    candidate_movies["Genre_Score"] = genre_scores
    candidate_movies["Rating_Score"] = rating_scores
    candidate_movies["Similarity_Score"] = scores
    candidate_movies["Reason"] = reasons

    # Step 5: Sort by Similarity Score descending, then Rating descending
    ranked_candidates = candidate_movies.sort_values(
        by=["Similarity_Score", "Rating"],
        ascending=[False, False]
    ).head(top_n).reset_index(drop=True)

    return watched_movies, ranked_candidates, preferred_genres, genre_counts

# -----------------------------------------------------------------------------
# 4. Streamlit User Interface
# -----------------------------------------------------------------------------
def main():
    movies_df, history_df = load_and_clean_data()

    # Header section
    st.title("🎬 Movie Recommendation System")
    st.markdown("##### *Get personalized movie recommendations based on your previously watched movies.*")
    st.caption("B.Tech CSE (Data Science) Project • Content-Based Filtering Using User Watch History")
    st.markdown("---")

    # Sidebar: User Selection & Project Info
    st.sidebar.header("👤 Select User")
    all_users = sorted(history_df["User_ID"].unique().tolist())
    
    selected_user = st.sidebar.selectbox(
        "Choose a User ID to analyze:",
        options=all_users,
        format_func=lambda uid: f"User {uid}"
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **How it works:**\n"
        "1. Analyzes movies watched by the selected user.\n"
        "2. Identifies top genre preferences.\n"
        "3. Filters out movies already seen.\n"
        "4. Ranks unwatched movies by genre match & rating."
    )

    # Main Area: Split into two columns for layout
    col1, col2 = st.columns([1, 1])

    # Pre-calculate data for selected user
    watched_movies, recommendations, preferred_genres, genre_counts = calculate_recommendations(
        selected_user, movies_df, history_df, top_n=5
    )

    with col1:
        st.subheader(f"📺 Movies You Have Watched (User {selected_user})")
        if watched_movies.empty:
            st.warning("No watch history found for this user.")
        else:
            display_watched = watched_movies[["Movie_Title", "Genre", "Rating", "Year"]].copy()
            display_watched["Genre"] = display_watched["Genre"].str.replace("|", ", ")
            st.dataframe(
                display_watched.rename(columns={
                    "Movie_Title": "Title",
                    "Rating": "Rating (★/10)"
                }),
                use_container_width=True,
                hide_index=True
            )

        # Show user's preferred genres
        if preferred_genres:
            st.markdown(f"**Your preferred genres:** `{', '.join(preferred_genres[:4])}`")
            # Small pill breakdown of genre frequency
            genre_summary_text = " • ".join([f"{genre} ({count})" for genre, count in genre_counts.most_common(5)])
            st.caption(f"Watch count per genre: {genre_summary_text}")

    with col2:
        st.subheader("🎯 Generate Recommendations")
        st.write("Click below to analyze your watch history and find similar movies you haven't watched yet.")
        
        # State tracking for button click
        recommend_clicked = st.button("🚀 Recommend Movies", type="primary", use_container_width=True)

    st.markdown("---")

    # Recommendations Section (Triggered on click or session state)
    if recommend_clicked or "has_run" in st.session_state:
        st.session_state["has_run"] = True
        st.subheader(f"⭐ Recommended Movies for User {selected_user}")

        if recommendations.empty:
            st.info("No suitable new recommendations found. You might have watched all available movies in the catalog!")
        else:
            st.caption(f"Showing Top {len(recommendations)} personalized movie recommendations:")

            for idx, row in recommendations.iterrows():
                with st.container():
                    r_col1, r_col2 = st.columns([3, 1])
                    with r_col1:
                        genres_display = str(row['Genre']).replace('|', ', ')
                        st.markdown(f"### {idx + 1}. {row['Movie_Title']} ({row['Year']})")
                        st.markdown(f"**Genres:** {genres_display} &nbsp;|&nbsp; **Rating:** ⭐ {row['Rating']}/10")
                        st.markdown(f"💬 *{row['Reason']}*")
                    with r_col2:
                        st.metric(
                            label="Match Score",
                            value=f"{row['Similarity_Score']:.2f}",
                            help=f"Genre Match: {row['Genre_Score']} + Rating Contribution: {row['Rating_Score']}"
                        )
                    st.divider()

            if len(recommendations) < 5:
                st.warning(f"Note: Only {len(recommendations)} candidate movies matched the criteria after excluding previously watched movies.")

    # -------------------------------------------------------------------------
    # 5. Exploratory Data Analysis & Statistics Section
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.header("📊 Dataset Statistics & Exploratory Data Analysis")

    # High-level metric cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Users", len(all_users))
    m2.metric("Total Movies", len(movies_df))
    m3.metric("Watch History Records", len(history_df))
    m4.metric("Average Movie Rating", f"{movies_df['Rating'].mean():.2f} / 10")

    st.write("")

    # Visualizations using Matplotlib
    viz_col1, viz_col2, viz_col3 = st.columns(3)

    # Plot 1: Movie Rating Distribution
    with viz_col1:
        st.markdown("**1. Movie Rating Distribution**")
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        ax1.hist(movies_df["Rating"], bins=8, color="#3b82f6", edgecolor="black", alpha=0.8)
        ax1.set_xlabel("Rating (out of 10)")
        ax1.set_ylabel("Number of Movies")
        ax1.set_title("Distribution of Movie Ratings", fontsize=11)
        ax1.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig1)

    # Plot 2: Most Watched Movies in Watch History
    with viz_col2:
        st.markdown("**2. Most Popular Watched Movies**")
        pop_counts = history_df["Movie_ID"].value_counts().head(6)
        pop_df = pd.DataFrame({"Movie_ID": pop_counts.index, "Views": pop_counts.values})
        pop_df = pop_df.merge(movies_df[["Movie_ID", "Movie_Title"]], on="Movie_ID", how="left")

        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.barh(pop_df["Movie_Title"], pop_df["Views"], color="#10b981", edgecolor="black", alpha=0.8)
        ax2.set_xlabel("Watch Count")
        ax2.set_title("Most Watched Movies", fontsize=11)
        ax2.invert_yaxis()
        ax2.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig2)

    # Plot 3: Genre Distribution Across Entire Catalog
    with viz_col3:
        st.markdown("**3. Genre Distribution**")
        all_catalog_genres = []
        for g_str in movies_df["Genre"]:
            all_catalog_genres.extend(extract_genres(g_str))
        genre_dist = Counter(all_catalog_genres).most_common(7)
        g_names = [g[0] for g in genre_dist]
        g_vals = [g[1] for g in genre_dist]

        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.bar(g_names, g_vals, color="#f59e0b", edgecolor="black", alpha=0.8)
        ax3.set_ylabel("Count")
        ax3.set_title("Top Genres in Catalog", fontsize=11)
        ax3.tick_params(axis="x", rotation=45)
        ax3.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig3)

    # Footer note for academic submission
    st.caption("College Project: Movie Recommendation System Using User Watch History | B.Tech CSE Data Science")

if __name__ == "__main__":
    main()

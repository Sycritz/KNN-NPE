import pandas as pd
import numpy as np
from typing import Tuple
import streamlit as st


ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]


@st.cache_data
def encode_genres(movies_df: pd.DataFrame) -> pd.DataFrame:
    genre_encoded = movies_df["genres"].str.get_dummies(sep="|")
    for genre in ALL_GENRES:
        if genre not in genre_encoded.columns:
            genre_encoded[genre] = 0
    genre_encoded = genre_encoded[ALL_GENRES]
    return genre_encoded


@st.cache_data
def create_rating_features(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    movie_rating_stats = ratings_df.groupby("movie_id").agg({
        "rating": ["mean", "std", "count"]
    }).fillna(0)
    movie_rating_stats.columns = ["avg_rating", "rating_std", "rating_count"]
    movie_rating_stats = movie_rating_stats.reindex(movies_df["movie_id"], fill_value=0)
    movie_rating_stats = movie_rating_stats.fillna(0)
    return movie_rating_stats


@st.cache_data
def create_feature_matrix(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame
) -> Tuple[np.ndarray, pd.DataFrame]:
    genre_features = encode_genres(movies_df)
    rating_features = create_rating_features(ratings_df, movies_df)
    
    combined_features = pd.concat([
        genre_features,
        rating_features[["avg_rating", "rating_std", "rating_count"]]
    ], axis=1)
    
    combined_features = combined_features.fillna(0)
    feature_matrix = combined_features.values
    return feature_matrix, combined_features

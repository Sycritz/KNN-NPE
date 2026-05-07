import pandas as pd
from pathlib import Path
from typing import Tuple
import streamlit as st


@st.cache_data
def load_movies(data_path: Path) -> pd.DataFrame:
    movies_df = pd.read_csv(
        data_path / "movies.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["movie_id", "title", "genres"],
        header=None
    )
    return movies_df


@st.cache_data
def load_ratings(data_path: Path) -> pd.DataFrame:
    ratings_df = pd.read_csv(
        data_path / "ratings.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["user_id", "movie_id", "rating", "timestamp"],
        header=None
    )
    return ratings_df


@st.cache_data
def load_data(data_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    movies_df = load_movies(data_path)
    ratings_df = load_ratings(data_path)
    return movies_df, ratings_df

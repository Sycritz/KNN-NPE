import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from data_loader import load_data
from feature_engineering import create_feature_matrix
from utils import normalize_features
from model import KNNRecommender
from visualization import create_3d_scatter, get_pca_coordinates
from ui import render_sidebar, render_movie_card, render_theory_note


st.set_page_config(
    page_title="k-NN Movie Recommender",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
    }
    .stSidebar {
        background-color: #161b22;
    }
    .stSidebar [data-testid="stSidebar"] {
        background-color: #161b22;
    }
    h1, h2, h3, h4 {
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    p {
        color: #c9d1d9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stSlider > label > div {
        color: #ffffff;
    }
    .stSelectbox > label > div {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

st.title("k-NN Movie Recommendation System")
st.markdown("---")

data_path = Path(__file__).parent.parent / "Datasets" / "ml-1m"

movies_df, ratings_df = load_data(data_path)

feature_matrix, feature_df = create_feature_matrix(movies_df, ratings_df)

normalized_features = normalize_features(feature_matrix)

k, metric, n_display, selected_movie = render_sidebar(movies_df)

query_idx = movies_df[movies_df["title"] == selected_movie].index[0]

model = KNNRecommender(n_neighbors=k, metric=metric)
model.fit(normalized_features, movies_df["movie_id"].values)

distances, indices = model.find_neighbors(query_idx)

neighbors_with_scores = model.get_neighbors_with_scores(query_idx)

pca_coords = get_pca_coordinates(normalized_features)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 3D Feature Space Visualization")
    st.markdown("<p style='color: #8b949e; font-size: 0.9em;'>PCA-reduced 3D projection of movie feature space. Movies colored by primary genre. Gold diamond = query movie, Blue = k-nearest neighbors.</p>", unsafe_allow_html=True)
    fig = create_3d_scatter(
        pca_coords,
        query_idx,
        indices,
        movies_df["title"].tolist(),
        movies_df["genres"].tolist(),
        n_display
    )
    st.plotly_chart(fig, width="stretch")

with col2:
    st.markdown("### Recommendations")
    st.markdown(f"<p style='color: #c9d1d9;'><strong>Query Movie:</strong> {selected_movie}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #c9d1d9;'><strong>Found {k} neighbors</strong></p>", unsafe_allow_html=True)
    st.markdown("---")
    
    for idx, (neighbor_idx, similarity) in enumerate(neighbors_with_scores):
        if neighbor_idx == query_idx:
            continue
        neighbor_title = movies_df.iloc[neighbor_idx]["title"]
        neighbor_genres = movies_df.iloc[neighbor_idx]["genres"]
        render_movie_card(neighbor_title, neighbor_genres, similarity, idx)

st.markdown("---")
render_theory_note(k, metric)

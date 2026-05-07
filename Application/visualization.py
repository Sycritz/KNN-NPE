import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple
import streamlit as st
from utils import compute_pca

GENRE_COLORS = {
    "Action": "#e74c3c",
    "Adventure": "#e67e22",
    "Animation": "#f39c12",
    "Children's": "#f1c40f",
    "Comedy": "#2ecc71",
    "Crime": "#1abc9c",
    "Documentary": "#3498db",
    "Drama": "#9b59b6",
    "Fantasy": "#8e44ad",
    "Film-Noir": "#34495e",
    "Horror": "#c0392b",
    "Musical": "#d35400",
    "Mystery": "#16a085",
    "Romance": "#e91e63",
    "Sci-Fi": "#00bcd4",
    "Thriller": "#607d8b",
    "War": "#795548",
    "Western": "#ff5722"
}


@st.cache_data
def get_pca_coordinates(feature_matrix: np.ndarray) -> np.ndarray:
    pca_result, _ = compute_pca(feature_matrix, n_components=3)
    return pca_result


def get_primary_genre(genres: str) -> str:
    genre_list = genres.split("|")
    return genre_list[0] if genre_list else "Drama"

def create_3d_scatter(
    pca_coords: np.ndarray,
    query_idx: int,
    neighbor_indices: np.ndarray,
    movie_titles: List[str],
    movie_genres: List[str],
    n_display: int = 1000
) -> go.Figure:
    n_movies = len(movie_titles)
    display_indices = np.random.choice(n_movies, min(n_display, n_movies), replace=False)
    
    x = pca_coords[:, 0]
    y = pca_coords[:, 1]
    z = pca_coords[:, 2]
    
    colors = []
    sizes = np.array([4] * n_movies)
    markers = np.array(["circle"] * n_movies)
    
    for i in range(n_movies):
        if i == query_idx:
            colors.append("rgba(255, 215, 0, 1)")
            sizes[i] = 35
            markers[i] = "diamond"
        elif i in neighbor_indices:
            colors.append("rgba(88, 166, 255, 1)")
            sizes[i] = 18
        else:
            primary_genre = get_primary_genre(movie_genres[i])
            colors.append(GENRE_COLORS.get(primary_genre, "rgba(139, 148, 155, 0.6)"))
    
    colors = np.array(colors)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x[display_indices],
        y=y[display_indices],
        z=z[display_indices],
        mode="markers",
        marker=dict(
            size=sizes[display_indices],
            color=colors[display_indices],
            symbol=markers[display_indices],
            opacity=0.7,
            line=dict(width=0.5, color='rgba(255,255,255,0.2)')
        ),
        text=[f"{movie_titles[i]}<br>Genre: {movie_genres[i]}" for i in display_indices],
        name="Movies",
        hovertemplate="<b style='color: white;'>%{text}</b><br>" +
                     "PC1: %{x:.3f}<br>" +
                     "PC2: %{y:.3f}<br>" +
                     "PC3: %{z:.3f}<extra></extra>",
        hoverlabel=dict(
            bgcolor="rgba(22, 27, 34, 0.95)",
            bordercolor="rgba(88, 166, 255, 0.5)",
            font_size=12,
            font_family="Segoe UI"
        )
    )])
    
    query_pos = pca_coords[query_idx]
    
    for idx in neighbor_indices:
        neighbor_pos = pca_coords[idx]
        fig.add_trace(go.Scatter3d(
            x=[query_pos[0], neighbor_pos[0]],
            y=[query_pos[1], neighbor_pos[1]],
            z=[query_pos[2], neighbor_pos[2]],
            mode="lines",
            line=dict(color="rgba(255, 255, 255, 0.4)", width=1.5, dash='dot'),
            showlegend=False,
            hoverinfo="skip"
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(text="PC1 (Variance Explained)", font=dict(size=14, color='#c9d1d9')),
                tickfont=dict(size=11, color='#8b949e'),
                gridcolor='rgba(48, 54, 61, 0.5)',
                showbackground=True,
                backgroundcolor='rgba(13, 17, 23, 0.8)'
            ),
            yaxis=dict(
                title=dict(text="PC2 (Variance Explained)", font=dict(size=14, color='#c9d1d9')),
                tickfont=dict(size=11, color='#8b949e'),
                gridcolor='rgba(48, 54, 61, 0.5)',
                showbackground=True,
                backgroundcolor='rgba(13, 17, 23, 0.8)'
            ),
            zaxis=dict(
                title=dict(text="PC3 (Variance Explained)", font=dict(size=14, color='#c9d1d9')),
                tickfont=dict(size=11, color='#8b949e'),
                gridcolor='rgba(48, 54, 61, 0.5)',
                showbackground=True,
                backgroundcolor='rgba(13, 17, 23, 0.8)'
            ),
            bgcolor='rgba(13, 17, 23, 1)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        paper_bgcolor='rgba(13, 17, 23, 1)',
        plot_bgcolor='rgba(13, 17, 23, 1)',
        font=dict(
            family='Segoe UI, Tahoma, Geneva, Verdana, sans-serif',
            color='#c9d1d9',
            size=12
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=550,
        showlegend=False,
        annotations=[
            dict(
                text=f"Showing {min(n_display, n_movies)} of {n_movies} movies",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                font=dict(size=11, color='#8b949e')
            ),
            dict(
                text="Diamond = Query Movie | Blue = Neighbors | Colors = Genres",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.94,
                xanchor="left",
                yanchor="top",
                font=dict(size=10, color='#58a6ff')
            )
        ]
    )
    
    return fig

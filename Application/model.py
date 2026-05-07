import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List
import streamlit as st


@st.cache_data
def fit_nearest_neighbors(feature_matrix: np.ndarray, n_neighbors: int, metric: str):
    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
        algorithm="auto"
    )
    model.fit(feature_matrix)
    return model


class KNNRecommender:
    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = None
        self.feature_matrix = None
        self.movie_ids = None

    def fit(self, feature_matrix: np.ndarray, movie_ids: np.ndarray) -> "KNNRecommender":
        self.model = fit_nearest_neighbors(feature_matrix, self.n_neighbors, self.metric)
        self.feature_matrix = feature_matrix
        self.movie_ids = movie_ids
        return self

    def find_neighbors(self, query_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        distances, indices = self.model.kneighbors(
            self.feature_matrix[query_idx].reshape(1, -1)
        )
        return distances[0], indices[0]

    def get_neighbors_with_scores(
        self,
        query_idx: int,
        max_distance: float = None
    ) -> List[Tuple[int, float]]:
        distances, indices = self.find_neighbors(query_idx)
        
        if max_distance is not None:
            similarities = 1 - (distances / max_distance)
        else:
            similarities = 1 - (distances / distances.max())
        
        neighbors = list(zip(indices, similarities))
        return neighbors

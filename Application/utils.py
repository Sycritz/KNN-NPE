import numpy as np
from typing import Tuple
import streamlit as st


@st.cache_data
def normalize_features(feature_matrix: np.ndarray) -> np.ndarray:
    min_vals = feature_matrix.min(axis=0)
    max_vals = feature_matrix.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    normalized = (feature_matrix - min_vals) / range_vals
    return normalized


@st.cache_data
def compute_pca(feature_matrix: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, object]:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(feature_matrix)
    return pca_result, pca

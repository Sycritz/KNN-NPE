import streamlit as st
from typing import List, Tuple


def render_movie_card(movie_title: str, genres: str, similarity: float, index: int) -> None:
    with st.container():
        color_intensity = int(similarity * 255)
        similarity_color = f"rgb({color_intensity}, {color_intensity}, {color_intensity})"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(22, 27, 34, 0.95) 0%, rgba(13, 17, 23, 0.95) 100%);
            border: 1px solid rgba(48, 54, 61, 0.8);
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        ">
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            ">
                <span style="
                    background-color: rgba(56, 139, 253, 0.2);
                    color: #58a6ff;
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-size: 0.85em;
                    font-weight: 600;
                ">#{index + 1}</span>
            </div>
            <h4 style="
                color: #ffffff;
                margin: 0 0 8px 0;
                font-size: 1.05em;
                font-weight: 600;
                line-height: 1.3;
            ">{movie_title}</h4>
            <p style="
                color: #8b949e;
                font-size: 0.85em;
                margin: 0 0 12px 0;
                font-style: italic;
            ">{genres}</p>
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <div style="
                    flex-grow: 1;
                    background: rgba(48, 54, 61, 0.6);
                    border-radius: 6px;
                    height: 8px;
                    margin-right: 12px;
                    overflow: hidden;
                ">
                    <div style="
                        width: {similarity * 100}%;
                        height: 100%;
                        background: linear-gradient(90deg, #238636, #2ea043);
                        border-radius: 6px;
                    "></div>
                </div>
                <span style="
                    color: #2ea043;
                    font-weight: 700;
                    font-size: 0.95em;
                    min-width: 70px;
                    text-align: right;
                ">{similarity:.3f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_theory_note(k: int, metric: str) -> None:
    k_effect = (
        "Small k (1-3): Highly local, sensitive to noise. Captures very similar movies but may miss broader patterns."
        if k <= 3
        else "Medium k (4-10): Balanced between local and global patterns. Good for general recommendations."
        if k <= 10
        else "Large k (11+): More global patterns, robust to noise. May include less similar movies but captures broader genres."
    )
    
    metric_effect = (
        "Euclidean: Measures absolute distance in feature space. Good when magnitude of features matters (e.g., rating counts)."
        if metric == "euclidean"
        else "Cosine: Measures angular similarity. Good when direction matters more than magnitude (e.g., genre patterns)."
    )
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.95) 0%, rgba(13, 17, 23, 0.95) 100%);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 24px;
        margin: 24px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    ">
        <div style="
            display: flex;
            align-items: center;
            margin-bottom: 16px;
        ">
            <div style="
                background-color: rgba(56, 139, 253, 0.2);
                color: #58a6ff;
                padding: 8px 16px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 0.95em;
            ">Theory Note</div>
        </div>
        <div style="
            background-color: rgba(48, 54, 61, 0.4);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        ">
            <p style="
                color: #c9d1d9;
                margin: 0 0 8px 0;
                font-size: 0.9em;
            "><strong style="color: #58a6ff;">k = {k}:</strong></p>
            <p style="color: #8b949e; margin: 0; font-size: 0.9em; line-height: 1.5;">{k_effect}</p>
        </div>
        <div style="
            background-color: rgba(48, 54, 61, 0.4);
            border-radius: 8px;
            padding: 16px;
        ">
            <p style="
                color: #c9d1d9;
                margin: 0 0 8px 0;
                font-size: 0.9em;
            "><strong style="color: #58a6ff;">Metric = {metric}:</strong></p>
            <p style="color: #8b949e; margin: 0; font-size: 0.9em; line-height: 1.5;">{metric_effect}</p>
        </div>
    </div>
    """.format(k=k, k_effect=k_effect, metric=metric, metric_effect=metric_effect), unsafe_allow_html=True)


def render_sidebar(movies_df, k_range: Tuple[int, int] = (1, 20)) -> Tuple[int, str, int, str]:
    st.sidebar.markdown("### Hyperparameters")
    st.sidebar.markdown("<p style='color: #8b949e; font-size: 0.85em; margin-bottom: 16px;'>Configure the k-NN algorithm parameters</p>", unsafe_allow_html=True)
    
    k = st.sidebar.slider(
        "Number of Neighbors (k)",
        min_value=k_range[0],
        max_value=k_range[1],
        value=5,
        step=1
    )
    
    metric = st.sidebar.selectbox(
        "Distance Metric",
        options=["euclidean", "cosine"],
        index=0
    )
    
    n_display = st.sidebar.slider(
        "Movies to Display in 3D Plot",
        min_value=100,
        max_value=2000,
        value=500,
        step=100
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Movie Selection")
    st.sidebar.markdown("<p style='color: #8b949e; font-size: 0.85em; margin-bottom: 16px;'>Choose a movie to find recommendations</p>", unsafe_allow_html=True)
    
    movie_options = movies_df["title"].tolist()
    selected_movie = st.sidebar.selectbox(
        "Select a Movie",
        options=movie_options,
        index=0
    )
    
    return k, metric, n_display, selected_movie

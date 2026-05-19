from __future__ import annotations

import json
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import StandardScaler

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from database.db_manager import DBManager
from processing.vectorizer import FEATURE_NAMES


def build_dashboard_theme() -> str:
    return '''
    <style>
        .reportview-container {
            background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
        }

        .stButton>button {
            background-color: #2563eb;
            color: white;
            border-radius: 0.75rem;
            border: 1px solid #1d4ed8;
            box-shadow: 0 8px 24px rgba(45, 212, 191, 0.18);
        }

        .stButton>button:hover {
            background-color: #1d4ed8;
        }

        .css-18e3th9 {
            padding-top: 0.75rem;
        }

        .css-1d391kg {
            padding: 1rem 1rem 2rem;
        }

        .stSidebar {
            background-color: #020617;
            color: #e2e8f0;
        }

        .stAlert {
            background-color: rgba(15, 23, 42, 0.95);
            border: 1px solid rgba(147, 197, 253, 0.2);
        }
    </style>
    '''


def generate_synthetic_state_space(samples: int = 120, features: int = 18, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    raw = rng.normal(loc=0.0, scale=1.0, size=(samples, features))
    feature_names = [f"feature_{idx + 1:02d}" for idx in range(features)]
    df = pd.DataFrame(raw, columns=feature_names)
    df = pd.DataFrame(StandardScaler().fit_transform(df), columns=feature_names)
    df["market_bias"] = rng.choice(["Bull", "Bear", "Neutral"], size=samples, p=[0.36, 0.28, 0.36])
    df["policy_action"] = rng.choice(["HOLD", "BUY", "SELL", "BUY_ALL_IN"], size=samples, p=[0.34, 0.28, 0.28, 0.10])
    df["confidence"] = np.clip(rng.normal(loc=0.6, scale=0.16, size=samples), 0.1, 0.99)
    df["drawdown_risk"] = np.clip(rng.exponential(scale=0.08, size=samples), 0.0, 0.35)
    df["insider_signal_strength"] = np.clip(rng.normal(loc=0.45, scale=0.22, size=samples), 0.0, 1.0)
    df["action"] = df["policy_action"]
    df["ticker"] = [f"SYNTH_{idx+1:03d}" for idx in range(samples)]
    df["alpha_score"] = np.clip(rng.normal(loc=0.55, scale=0.18, size=samples), 0.0, 1.0)
    df["sentiment_score"] = np.clip(rng.normal(loc=0.5, scale=0.25, size=samples), 0.0, 1.0)
    df["source"] = "synthetic"
    return df


def load_live_state_space(sample_limit: int = 240) -> pd.DataFrame:
    try:
        db = DBManager()
        candidates = db.get_recommended_candidates(limit=sample_limit, min_score=0.0)
        rows: list[dict[str, object]] = []

        for candidate in candidates:
            vector = candidate.get("vector")
            if isinstance(vector, str):
                try:
                    vector = json.loads(vector)
                except json.JSONDecodeError:
                    continue
            if not isinstance(vector, (list, tuple)) or len(vector) < 1:
                continue

            values = [float(v) for v in vector[: len(FEATURE_NAMES)]]
            if len(values) < len(FEATURE_NAMES):
                values += [0.5] * (len(FEATURE_NAMES) - len(values))

            row = {FEATURE_NAMES[i]: values[i] for i in range(len(FEATURE_NAMES))}
            row["ticker"] = candidate.get("ticker", "UNKNOWN") or "UNKNOWN"
            row["action"] = candidate.get("action", "UNKNOWN") or "UNKNOWN"
            row["alpha_score"] = float(candidate.get("alpha_score", 0.0) or 0.0)
            row["sentiment_score"] = float(candidate.get("sentiment_score", 0.5) or 0.5)
            row["source"] = candidate.get("source", "database") or "database"
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)
    except Exception:
        st.warning(
            "Live database state vectors are unavailable. Falling back to synthetic demo data."
        )
        return pd.DataFrame()


def build_dashboard_dataset(
    live_mode: bool,
    sample_count: int,
    selected_dim_count: int,
) -> tuple[pd.DataFrame, bool, str]:
    source = "synthetic"
    if live_mode:
        dataset = load_live_state_space(sample_limit=sample_count)
        if not dataset.empty:
            source = "database"
        else:
            dataset = generate_synthetic_state_space(samples=sample_count)
    else:
        dataset = generate_synthetic_state_space(samples=sample_count)

    feature_cols = FEATURE_NAMES[:selected_dim_count]
    metadata_cols = ["ticker", "action", "alpha_score", "sentiment_score", "source"]

    for col in metadata_cols:
        if col not in dataset.columns:
            dataset[col] = "synthetic" if col == "source" else 0.0

    return dataset.loc[:, feature_cols + metadata_cols], source == "database", source


def project_state_space(
    state_df: pd.DataFrame,
    method: Literal["PCA", "t-SNE", "UMAP", "MDS"] = "PCA",
    n_components: int = 2,
    perplexity: int = 30,
    learning_rate: float = 200.0,
    tsne_iter: int = 1000,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
) -> pd.DataFrame:
    if state_df.empty:
        return pd.DataFrame(columns=["x", "y"])

    projection_data = state_df.select_dtypes(include=[np.number])
    if projection_data.shape[1] < 2:
        raise ValueError("At least two numeric features are required for projection.")

    n_samples = len(projection_data)
    if n_samples < 2:
        return pd.DataFrame(columns=["x", "y"])

    if method == "PCA":
        model = PCA(n_components=n_components, random_state=42)
    elif method == "t-SNE":
        perplexity = min(perplexity, max(2, (n_samples - 1) // 3))
        model = TSNE(
            n_components=n_components,
            init="pca",
            perplexity=perplexity,
            learning_rate=max(10.0, min(learning_rate, 1000.0)),
            n_iter=tsne_iter,
            random_state=42,
        )
    elif method == "UMAP":
        if not HAS_UMAP:
            raise ImportError("UMAP is not installed. Install umap-learn or choose another projection method.")
        model = UMAP(
            n_components=n_components,
            n_neighbors=max(2, umap_neighbors),
            min_dist=max(0.0, min(0.99, umap_min_dist)),
            init="pca",
            random_state=42,
        )
    else:
        model = MDS(n_components=n_components, random_state=42)

    coords = model.fit_transform(projection_data)
    coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(coords, columns=["x", "y"])


def build_cosine_similarity(state_df: pd.DataFrame) -> pd.DataFrame:
    if state_df.empty:
        return pd.DataFrame()

    numeric = state_df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    norms = np.linalg.norm(numeric, axis=1, keepdims=True).clip(min=1e-10)
    numeric = numeric / norms
    matrix = numeric @ numeric.T
    matrix = np.clip(matrix, -1.0, 1.0)
    return pd.DataFrame(matrix, index=state_df.index, columns=state_df.index)


def render_sidebar(live_dataset_available: bool) -> tuple[
    bool, int, int, Literal["PCA", "t-SNE", "UMAP", "MDS"], str, int, float, int, int, float
]:
    st.sidebar.header("State Space Controls")
    st.sidebar.markdown(
        "Choose the vector source, feature dimensionality, and projection hyperparameters for the visualization."
    )

    data_source = st.sidebar.selectbox(
        "Vector source",
        ["Live DB signals", "Synthetic demo"],
        index=0 if live_dataset_available else 1,
    )
    live_mode = data_source == "Live DB signals"

    sample_count = st.sidebar.slider("Sample count", min_value=40, max_value=240, value=120, step=20)
    selected_dim_count = st.sidebar.slider("Feature dimensions", min_value=11, max_value=18, value=18, step=1)
    projection_method = st.sidebar.selectbox(
        "Projection method",
        ["PCA", "t-SNE", "UMAP", "MDS"],
        index=0,
    )
    similarity_label = st.sidebar.selectbox(
        "Similarity label",
        ["action", "ticker", "alpha_score", "sentiment_score"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.metric("DB mode active", "Yes" if live_mode else "No")
    st.sidebar.metric("Live vectors available", "Yes" if live_dataset_available else "No")
    st.sidebar.markdown(
        "Use live signal vectors from the local DB when they exist, otherwise the dashboard falls back to synthetic demonstration data."
    )

    tsne_perplexity = 30
    tsne_learning_rate = 200.0
    tsne_iter = 1000
    umap_neighbors = 15
    umap_min_dist = 0.1

    if projection_method == "t-SNE":
        st.sidebar.markdown("---")
        tsne_perplexity = st.sidebar.slider("t-SNE perplexity", 5, 50, 30)
        tsne_learning_rate = st.sidebar.slider("t-SNE learning rate", 50, 1000, 200, step=25)
        tsne_iter = st.sidebar.slider("t-SNE iterations", 250, 1500, 1000, step=250)
    elif projection_method == "UMAP":
        st.sidebar.markdown("---")
        umap_neighbors = st.sidebar.slider("UMAP neighbors", 2, 50, 15)
        umap_min_dist = st.sidebar.slider("UMAP min distance", 0.0, 0.99, 0.1, step=0.05)

    st.sidebar.divider()
    st.sidebar.info(
        "Run the dashboard locally with: `streamlit run visualization/high_dim_state_dashboard.py`."
    )
    if projection_method == "UMAP" and not HAS_UMAP:
        st.sidebar.warning("UMAP is not installed. Select PCA, t-SNE, or MDS instead.")

    return (
        live_mode,
        sample_count,
        selected_dim_count,
        projection_method,
        similarity_label,
        tsne_perplexity,
        tsne_learning_rate,
        tsne_iter,
        umap_neighbors,
        umap_min_dist,
    )


def render_projection_tab(
    state_df: pd.DataFrame,
    projection_method: Literal["PCA", "t-SNE", "UMAP", "MDS"],
    similarity_label: str,
    tsne_perplexity: int,
    tsne_learning_rate: float,
    tsne_iter: int,
    umap_neighbors: int,
    umap_min_dist: float,
):
    st.subheader("High-Dimensional State Projections")
    st.markdown(
        "Project the 18-dimensional observation space into a 2D plane, comparing policy clusters and disclosure signal patterns."
    )
    try:
        projection = project_state_space(
            state_df,
            method=projection_method,
            n_components=2,
            perplexity=tsne_perplexity,
            learning_rate=tsne_learning_rate,
            tsne_iter=tsne_iter,
            umap_neighbors=umap_neighbors,
            umap_min_dist=umap_min_dist,
        )
    except Exception as exc:
        st.error(f"Projection failed: {exc}")
        return

    scatter = go.Figure(
        data=go.Scatter(
            x=projection["x"],
            y=projection["y"],
            mode="markers",
            marker=dict(
                size=10,
                color=state_df[similarity_label] if similarity_label in state_df else state_df["alpha_score"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=similarity_label),
                opacity=0.88,
            ),
            text=state_df[["ticker", "action", "alpha_score", "sentiment_score"]].astype(str).agg(" | ".join, axis=1),
            hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
        )
    )
    scatter.update_layout(
        margin=dict(l=24, r=24, t=40, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,10,24,0.85)",
        font_color="#e2e8f0",
        title=f"{projection_method} projection colored by {similarity_label}",
    )
    st.plotly_chart(scatter, use_container_width=True)


def render_parallel_coordinates_tab(state_df: pd.DataFrame):
    st.subheader("Parallel Coordinates Overview")
    st.markdown(
        "Compare filtered feature values across state vectors to identify the dimensions that drive insider or congressional signal clusters."
    )
    numeric_columns = list(state_df.select_dtypes(include=[np.number]).columns[:8])
    if len(numeric_columns) < 2:
        st.warning("Not enough numeric features to build a parallel coordinates chart.")
        return

    labels = state_df["action"].astype("category").cat.codes if "action" in state_df else np.arange(len(state_df))
    parallel_fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=labels,
                colorscale="Portland",
                showscale=False,
                cmin=0,
                cmax=max(int(labels.max()) if hasattr(labels, "max") else 0, 3),
            ),
            dimensions=[
                dict(label=column, values=state_df[column])
                for column in numeric_columns
            ],
        )
    )
    parallel_fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
    )
    st.plotly_chart(parallel_fig, use_container_width=True)


def render_similarity_tab(state_df: pd.DataFrame, similarity_label: str):
    st.subheader("Similarity Matrix")
    st.markdown(
        "Cosine similarity across selected state vectors, ordered by the chosen label to expose regime coherence."
    )
    similarity_matrix = build_cosine_similarity(state_df.select_dtypes(include=[np.number]))
    if similarity_matrix.empty:
        st.warning("Similarity matrix could not be computed for the current selection.")
        return

    if similarity_label in state_df.columns and similarity_label in ["alpha_score", "sentiment_score"]:
        ordered_index = state_df.sort_values(by=similarity_label, ascending=False).index
        similarity_matrix = similarity_matrix.loc[ordered_index, ordered_index]

    heatmap = go.Figure(
        data=go.Heatmap(
            z=similarity_matrix.values,
            x=similarity_matrix.columns.astype(str),
            y=similarity_matrix.index.astype(str),
            colorscale="blues",
            zmin=-1,
            zmax=1,
        )
    )
    heatmap.update_layout(
        margin=dict(l=24, r=24, t=40, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
    )
    st.plotly_chart(heatmap, use_container_width=True)
    st.markdown(
        f"**Ordered by:** `{similarity_label}` — similar states are grouped by this label when available."
    )


def main() -> None:
    st.set_page_config(
        page_title="Stock Hawk High-Dimensional State Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(build_dashboard_theme(), unsafe_allow_html=True)
    st.title("Stock Hawk: High-Dimensional State Visualization")
    st.write(
        "This dashboard presents 18-dimensional trading state vectors from live repository signals when available, with a synthetic fallback for exploratory analysis."
    )

    live_stats = {}
    try:
        db_manager = DBManager()
        live_stats = db_manager.get_stats()
    except Exception:
        live_stats = {}

    (
        live_mode,
        sample_count,
        selected_dim_count,
        projection_method,
        similarity_label,
        tsne_perplexity,
        tsne_learning_rate,
        tsne_iter,
        umap_neighbors,
        umap_min_dist,
    ) = render_sidebar(bool(live_stats))

    dataset, is_live, source = build_dashboard_dataset(
        live_mode=live_mode,
        sample_count=sample_count,
        selected_dim_count=selected_dim_count,
    )

    if source == "database":
        st.success("Loaded state vectors from the local database.")
        st.json({
            "congress_trades": live_stats.get("congress_trades", 0),
            "insider_trades": live_stats.get("insider_trades", 0),
            "news_sentiment": live_stats.get("news_sentiment", 0),
        })
    else:
        st.info("Using synthetic demonstration data. Populate the DB and save recommended candidates to enable live mode.")

    st.markdown(
        f"**Source:** {source} | **Samples:** {len(dataset):,} | **Dimensionality:** {selected_dim_count} | **Projection:** {projection_method}"
    )
    st.divider()

    if len(dataset) == 0:
        st.warning("No dataset is available. Ensure the local DB contains recommended signal vectors or switch to Synthetic demo mode.")
        return

    projection_tab, parallel_tab, similarity_tab = st.tabs([
        "Projection", "Parallel Coordinates", "Similarity Matrix"
    ])

    with projection_tab:
        render_projection_tab(
            dataset,
            projection_method,
            similarity_label,
            tsne_perplexity,
            tsne_learning_rate,
            tsne_iter,
            umap_neighbors,
            umap_min_dist,
        )

    with parallel_tab:
        render_parallel_coordinates_tab(dataset)

    with similarity_tab:
        render_similarity_tab(dataset, similarity_label)

    if is_live:
        st.markdown("---")
        st.markdown("### Live vector preview")
        st.dataframe(dataset[["ticker", "action", "alpha_score", "sentiment_score", "source"]].head(20), use_container_width=True)

    st.sidebar.info(
        "Launch the dashboard with `python main.py dashboard` or `streamlit run visualization/high_dim_state_dashboard.py`."
    )


if __name__ == "__main__":
    main()

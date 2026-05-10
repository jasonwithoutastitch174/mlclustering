import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AgglomerativeClustering,
    Birch,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.datasets import (
    load_iris,
    load_wine,
    make_blobs,
    make_circles,
    make_moons,
)
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler

st.set_page_config(
    page_title="ML Clustering Explorer",
    page_icon="🧩",
    layout="wide",
)


# ---------------------- Data loading ----------------------
SAMPLE_DATASETS = [
    "Iris",
    "Wine",
    "Blobs (synthetic)",
    "Moons (synthetic)",
    "Circles (synthetic)",
]


@st.cache_data
def load_sample(name: str) -> pd.DataFrame:
    if name == "Iris":
        d = load_iris(as_frame=True)
        df = d.frame.rename(columns={"target": "label"})
        return df
    if name == "Wine":
        d = load_wine(as_frame=True)
        df = d.frame.rename(columns={"target": "label"})
        return df
    if name == "Blobs (synthetic)":
        X, y = make_blobs(n_samples=500, centers=4, n_features=4, random_state=42)
        df = pd.DataFrame(X, columns=[f"feat_{i+1}" for i in range(X.shape[1])])
        df["label"] = y
        return df
    if name == "Moons (synthetic)":
        X, y = make_moons(n_samples=500, noise=0.08, random_state=42)
        df = pd.DataFrame(X, columns=["feat_1", "feat_2"])
        df["label"] = y
        return df
    if name == "Circles (synthetic)":
        X, y = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42)
        df = pd.DataFrame(X, columns=["feat_1", "feat_2"])
        df["label"] = y
        return df
    raise ValueError(name)


def get_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number])


# ---------------------- Sidebar ----------------------
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("### 1. Data")
data_source = st.sidebar.radio(
    "Data source", ["Sample dataset", "Upload CSV"], horizontal=False
)

df: pd.DataFrame | None = None
if data_source == "Sample dataset":
    sample_name = st.sidebar.selectbox("Choose sample", SAMPLE_DATASETS)
    df = load_sample(sample_name)
else:
    up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Preprocessing")
scaler_choice = st.sidebar.selectbox(
    "Scaling", ["StandardScaler", "MinMaxScaler", "None"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 3. Algorithm")
algo = st.sidebar.selectbox(
    "Algorithm",
    [
        "K-Means",
        "Mini-Batch K-Means",
        "Gaussian Mixture (EM)",
        "DBSCAN",
        "OPTICS",
        "Agglomerative (Hierarchical)",
        "Spectral Clustering",
        "Mean Shift",
        "BIRCH",
        "PCA (dimensionality reduction)",
        "ICA (dimensionality reduction)",
        "Truncated SVD (dimensionality reduction)",
        "t-SNE (dimensionality reduction)",
    ],
)


# ---------------------- Main ----------------------
st.title("🧩 ML Clustering Explorer")
st.caption(
    "Explore unsupervised learning algorithms — clustering and dimensionality reduction — on your own data or built-in samples."
)

if df is None:
    st.info("👈 Choose a sample dataset or upload a CSV file to get started.")
    st.stop()

with st.expander("📄 Data preview", expanded=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Numeric columns", get_numeric(df).shape[1])
    st.dataframe(df.head(20), width='stretch')

numeric_df = get_numeric(df)
if numeric_df.shape[1] < 2:
    st.error("Need at least 2 numeric columns for clustering.")
    st.stop()

default_features = [c for c in numeric_df.columns if c != "label"]
features = st.multiselect(
    "Feature columns to use",
    options=list(numeric_df.columns),
    default=default_features,
)
if len(features) < 2:
    st.warning("Pick at least 2 feature columns.")
    st.stop()

X = numeric_df[features].dropna().values
if scaler_choice == "StandardScaler":
    X_proc = StandardScaler().fit_transform(X)
elif scaler_choice == "MinMaxScaler":
    X_proc = MinMaxScaler().fit_transform(X)
else:
    X_proc = X.copy()


# ---------------------- Algorithm params ----------------------
st.subheader(f"🔧 {algo} parameters")

labels: np.ndarray | None = None
embedding: np.ndarray | None = None
model_info: dict = {}

cols = st.columns(4)

if algo == "K-Means":
    k = cols[0].slider("n_clusters (k)", 2, 15, 3)
    init = cols[1].selectbox("init", ["k-means++", "random"])
    n_init = cols[2].slider("n_init", 1, 20, 10)
    max_iter = cols[3].slider("max_iter", 50, 1000, 300, step=50)
    model = KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=max_iter, random_state=42)
    labels = model.fit_predict(X_proc)
    model_info["inertia"] = float(model.inertia_)

elif algo == "Mini-Batch K-Means":
    k = cols[0].slider("n_clusters", 2, 15, 3)
    batch = cols[1].slider("batch_size", 32, 1024, 256, step=32)
    model = MiniBatchKMeans(n_clusters=k, batch_size=batch, n_init=10, random_state=42)
    labels = model.fit_predict(X_proc)
    model_info["inertia"] = float(model.inertia_)

elif algo == "Gaussian Mixture (EM)":
    k = cols[0].slider("n_components", 2, 15, 3)
    cov = cols[1].selectbox("covariance_type", ["full", "tied", "diag", "spherical"])
    max_iter = cols[2].slider("max_iter", 50, 500, 100, step=50)
    model = GaussianMixture(n_components=k, covariance_type=cov, max_iter=max_iter, random_state=42)
    model.fit(X_proc)
    labels = model.predict(X_proc)
    model_info["log_likelihood"] = float(model.score(X_proc))
    model_info["AIC"] = float(model.aic(X_proc))
    model_info["BIC"] = float(model.bic(X_proc))

elif algo == "DBSCAN":
    eps = cols[0].slider("eps", 0.05, 5.0, 0.5, step=0.05)
    min_s = cols[1].slider("min_samples", 2, 50, 5)
    model = DBSCAN(eps=eps, min_samples=min_s)
    labels = model.fit_predict(X_proc)
    model_info["n_noise"] = int(np.sum(labels == -1))

elif algo == "OPTICS":
    min_s = cols[0].slider("min_samples", 2, 50, 5)
    xi = cols[1].slider("xi", 0.001, 0.5, 0.05, step=0.005)
    model = OPTICS(min_samples=min_s, xi=xi)
    labels = model.fit_predict(X_proc)
    model_info["n_noise"] = int(np.sum(labels == -1))

elif algo == "Agglomerative (Hierarchical)":
    k = cols[0].slider("n_clusters", 2, 15, 3)
    linkage = cols[1].selectbox("linkage", ["ward", "complete", "average", "single"])
    metric = cols[2].selectbox(
        "metric", ["euclidean", "manhattan", "cosine"], disabled=(linkage == "ward")
    )
    if linkage == "ward":
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    else:
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric)
    labels = model.fit_predict(X_proc)

elif algo == "Spectral Clustering":
    k = cols[0].slider("n_clusters", 2, 15, 3)
    affinity = cols[1].selectbox("affinity", ["rbf", "nearest_neighbors"])
    model = SpectralClustering(
        n_clusters=k, affinity=affinity, random_state=42, assign_labels="kmeans"
    )
    labels = model.fit_predict(X_proc)

elif algo == "Mean Shift":
    bw = cols[0].slider("bandwidth (0 = auto)", 0.0, 5.0, 0.0, step=0.1)
    model = MeanShift(bandwidth=None if bw == 0 else bw)
    labels = model.fit_predict(X_proc)
    model_info["n_clusters_found"] = int(len(np.unique(labels)))

elif algo == "BIRCH":
    k = cols[0].slider("n_clusters", 2, 15, 3)
    threshold = cols[1].slider("threshold", 0.1, 2.0, 0.5, step=0.05)
    model = Birch(n_clusters=k, threshold=threshold)
    labels = model.fit_predict(X_proc)

elif algo == "PCA (dimensionality reduction)":
    n_comp = cols[0].slider("n_components", 2, min(10, X_proc.shape[1]), 2)
    model = PCA(n_components=n_comp, random_state=42)
    embedding = model.fit_transform(X_proc)
    model_info["explained_variance_ratio"] = model.explained_variance_ratio_.tolist()
    model_info["total_explained"] = float(np.sum(model.explained_variance_ratio_))

elif algo == "ICA (dimensionality reduction)":
    n_comp = cols[0].slider("n_components", 2, min(10, X_proc.shape[1]), 2)
    model = FastICA(n_components=n_comp, random_state=42, max_iter=500)
    embedding = model.fit_transform(X_proc)

elif algo == "Truncated SVD (dimensionality reduction)":
    n_comp = cols[0].slider("n_components", 2, min(10, X_proc.shape[1] - 1), 2)
    model = TruncatedSVD(n_components=n_comp, random_state=42)
    embedding = model.fit_transform(X_proc)
    model_info["explained_variance_ratio"] = model.explained_variance_ratio_.tolist()
    model_info["total_explained"] = float(np.sum(model.explained_variance_ratio_))

elif algo == "t-SNE (dimensionality reduction)":
    n_comp = cols[0].selectbox("n_components", [2, 3], index=0)
    perplexity = cols[1].slider("perplexity", 5, 50, 30)
    model = TSNE(n_components=n_comp, perplexity=perplexity, random_state=42, init="pca")
    embedding = model.fit_transform(X_proc)


# ---------------------- Results: clustering ----------------------
if labels is not None:
    st.subheader("📊 Clustering results")

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clusters found", n_clusters)
    m2.metric("Samples", len(labels))
    if -1 in labels:
        m3.metric("Noise points", int(np.sum(labels == -1)))

    # Internal validation metrics (need >=2 clusters and not all noise)
    valid_mask = labels != -1
    metrics_box = st.columns(3)
    if n_clusters >= 2 and np.sum(valid_mask) > n_clusters:
        try:
            sil = silhouette_score(X_proc[valid_mask], labels[valid_mask])
            metrics_box[0].metric("Silhouette", f"{sil:.3f}", help="Range [-1, 1], higher is better")
        except Exception as e:
            metrics_box[0].write(f"Silhouette: n/a ({e})")
        try:
            ch = calinski_harabasz_score(X_proc[valid_mask], labels[valid_mask])
            metrics_box[1].metric("Calinski-Harabasz", f"{ch:.2f}", help="Higher is better")
        except Exception:
            pass
        try:
            db = davies_bouldin_score(X_proc[valid_mask], labels[valid_mask])
            metrics_box[2].metric("Davies-Bouldin", f"{db:.3f}", help="Lower is better")
        except Exception:
            pass
    else:
        st.info("Need ≥ 2 clusters to compute validation metrics.")

    if model_info:
        with st.expander("Model details"):
            st.json(model_info)

    # 2D projection for visualization
    if X_proc.shape[1] == 2:
        viz = X_proc
        x_lbl, y_lbl = features[0], features[1]
    else:
        pca_viz = PCA(n_components=2, random_state=42).fit_transform(X_proc)
        viz = pca_viz
        x_lbl, y_lbl = "PC1", "PC2"

    plot_df = pd.DataFrame({"x": viz[:, 0], "y": viz[:, 1], "cluster": labels.astype(str)})
    fig = px.scatter(
        plot_df, x="x", y="y", color="cluster",
        title=f"{algo} — 2D visualization",
        labels={"x": x_lbl, "y": y_lbl},
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
    st.plotly_chart(fig, width='stretch')

    # Cluster size distribution
    counts = pd.Series(labels).value_counts().sort_index()
    bar = px.bar(
        x=counts.index.astype(str), y=counts.values,
        labels={"x": "Cluster", "y": "Count"},
        title="Cluster size distribution",
    )
    st.plotly_chart(bar, width='stretch')

    # Download labeled data
    out = df.copy()
    out = out.iloc[: len(labels)].copy()
    out["cluster"] = labels
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    st.download_button(
        "⬇️ Download labeled CSV",
        buf.getvalue(),
        file_name="clustered.csv",
        mime="text/csv",
    )

    # K-Means specific: elbow plot
    if algo in ("K-Means", "Mini-Batch K-Means"):
        with st.expander("📈 Elbow & silhouette analysis (k = 2..10)"):
            inertias, sils = [], []
            ks = list(range(2, 11))
            for kk in ks:
                m = KMeans(n_clusters=kk, n_init=10, random_state=42).fit(X_proc)
                inertias.append(m.inertia_)
                try:
                    sils.append(silhouette_score(X_proc, m.labels_))
                except Exception:
                    sils.append(np.nan)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers", name="Inertia"))
            fig2.update_layout(title="Elbow (inertia vs k)", xaxis_title="k", yaxis_title="Inertia")
            st.plotly_chart(fig2, width='stretch')

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=ks, y=sils, mode="lines+markers", name="Silhouette"))
            fig3.update_layout(title="Silhouette vs k", xaxis_title="k", yaxis_title="Silhouette")
            st.plotly_chart(fig3, width='stretch')


# ---------------------- Results: dimensionality reduction ----------------------
if embedding is not None:
    st.subheader("📉 Dimensionality reduction results")
    color_col = None
    if "label" in df.columns:
        color_col = df["label"].astype(str).iloc[: len(embedding)]

    if embedding.shape[1] >= 3:
        fig = px.scatter_3d(
            x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2],
            color=color_col,
            title=f"{algo} — 3D projection",
        )
    else:
        fig = px.scatter(
            x=embedding[:, 0], y=embedding[:, 1],
            color=color_col,
            title=f"{algo} — 2D projection",
        )
    st.plotly_chart(fig, width='stretch')

    if "explained_variance_ratio" in model_info:
        evr = model_info["explained_variance_ratio"]
        bar = px.bar(
            x=[f"PC{i+1}" for i in range(len(evr))], y=evr,
            labels={"x": "Component", "y": "Explained variance ratio"},
            title=f"Explained variance (total: {model_info['total_explained']:.3f})",
        )
        st.plotly_chart(bar, width='stretch')


# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#666; padding: 12px 0;">
        Powered by <a href="https://www.tertiarycourses.com.sg/" target="_blank">
        Tertiary Infotech Academy Pte Ltd</a>
    </div>
    """,
    unsafe_allow_html=True,
)

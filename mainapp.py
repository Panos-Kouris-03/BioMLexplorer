# app.py
import os
import tempfile
from io import BytesIO

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

import scanpy as sc
from anndata import AnnData
from typing import Optional, Tuple

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

# utils
import scipy.sparse as sp
import joblib


def _downsample_idx(n, max_n=120_000, seed=0):
    import numpy as np
    if n <= max_n: return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, max_n, replace=False)

st.set_page_config(page_title="BioML Explorer", layout="wide")
st.title("🧬 BioML Explorer")
st.caption("Upload an .h5ad, explore, preprocess, and run ML on raw genes/HVGs/PCA.")

# -------------------------
# Sidebar — data loading & global settings
# -------------------------
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload .h5ad", type=["h5ad"], accept_multiple_files=False)


    st.divider()
    st.header("Feature Source for ML")
    feat_src = st.radio(
        "Select features", ["Raw (X)", "HVGs", "PCA"], index=2,
        help="Raw genes can be heavy; HVGs/PCA speed up training."
    )
    n_hvg = st.slider("Top HVGs", min_value=500, max_value=5000, value=2000, step=100)
    n_pcs = st.slider("PCA components", min_value=10, max_value=100, value=50, step=5)

    st.divider()
    st.header("Target / Labels")
    label_hint = st.text_input(
        "Default target column (optional)", value="cell_type",
        help="We'll preselect this obs column if present."
    )

@st.cache_resource(show_spinner=False)
def _load_adata_from_bytes(content: bytes) -> Tuple[Optional[AnnData], Optional[str]]:
    if content is None:
        return None, None
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".h5ad")
    tf.write(content)
    tf.flush(); tf.close()
    path = tf.name
    adata = sc.read_h5ad(path)
    return adata, path

if uploaded is None:
    st.info("Drop an .h5ad to begin. Tip: try a small dataset first (<100MB).")
    st.stop()

content_bytes = uploaded.getbuffer().tobytes()
adata, temp_path = _load_adata_from_bytes(content_bytes)

# Warn about size
size_mb = len(content_bytes)/(1024**2)
st.toast(f"Loaded file ~{size_mb:.1f} MB", icon="📦")
if size_mb > 500:
    st.warning("This file is >500MB. Consider backed mode or using PCA/HVG features for ML.")


# -------------------------
# Top‑level tabs
# -------------------------
obs_tab, prep_tab, emb_tab, ml_tab, explore_tab = st.tabs(
    ["Observations", "Preprocessing", "Embeddings", "Machine learning", "Exploration"]
)


# -------------------------
# Observations tab
# -------------------------
with obs_tab:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cells", adata.n_obs)
    with col2:
        st.metric("Genes", adata.n_vars)
    with col3:
        st.metric("Layers", len(adata.layers))
    with col4:
        st.metric(
            "Embeddings",
            sum([1 for k, v in adata.obsm.items() if hasattr(v, "shape") and v.shape[1] >= 2])
        )

    st.subheader("Observations (obs) — preview")
    if adata.obs.shape[1] == 0:
        st.write("No obs columns found.")
    else:
        st.dataframe(adata.obs.head(), use_container_width=True)

# -------------------------
# Preprocessing tab
# -------------------------
with prep_tab:
    st.subheader("Quality control & normalization")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        do_filter = st.checkbox("Filter cells/genes", value=False)
    with c2:
        min_counts = st.number_input("min_counts (cells)", 0, 10000, 0)
    with c3:
        min_genes = st.number_input("min_genes (cells)", 0, 10000, 0)
    with c4:
        min_cells = st.number_input("min_cells (genes)", 0, 10000, 0)

    c5, c6 = st.columns(2)
    with c5:
        do_norm = st.checkbox("Normalize total", value=False)
    with c6:
        do_log1p = st.checkbox("Log1p", value=False)

    run_pp = st.button("Apply preprocessing", type="primary", use_container_width=True)

    if 'pp_applied' not in st.session_state:
        st.session_state.pp_applied = False

    if run_pp:
        if adata.isbacked:
            st.error("Preprocessing that modifies AnnData isn't supported in backed mode. Reload without backed.")
        else:
            if do_filter:
                if min_counts > 0:
                    sc.pp.filter_cells(adata, min_counts=min_counts)
                if min_genes > 0:
                    sc.pp.filter_cells(adata, min_genes=min_genes)
                if min_cells > 0:
                    sc.pp.filter_genes(adata, min_cells=min_cells)
            if do_norm:
                sc.pp.normalize_total(adata)
            if do_log1p:
                sc.pp.log1p(adata)
            st.session_state.pp_applied = True
            st.success("Preprocessing applied.")

# -------------------------
# Embeddings tab
# -------------------------
with emb_tab:
    st.subheader("Compute / view embeddings")
    embeddings = {
        k: v for k, v in adata.obsm.items()
        if k.startswith("X_") and getattr(v, "shape", None) is not None and v.shape[1] >= 2
    }

    def ensure_pca_neighbors_umap(adata: AnnData, n_pcs: int, n_neighbors: int = 15):
        if adata.isbacked:
            st.info("Compute embeddings requires in-memory AnnData. Reload without backed if needed.")
            return
        if "X_pca" not in adata.obsm:
            sc.tl.pca(adata, n_comps=n_pcs)
        # always refresh neighbors with desired n_neighbors
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=min(n_pcs, adata.obsm["X_pca"].shape[1]))
        if "X_umap" not in adata.obsm:
            sc.tl.umap(adata)

    c1, c2 = st.columns([2,1])
    with c1:
        if not embeddings:
            st.warning("No embeddings in .obsm (e.g., 'X_umap', 'X_pca'). You can compute them.")
        n_neighbors_umap = st.slider("n_neighbors (graph)", 5, 50, 15, 1)
        compute_umap = st.button("Compute/refresh PCA→Neighbors→UMAP", disabled=adata.isbacked)
        if compute_umap:
            ensure_pca_neighbors_umap(adata, n_pcs, n_neighbors=n_neighbors_umap)
            embeddings = {
                k: v for k, v in adata.obsm.items()
                if k.startswith("X_") and getattr(v, "shape", None) is not None and v.shape[1] >= 2
            }
            st.success("Embeddings ready.")

    with c2:
        if embeddings:
            default_basis = "X_umap" if "X_umap" in embeddings else sorted(embeddings.keys())[0]
            basis = st.selectbox(
                "Embedding", options=sorted(embeddings.keys()),
                index=sorted(embeddings.keys()).index(default_basis)
            )
            obs_cols = list(adata.obs.columns)
            preselect = obs_cols.index(label_hint) + 1 if label_hint in obs_cols else 0
            color_by = st.selectbox("Color by (obs)", options=[None] + obs_cols, index=preselect)

    if embeddings:
        coords = embeddings[basis]
        x, y = coords[:, 0], coords[:, 1]
        df = pd.DataFrame({"x": x, "y": y})
        if color_by is not None:
            df[color_by] = adata.obs[color_by].values
        fig = px.scatter(df, x="x", y="y", color=color_by, title=f"{basis} embedding", hover_data=df.columns.tolist())
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Feature matrix helper (cached)
# -------------------------
@st.cache_data(show_spinner=False)
def build_feature_matrix(_adata: AnnData, source: str, n_hvg: int, n_pcs: int, dataset_key: str):
    """Return (X, feature_names) for the chosen feature source.
    Note: _adata is unhashable; dataset_key is used to make the cache key stable.
    """
    adata = _adata  # keep local name

    if source == "Raw (X)":
        X = adata.X
        if sp.issparse(X):
            X = X.tocsr()
        else:
            X = np.asarray(X)
        feat_names = adata.var_names.to_list()
        return X, feat_names

    if source == "HVGs":
        if adata.isbacked:
            raise RuntimeError("HVG selection requires in-memory AnnData.")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=False, flavor="seurat_v3")
        hv_mask = adata.var["highly_variable"].values
        X = adata.X[:, hv_mask]
        if sp.issparse(X):
            X = X.tocsr()
        else:
            X = np.asarray(X)
        feat_names = adata.var_names[hv_mask].to_list()
        return X, feat_names

    if source == "PCA":
        if "X_pca" not in adata.obsm:
            if adata.isbacked:
                raise RuntimeError("PCA requires in-memory AnnData.")
            sc.tl.pca(adata, n_comps=n_pcs)
        X = adata.obsm["X_pca"][:, :n_pcs]
        feat_names = [f"PC{i+1}" for i in range(X.shape[1])]
        return np.asarray(X), feat_names

    raise ValueError("Unknown feature source")

# -------------------------
# Machine learning tab (with subtabs)
# -------------------------
with ml_tab:
    clf_tab, cluster_tab, predict_tab, markers_tab, deg_tab, export_tab = st.tabs(
        ["Classification", "Clustering", "Predict", "Markers", "DEG", "Export"]
    )

    # ---- Classification ----
    with clf_tab:
        obs_cols = list(adata.obs.columns)
        preselect = obs_cols.index(label_hint) if label_hint in obs_cols else 0 if obs_cols else 0
        target_col = st.selectbox(
            "Target (obs)", options=obs_cols if obs_cols else [""],
            index=preselect if obs_cols else 0,
            help="Column with ground‑truth labels (e.g., cell_type/condition)."
        )

        X, feat_names = build_feature_matrix(adata, feat_src, n_hvg, n_pcs, temp_path)
        y = adata.obs[target_col].astype(str).values if target_col else None

        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            clf_name = st.selectbox("Classifier", ["LogisticRegression", "RandomForest", "KNN"])
        with c2:
            test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        with c3:
            scale_X = st.toggle("Standardize features", value=(feat_src != "Raw (X)"))

        knn_k = None
        if clf_name == "KNN":
            knn_k = st.slider("k (neighbors for KNN)", 1, 51, 15, 2)

        if st.button("Train classifier", type="primary"):
            unique_y = np.unique(y)
            if unique_y.shape[0] < 2:
                st.error("Target has <2 classes. Pick another column.")
            else:
                try:
                    strat = y
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=strat
                    )
                except ValueError:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )

                if clf_name == "LogisticRegression":
                    base = LogisticRegression(max_iter=2000, solver="saga")
                elif clf_name == "RandomForest":
                    base = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
                else:
                    base = KNeighborsClassifier(n_neighbors=knn_k or 15)

                steps = []
                if scale_X and not sp.issparse(X):
                    steps.append(("scaler", StandardScaler()))
                steps.append(("clf", base))
                pipe = Pipeline(steps)
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")

                st.success(f"Accuracy: {acc:.3f} | F1‑weighted: {f1:.3f}")
                st.text("Classification report:" + classification_report(y_test, y_pred))

                cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
                cm_df = pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y))
                fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", title="Confusion matrix")
                st.plotly_chart(fig_cm, use_container_width=True)

                # Download model
                from datetime import datetime
                meta = {
                    "feature_source": feat_src,
                    "feature_names": feat_names if len(feat_names) <= 5000 else feat_names[:5000],
                    "n_features": len(feat_names),
                    "target_col": target_col,
                    "classes_": list(np.unique(y)),
                    "created_at": datetime.now().isoformat(),
                    "classifier": clf_name,
                }
                model_blob = BytesIO()
                joblib.dump({"pipeline": pipe, "meta": meta}, model_blob)
                model_blob.seek(0)
                st.download_button(
                    "Download trained model (.joblib)", data=model_blob,
                    file_name="bioml_model.joblib", mime="application/octet-stream"
                )
    # ---- DEG ----
    with deg_tab:
        st.write("Differential expression between two groups in an obs column")
        groupby = st.selectbox("Group column (obs)", list(adata.obs.columns))
        groups = sorted(adata.obs[groupby].dropna().astype(str).unique().tolist())
        if len(groups) < 2:
            st.info("Need at least 2 groups.")
        else:
            case = st.selectbox("Case group", groups, index=0)
            ref = st.selectbox("Reference group", groups, index=1 if len(groups) > 1 else 0)
            if st.button("Run DEG (Wilcoxon)"):
                ad = adata.copy() if not adata.isbacked else adata.to_memory()
                with st.spinner("Computing DEGs..."):
                    sc.tl.rank_genes_groups(ad, groupby=groupby, groups=[case], reference=ref, method="wilcoxon")
                    res = ad.uns["rank_genes_groups"]
                    df = pd.DataFrame({
                        "genes": res["names"][case],
                        "pvals": res["pvals"][case],
                        "pvals_adj": res["pvals_adj"][case],
                        "logfoldchanges": res["logfoldchanges"][case],
                    })
                    tiny = np.finfo(np.float64).tiny
                    df["neg_log10_pval"] = -np.log10(df["pvals"].clip(lower=tiny))
                    df["diffexpressed"] = "NS"
                    df.loc[(df["logfoldchanges"] > 1) & (df["pvals"] < 0.05), "diffexpressed"] = "UP"
                    df.loc[(df["logfoldchanges"] < -1) & (df["pvals"] < 0.05), "diffexpressed"] = "DOWN"
                st.dataframe(df.head(50), use_container_width=True)
                figv = px.scatter(df, x="logfoldchanges", y="neg_log10_pval",
                                  color="diffexpressed", title=f"Volcano: {case} vs {ref}")
                st.plotly_chart(figv, use_container_width=True)
                st.download_button("Download DEG table (.csv)", df.to_csv(index=False).encode(),
                                   "degs.csv", "text/csv")
    # ---- Clustering ----
    with cluster_tab:
        X, feat_names = build_feature_matrix(adata, feat_src, n_hvg, n_pcs, temp_path)
        method = st.selectbox("Clustering method", ["Leiden (graph)", "KMeans"], index=0)

        # feature matrix (Raw/HVGs/PCA)
        X, feat_names = build_feature_matrix(adata, feat_src, n_hvg, n_pcs, temp_path)

        # guard: backed mode can't write cluster labels
        if hasattr(adata, "isbacked") and adata.isbacked:
            st.error("Disable memory-light (backed) mode to run clustering.")
            st.stop()

        if method == "KMeans":
            from sklearn.cluster import KMeans, MiniBatchKMeans

            k = st.slider("k (clusters)", 2, 40, 10)

            # KMeans doesn't like very sparse/high-D; use MiniBatchKMeans when sparse
            use_mb = sp.issparse(X)
            # soft safeguard for massive raw feature spaces
            if X.shape[1] > 5000 and feat_src == "Raw (X)":
                st.warning("Raw genes are very high-dimensional; consider HVGs or PCA for faster clustering.")

            with st.spinner("Clustering (K-Means)..."):
                km = (MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10, batch_size=2048)
                      if use_mb else
                      KMeans(n_clusters=k, random_state=42, n_init=10))
                labels = km.fit_predict(X)

            adata.obs["kmeans_label"] = labels.astype(str)
            st.success("Done.")
            st.write(pd.Series(labels).value_counts().rename("cells per cluster"))
            if "X_umap" in adata.obsm:
                emb = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
                emb["cluster"] = adata.obs["kmeans_label"].values
                st.plotly_chart(px.scatter(emb, x="UMAP1", y="UMAP2", color="cluster"), use_container_width=True)

        else:  # Leiden
            try:
                import leidenalg  # noqa: F401
                import igraph  # noqa: F401
            except Exception:
                st.error("Leiden needs `python-igraph` and `leidenalg`. Install them and rerun.")
                st.stop()

            n_neighbors = st.slider("n_neighbors", 5, 50, 15)
            resolution = st.slider("resolution", 0.2, 2.0, 1.0)

            with st.spinner("Building graph + running Leiden..."):
                # ensure PCA exists for the neighbor graph
                if "X_pca" not in adata.obsm:
                    sc.pp.pca(adata, n_comps=min(50, adata.n_vars))
                sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca")
                sc.tl.leiden(adata, resolution=resolution, key_added="leiden")

            st.success("Done.")
            st.write(adata.obs["leiden"].value_counts().rename("cells per cluster"))
            if "X_umap" in adata.obsm:
                emb = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
                emb["cluster"] = adata.obs["leiden"].astype(str).values
                st.plotly_chart(px.scatter(emb, x="UMAP1", y="UMAP2", color="cluster"), use_container_width=True)

    # ---- Predict ----
    with predict_tab:
        st.write("Load a previously saved model and predict labels for the current dataset.")
        model_file = st.file_uploader("Upload trained model (.joblib or .pkl)", type=["joblib", "pkl"])
        if model_file is not None:
            try:
                obj = joblib.load(model_file)
                pipe = obj.get("pipeline", obj)
                meta = obj.get("meta", {})

                model_feat_src = meta.get("feature_source", feat_src)
                X_now, names_now = build_feature_matrix(adata, model_feat_src, n_hvg, n_pcs, temp_path)

                want_names = meta.get("feature_names")
                if want_names and model_feat_src == "Raw (X)":
                    name_to_idx = {n: i for i, n in enumerate(adata.var_names)}
                    idx = [name_to_idx[n] for n in want_names if n in name_to_idx]
                    if len(idx) == len(want_names):
                        X_now = X_now[:, idx]
                    else:
                        st.warning("Model feature names don't perfectly match this dataset; proceeding with intersection.")
                        X_now = X_now[:, idx]

                preds = pipe.predict(X_now)
                adata.obs["pred_label"] = preds.astype(str)
                st.success("Predictions added to adata.obs['pred_label'].")

                cols_to_show = [c for c in [meta.get("target_col", None), "pred_label"] if c in adata.obs.columns]
                st.dataframe(adata.obs[cols_to_show].head(20), use_container_width=True)

                pred_csv = pd.DataFrame({
                    "cell": adata.obs_names.astype(str),
                    "pred_label": adata.obs["pred_label"].astype(str)
                })
                st.download_button(
                    "Download predictions (.csv)", data=pred_csv.to_csv(index=False).encode(),
                    file_name="predictions.csv", mime="text/csv"
                )
            except Exception as e:
                st.exception(e)

    # ---- Markers ----
    with markers_tab:
        if adata.isbacked:
            st.info("Markers require in-memory AnnData. Reload without backed to compute.")
        else:
            obs_cols = list(adata.obs.columns)
            cluster_col = st.selectbox(
                "Cluster column (obs)",
                options=[c for c in obs_cols if
                         adata.obs[c].dtype.name == "category" or adata.obs[c].dtype == object] or obs_cols,
            )
            method = st.selectbox("Method", ["wilcoxon", "t-test", "logreg"], index=0)
            top_n = st.slider("Top genes per cluster", 5, 50, 10, 1)

            if st.button("Find markers"):
                try:
                    # --- validation & casting ---
                    g = adata.obs[cluster_col].copy()
                    # drop NA labels
                    valid = ~g.isna()
                    if valid.sum() < 2:
                        st.error("Not enough labeled cells for marker testing.")
                        st.stop()
                    if valid.sum() < len(g):
                        st.warning(f"Dropped {(~valid).sum()} cells with NA in '{cluster_col}'.")
                    g = g[valid].astype("category")
                    ad_sub = adata[valid].copy()  # isolate to avoid mixing NA rows

                    if g.cat.categories.size < 2:
                        st.error(f"'{cluster_col}' has <2 groups after filtering.")
                        st.stop()

                    # --- control compute size ---
                    # cap the number of genes Scanpy ranks (huge speed win with large gene sets)
                    n_genes_cap = int(min(3000, ad_sub.n_vars))  # tune if you want
                    use_raw = getattr(ad_sub, "raw", None) is not None

                    # method-specific kwargs
                    rg_kwargs = {}
                    if method == "wilcoxon":
                        rg_kwargs["tie_correct"] = True
                    if method == "logreg":
                        rg_kwargs["max_iter"] = 5000

                    with st.spinner("Ranking marker genes..."):
                        sc.tl.rank_genes_groups(
                            ad_sub,
                            groupby=cluster_col,
                            method=method,
                            n_genes=n_genes_cap,
                            use_raw=use_raw,
                            **rg_kwargs,
                        )

                    # --- collect results robustly across Scanpy versions ---
                    try:
                        # Newer Scanpy: pass group=None to get long-form DF
                        df = sc.get.rank_genes_groups_df(ad_sub, group=None)
                        tiny = np.finfo(np.float64).tiny  # ~2.2e-308

                        # Clip so zeros become the smallest positive float, then add -log10 columns
                        df["pvals_clipped"] = df["pvals"].clip(lower=tiny)
                        df["pvals_adj_clipped"] = df["pvals_adj"].clip(lower=tiny)
                        df["-log10(pval)"] = -np.log10(df["pvals_clipped"])
                        df["-log10(padj)"] = -np.log10(df["pvals_adj_clipped"])
                    except TypeError:
                        # Older: sometimes accepts None without keyword, or returns per-group
                        df = sc.get.rank_genes_groups_df(ad_sub, None)
                        tiny = np.finfo(np.float64).tiny  # ~2.2e-308

                        # Clip so zeros become the smallest positive float, then add -log10 columns
                        df["pvals_clipped"] = df["pvals"].clip(lower=tiny)
                        df["pvals_adj_clipped"] = df["pvals_adj"].clip(lower=tiny)
                        df["-log10(pval)"] = -np.log10(df["pvals_clipped"])
                        df["-log10(padj)"] = -np.log10(df["pvals_adj_clipped"])

                    # keep top_n per cluster by score (descending)
                    out = (
                        df.sort_values(["group", "scores"], ascending=[True, False])
                        .groupby("group", as_index=False)
                        .head(top_n)
                    )

                    st.success("Markers computed.")
                    st.dataframe(out, use_container_width=True)

                    # quick bar plot for one group
                    gpick = st.selectbox("Plot markers for cluster", sorted(out["group"].unique()))
                    gdf = out[out["group"] == gpick]
                    figm = px.bar(gdf, x="names", y="scores", title=f"Top markers — {gpick}")
                    figm.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(figm, use_container_width=True)

                    # export CSV
                    st.download_button(
                        "Download markers (.csv)",
                        data=out.to_csv(index=False).encode(),
                        file_name="markers.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.exception(e)

    # ---- Export ----
    with export_tab:
        if adata.isbacked:
            st.info("Export requires in-memory AnnData. Reload without backed to export.")
        else:
            if st.button("Prepare .h5ad for download"):
                try:
                    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".h5ad").name
                    adata.write(tmp_out)
                    with open(tmp_out, "rb") as fh:
                        st.download_button(
                            "Download current dataset (.h5ad)", data=fh.read(),
                            file_name="dataset_updated.h5ad", mime="application/octet-stream"
                        )
                except Exception as e:
                    st.exception(e)

# -------------------------
# Exploration tab (Enrichment, Pathways, TF activity)
# -------------------------
with explore_tab:
    enr_tab, pathways_tab, tfs_tab = st.tabs([
        "Enrichment (GO/KEGG)", "Pathways (ssGSEA)", "TF activity"
    ])

    # ---------- Enrichment (GO/KEGG) ----------
    with enr_tab:
        st.write("Over-representation analysis (ORA) on a gene list (e.g., top markers of a cluster).")
        # Choose a cluster column and one group to characterize
        cluster_cols = [c for c in adata.obs.columns if c in ("leiden", "cluster_leiden", "kmeans_label", "cluster_kmeans") or "cluster" in c.lower()]
        if not cluster_cols:
            st.info("No clustering columns found. Run Leiden/K-Means first in the Machine learning tab.")
        else:
            cluster_col = st.selectbox("Cluster column", cluster_cols, index=0)
            groups = adata.obs[cluster_col].astype(str).unique().tolist()
            pick = st.selectbox("Cluster to analyze", sorted(groups))
            top_n = st.slider("Top genes", 20, 500, 200, 10,
                              help="Use top-N up genes for enrichment.")
            org = st.selectbox("Organism", ["Human", "Mouse"], index=0)
            # Optional: local GMT to be fully offline
            gmt = st.file_uploader("Optional: upload GMT gene set library", type=["gmt"], help="If empty, uses Enrichr libraries (needs internet).")

            if st.button("Run enrichment"):
                # Build a gene list from rank_genes_groups for the chosen cluster (compute if needed)
                try:
                    # Ensure we have markers for this cluster vs rest
                    sc.tl.rank_genes_groups(adata, groupby=cluster_col, groups=[pick], method="wilcoxon")
                    res = adata.uns["rank_genes_groups"]
                    genes = pd.Index(res["names"][pick]).astype(str).tolist()[:top_n]
                    if len(genes) < 5:
                        st.error("Not enough genes for enrichment.")
                    else:
                        try:
                            import gseapy as gp
                        except ImportError:
                            st.error("gseapy is not installed. Add `gseapy` to requirements.txt.")
                        else:
                            with st.spinner("Running ORA..."):
                                gene_sets = [gmt] if gmt is not None else ["GO_Biological_Process_2021", "KEGG_2021_Human" if org=="Human" else "KEGG_2021_Mouse"]
                                enr = gp.enrichr(gene_list=genes, gene_sets=gene_sets, organism=org, outdir=None, no_plot=True)
                                res_df = enr.results.sort_values("Adjusted P-value").head(20)
                                if res_df.empty:
                                    st.info("No significant terms found.")
                                else:
                                    show_cols = ["Term","Adjusted P-value","Overlap","Combined Score"]
                                    st.dataframe(res_df[show_cols], use_container_width=True)
                                    st.download_button(
                                        "Download enrichment (.csv)",
                                        res_df.to_csv(index=False).encode(),
                                        file_name="enrichment_results.csv",
                                        mime="text/csv",
                                    )
                except Exception as e:
                    st.exception(e)

    # ---------- Pathways (ssGSEA / GSVA-like) ----------
    with pathways_tab:
        st.write("Estimate pathway activity per cluster (fast via ssGSEA on cluster means).")
        cluster_cols = [c for c in adata.obs.columns if c in ("leiden", "cluster_leiden", "kmeans_label", "cluster_kmeans") or "cluster" in c.lower()]
        gs_file = st.file_uploader("Upload gene sets (.gmt)", type=["gmt"])
        if not cluster_cols:
            st.info("No clustering columns found. Run Leiden/K-Means first.")
        elif gs_file is None:
            st.info("Upload a GMT gene set file to continue (e.g., MSigDB subset).")
        else:
            cluster_col = st.selectbox("Cluster column", cluster_cols, index=0)
            if st.button("Run ssGSEA on cluster means"):
                try:
                    import gseapy as gp
                except ImportError:
                    st.error("gseapy is not installed. Add `gseapy` to requirements.txt.")
                else:
                    with st.spinner("Computing cluster means and ssGSEA..."):
                        # Expression matrix: cells × genes -> group by cluster -> genes × clusters
                        X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
                        expr = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
                        M = expr.groupby(adata.obs[cluster_col].astype(str)).mean().T
                        # ssGSEA expects features (genes) in rows, samples (clusters) in columns
                        ssg = gp.ssgsea(data=M, gene_sets=gs_file, outdir=None, sample_norm_method="rank")
                        scores = ssg.res2d  # pathways × clusters
                        st.dataframe(scores.head(20), use_container_width=True)
                        fig = px.imshow(scores, aspect="auto", title="ssGSEA scores (pathway × cluster)")
                        st.plotly_chart(fig, use_container_width=True)
                        st.download_button(
                            "Download ssGSEA matrix (.csv)",
                            scores.to_csv().encode(),
                            file_name="ssgsea_scores.csv", mime="text/csv"
                        )

    # ---------- TF activity (regulons; lightweight with decoupler ULM) ----------
    with tfs_tab:
        st.write("Score TF program activity using a prior TF→target network (upload CSV).")
        st.caption("Upload a CSV with columns: source (TF), target (gene), weight (signed). For fully offline use.")
        reg_file = st.file_uploader("TF regulons (.csv)", type=["csv"])
        cluster_cols = [c for c in adata.obs.columns if c in ("leiden", "cluster_leiden", "kmeans_label", "cluster_kmeans") or "cluster" in c.lower()]
        if not cluster_cols:
            st.info("No clustering columns found. Run Leiden/K-Means first.")
        elif reg_file is None:
            st.info("Upload a regulon CSV to continue.")
        else:
            cluster_col = st.selectbox("Cluster column", cluster_cols, index=0, key="tf_cluster_select")
            if st.button("Score TFs on cluster means"):
                try:
                    import decoupler as dc
                except ImportError:
                    st.error("decoupler is not installed. Add `decoupler` to requirements.txt.")
                else:
                    try:
                        net = pd.read_csv(reg_file)
                        expected = {"source","target","weight"}
                        if not expected.issubset(set(map(str.lower, net.columns))):
                            st.error("CSV must have columns: source, target, weight (case-insensitive).")
                        else:
                            # Normalize column names
                            cols_lower = {c.lower(): c for c in net.columns}
                            net = net.rename(columns={cols_lower["source"]:"source", cols_lower["target"]:"target", cols_lower["weight"]:"weight"})
                            with st.spinner("Computing TF activity..."):
                                X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
                                expr = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
                                M = expr.groupby(adata.obs[cluster_col].astype(str)).mean().T  # genes × clusters
                                acts = dc.run_ulm(mat=M, net=net, source="source", target="target", weight="weight")
                                # acts: sources (TFs) × samples (clusters)
                                st.dataframe(acts.head(20), use_container_width=True)
                                fig = px.imshow(acts, aspect="auto", title="TF activity (TF × cluster)")
                                st.plotly_chart(fig, use_container_width=True)
                                st.download_button(
                                    "Download TF activity (.csv)",
                                    acts.to_csv().encode(),
                                    file_name="tf_activity_ulm.csv", mime="text/csv"
                                )
                    except Exception as e:
                        st.exception(e)

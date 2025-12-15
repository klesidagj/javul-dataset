# --- FILE: dr.py ---
#!/usr/bin/env python3
"""
Dimensionality reduction wrappers (PCA, t-SNE, UMAP) and metrics.
"""
from typing import Dict, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
import umap

R_SEED = 42

class Embeddings2D:
    def __init__(self, pca2: np.ndarray, tsne2: np.ndarray, umap2: np.ndarray, metrics: Dict = None):
        self.pca = pca2
        self.tsne = tsne2
        self.umap = umap2
        self.metrics = metrics or {}

def run_dr(X_scaled: np.ndarray, X_pre: np.ndarray, labels) -> Embeddings2D:
    pca2 = PCA(n_components=2, random_state=R_SEED).fit_transform(X_scaled)
    tsne_in = X_pre if X_pre.shape[1] <= 50 else PCA(n_components=50, random_state=R_SEED).fit_transform(X_scaled)
    perp = 30 if len(labels) > 100 else max(5, max(5, len(labels)//5))
    tsne2 = TSNE(n_components=2, init='pca', perplexity=perp, random_state=R_SEED).fit_transform(tsne_in)
    um = umap.UMAP(n_neighbors=15, min_dist=0.1, init='random', metric='cosine', random_state=R_SEED)
    um2 = um.fit_transform(X_pre)
    metrics = {}
    try:
        if len(set(labels)) > 1 and len(labels) >= 3:
            metrics['sil_pca'] = float(silhouette_score(pca2, labels))
            metrics['sil_tsne'] = float(silhouette_score(tsne2, labels))
            metrics['sil_umap'] = float(silhouette_score(um2, labels))
            knn = KNeighborsClassifier(n_neighbors=5)
            cv = min(5, max(2, len(labels)))
            metrics['knn_pca'] = float(cross_val_score(knn, pca2, labels, cv=cv).mean())
            metrics['knn_tsne'] = float(cross_val_score(knn, tsne2, labels, cv=cv).mean())
            metrics['knn_umap'] = float(cross_val_score(knn, um2, labels, cv=cv).mean())
    except Exception as exc:
        metrics['error'] = str(exc)
    return Embeddings2D(pca2, tsne2, um2, metrics)
# --- FILE: preprocess.py ---
#!/usr/bin/env python3
"""
Preprocessing helpers: variance filter, scaling, jitter, PCA pre-reduction.
"""
from typing import Tuple, Optional, Sequence
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess(X: np.ndarray,
               graph_cols_orig: Optional[Sequence[int]] = None,
               jitter: float = 1e-6,
               pca_target: int = 30,
               seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    var = X.var(axis=0)
    nz_mask = var > 0
    X = X[:, nz_mask]
    if graph_cols_orig is not None:
        orig_len = len(var)
        graph_mask_orig = np.zeros(orig_len, dtype=bool)
        graph_mask_orig[list(graph_cols_orig)] = True
        graph_mask = graph_mask_orig[nz_mask]
        if graph_mask.any():
            X[:, graph_mask] = np.log1p(X[:, graph_mask])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rng = np.random.default_rng(seed)
    Xs = Xs + rng.normal(scale=jitter, size=Xs.shape)
    n_comp = min(pca_target, max(2, Xs.shape[1]-1))
    if Xs.shape[1] > n_comp:
        pca = PCA(n_components=n_comp, random_state=seed)
        Xr = pca.fit_transform(Xs)
    else:
        Xr = Xs
    return Xr, Xs, nz_mask

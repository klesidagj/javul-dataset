# --- FILE: metrics.py ---
#!/usr/bin/env python3
"""
Quantitative metrics helpers.
"""
from typing import Any, Dict, List, Tuple
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.cluster import KMeans

def safe_silhouette(X: np.ndarray, labels) -> float:
    try:
        if len(np.unique(labels)) < 2 or len(labels) < 4:
            return None
        return float(silhouette_score(X, labels))
    except Exception:
        return None

def knn_cv_acc(X: np.ndarray, labels, n_neighbors: int = 5) -> float:
    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    min_count = counts.min() if len(counts) > 0 else 0
    n_splits = min(5, int(min_count)) if min_count >= 2 else 0
    if n_splits < 2:
        return None
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores = cross_val_score(knn, X, labels, cv=skf, scoring='accuracy', n_jobs=1)
        return float(scores.mean())
    except Exception:
        return None

def nn_local_agreement(X: np.ndarray, labels, k: int = 1) -> float:
    labels = np.array(labels)
    if len(labels) <= 1:
        return None
    try:
        nn = NearestNeighbors(n_neighbors=k+1).fit(X)
        _, indices = nn.kneighbors(X)
        neighbors = indices[:, 1:k+1]
        agrees = []
        for i in range(len(labels)):
            neigh_labels = labels[neighbors[i]]
            agrees.append(np.mean(neigh_labels == labels[i]))
        return float(np.mean(agrees))
    except Exception:
        return None

def cluster_purity_weighted(coords: np.ndarray, labels, n_clusters: int) -> Tuple[float, List[Dict[str,Any]]]:
    if len(coords) == 0:
        return None, []
    try:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clus = km.fit_predict(coords)
    except Exception:
        return None, []
    details = []
    total = len(labels)
    weighted_sum = 0.0
    for c in sorted(set(clus)):
        idx = np.where(clus == c)[0]
        labs = np.array(labels)[idx]
        if len(labs) == 0:
            purity = 0.0
            dom_label = None
        else:
            cnt = Counter(labs)
            dom_label, dom_count = cnt.most_common(1)[0]
            purity = dom_count / len(labs)
        weighted_sum += purity * (len(labs) / total)
        details.append({
            'cluster': int(c),
            'size': int(len(labs)),
            'dominant_label': str(dom_label),
            'purity': float(purity)
        })
    return float(weighted_sum), details
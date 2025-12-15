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

def filter_labels(X, y, min_count=5):
    counts = Counter(y)
    mask = [counts[label] >= min_count for label in y]
    return X[mask], y[mask]

from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def filter_labels(X, y, min_count=5):
    """
    Remove samples whose class appears fewer than `min_count` times.
    """
    counts = Counter(y)
    mask = np.array([counts[label] >= min_count for label in y])
    return X[mask], y[mask]


def knn_cv_accuracy(X, y, n_neighbors=5, max_folds=5, min_class_count=5,):
    """
    Safe KNN cross-validation accuracy.
    Returns NaN if not statistically valid.
    """
    X_f, y_f = filter_labels(X, y, min_class_count)

    if len(set(y_f)) < 2:
        return float("nan")

    # Determine safe number of folds
    class_counts = Counter(y_f)
    n_splits = min(max_folds, min(class_counts.values()))

    if n_splits < 2:
        return float("nan")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    scores = cross_val_score(
        knn,
        X_f,
        y_f,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
    )

    return float(scores.mean())

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
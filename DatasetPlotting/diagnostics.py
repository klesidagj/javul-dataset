#!/usr/bin/env python3
"""
quantitative_checks.py

Compute quantitative metrics for 2D embeddings saved by the viz script.

Usage:
  python quantitative_checks.py --input embeddings_db_1756997121.csv --out_prefix metrics_out

Outputs:
 - CSV with per-(feature_set,projection) metrics: <out_prefix>_summary_metrics.csv
 - JSON with cluster purity details: <out_prefix>_cluster_details.json
 - Printed human-readable summary

Author: generated
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

def safe_silhouette(X, labels):
    """Compute silhouette if valid, else None."""
    try:
        if len(np.unique(labels)) < 2 or len(labels) < 4:
            return None
        return float(silhouette_score(X, labels))
    except Exception:
        return None

def knn_cv_acc(X, labels, n_neighbors=5):
    """Stratified K-Fold CV for kNN; returns mean accuracy or None if not possible."""
    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    min_count = counts.min() if len(counts) > 0 else 0
    # need at least 2 folds and each class >= n_splits
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

def nn_local_agreement(X, labels, k=1):
    """Compute fraction of points whose k-th nearest neighbor(s) share the same label.
       For k=1 we use the single nearest neighbor excluding self.
    """
    labels = np.array(labels)
    if len(labels) <= 1:
        return None
    try:
        nn = NearestNeighbors(n_neighbors=k+1).fit(X)
        distances, indices = nn.kneighbors(X)
        # indices[:,0] is self; consider indices[:,1:k+1]
        neighbors = indices[:, 1:k+1]
        agrees = []
        for i in range(len(labels)):
            neigh_labels = labels[neighbors[i]]
            agrees.append(np.mean(neigh_labels == labels[i]))
        return float(np.mean(agrees))
    except Exception:
        return None

def cluster_purity_weighted(coords, labels, n_clusters):
    """Compute cluster purity using KMeans with n_clusters.
       Returns weighted purity (weight = cluster size), and details per cluster.
    """
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
        labs = labels[idx]
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

def summarize_feature_projection(df_subset, proj_x, proj_y):
    """Return metric dict for given projection columns in subset dataframe."""
    coords = df_subset[[proj_x, proj_y]].values
    labels = df_subset['cwe_id'].astype(str).values
    # label codes for silhouette and classifiers
    label_codes = pd.Categorical(labels).codes
    metrics = {}
    metrics['n_samples'] = int(len(labels))
    metrics['n_labels'] = int(len(np.unique(labels)))
    metrics['label_counts'] = dict(Counter(labels))
    metrics['silhouette'] = safe_silhouette(coords, label_codes)
    metrics['knn_cv_acc'] = knn_cv_acc(coords, label_codes, n_neighbors=5)
    metrics['nn_agreement_k1'] = nn_local_agreement(coords, labels, k=1)
    metrics['nn_agreement_k5'] = nn_local_agreement(coords, labels, k=5) if len(labels) > 6 else None
    # cluster purities for several k
    purities = {}
    details_all = {}
    cluster_ks = [5, 10, 20, 30]  # sensible defaults
    for k in cluster_ks:
        if k >= len(labels):
            purities[f'k{k}'] = None
            details_all[f'k{k}'] = []
            continue
        wpur, details = cluster_purity_weighted(coords, labels, n_clusters=k)
        purities[f'k{k}'] = wpur
        details_all[f'k{k}'] = details
    metrics['cluster_purities'] = purities
    metrics['cluster_details'] = details_all
    return metrics

def run_all_metrics(emb_csv, out_prefix):
    df = pd.read_csv(emb_csv)
    # check expected columns
    required_cols = {'feature_set','cwe_id'}
    if not required_cols.issubset(set(df.columns)):
        raise SystemExit(f"Input CSV missing required columns. Found columns: {df.columns.tolist()}")
    projections = {
        'PCA': ('pca_x','pca_y'),
        'tSNE': ('tsne_x','tsne_y'),
        'UMAP': ('umap_x','umap_y')
    }
    feature_sets = sorted(df['feature_set'].unique())
    results = []
    cluster_details_out = {}
    for fs in feature_sets:
        df_fs = df[df['feature_set'] == fs].reset_index(drop=True)
        if df_fs.empty:
            continue
        for pname, (px, py) in projections.items():
            if px not in df_fs.columns or py not in df_fs.columns:
                print(f"Skipping {fs} {pname}: missing columns {px},{py}")
                continue
            print(f"Processing {fs} / {pname} ({len(df_fs)} samples, labels={df_fs['cwe_id'].nunique()})")
            metrics = summarize_feature_projection(df_fs, px, py)
            row = {
                'feature_set': fs,
                'projection': pname,
                'n_samples': metrics['n_samples'],
                'n_labels': metrics['n_labels'],
                'silhouette': metrics['silhouette'],
                'knn_cv_acc': metrics['knn_cv_acc'],
                'nn_agreement_k1': metrics['nn_agreement_k1'],
                'nn_agreement_k5': metrics['nn_agreement_k5'],
                'cluster_purity_k5': metrics['cluster_purities'].get('k5') if 'k5' in metrics['cluster_purities'] else metrics['cluster_purities'].get('k5', None), # placeholder safe access
            }
            # store full metrics and details
            results.append({**row, 'label_counts': metrics['label_counts'], 'cluster_purities': metrics['cluster_purities']})
            cluster_details_out[f"{fs}::{pname}"] = metrics['cluster_details']
    # write results csv
    out_csv = f"{out_prefix}_summary_metrics.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print("\nSaved summary metrics to", out_csv)
    # write cluster details json
    out_json = f"{out_prefix}_cluster_details.json"
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump(cluster_details_out, fh, indent=2)
    print("Saved detailed cluster info to", out_json)
    # Print readable summary
    print("\nReadable summary:")
    df_sum = pd.read_csv(out_csv)
    for _, r in df_sum.iterrows():
        print("----")
        print(f"{r['feature_set']} / {r['projection']}: samples={r['n_samples']} labels={r['n_labels']}")
        print(f"  silhouette={r['silhouette']}, knn_cv_acc={r['knn_cv_acc']}, nn_k1={r['nn_agreement_k1']}")
        print(f"  cluster_purity_k5 (weighted)={r.get('cluster_purity_k5')}")
    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='embeddings CSV created by viz script')
    parser.add_argument('--out_prefix', '-o', default='metrics_out', help='prefix for output files')
    args = parser.parse_args()
    run_all_metrics(args.input, args.out_prefix)
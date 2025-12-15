# --- FILE: main.py ---
#!/usr/bin/env python3
"""
Orchestrator CLI for loading, building features, running DR, plotting, and saving outputs.
"""
import argparse
import json
import logging
import time
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

from data.db_loader import load_from_csv, load_from_db
from features.features import build_feature_matrices
from features.preprocess import preprocess
from metrics.dr import run_dr
from plot import plot_grid

from metrics.metrics import cluster_purity_weighted

LOG = logging.getLogger("viz_refactor")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_SQL = """
SELECT id, cwe_id, is_vulnerable,
       css_vector, ast_graph, cfg_graph, dfg_graph
FROM javul_cl
WHERE cwe_id IS NOT NULL
  AND cwe_id <> 'NA'
  AND css_vector IS NOT NULL
  AND ast_graph IS NOT NULL
  AND cfg_graph IS NOT NULL
  AND dfg_graph IS NOT NULL
"""

DB_SAMPLE = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "klesi",
    "password": ""
}

def orchestrate(df_raw: pd.DataFrame, out_prefix: str, dedup: bool, max_per_class: int):
    bundle = build_feature_matrices(df_raw)
    df_keep = bundle.df
    css_X = bundle.css
    graph_X = bundle.graph
    combined_X = bundle.combined
    LOG.info("Parsed rows kept: %d", len(df_keep))
    if dedup:
        rounded = np.round(combined_X, 6)
        uniq, idx = np.unique(rounded, axis=0, return_index=True)
        idx_sorted = np.sort(idx)
        dup_count = combined_X.shape[0] - uniq.shape[0]
        LOG.info("Dedup removed %d duplicate vectors", dup_count)
        combined_X = combined_X[idx_sorted, :]
        css_X = css_X[idx_sorted, :]
        graph_X = graph_X[idx_sorted, :]
        df_keep = df_keep.iloc[idx_sorted].reset_index(drop=True)
    if max_per_class and max_per_class > 0:
        sampled_idx = []
        rng = np.random.default_rng(42)
        max_per = int(max_per_class)
        for lab in df_keep['cwe_id'].unique():
            ids = np.where(df_keep['cwe_id'].values == lab)[0]
            if len(ids) > max_per:
                chosen = rng.choice(ids, size=max_per, replace=False)
            else:
                chosen = ids
            sampled_idx.extend(chosen.tolist())
        sampled_idx = sorted(sampled_idx)
        combined_X = combined_X[sampled_idx, :]
        css_X = css_X[sampled_idx, :]
        graph_X = graph_X[sampled_idx, :]
        df_keep = df_keep.iloc[sampled_idx].reset_index(drop=True)
        LOG.info("After sampling per class, rows: %d", len(df_keep))
    css_dim = css_X.shape[1]
    graph_cols_orig = list(range(css_dim, css_dim + graph_X.shape[1]))
    outputs = {}
    labels = df_keep['cwe_id'].values
    for name, X in [('CSS', css_X), ('Graph', graph_X), ('Combined', combined_X)]:
        LOG.info("Preprocessing %s (shape=%s)", name, X.shape)
        if name == 'Combined':
            gr_cols = graph_cols_orig
        elif name == 'Graph':
            gr_cols = list(range(X.shape[1]))
        else:
            gr_cols = None
        X_pre, X_scaled, nz_mask = preprocess(X, graph_cols_orig=gr_cols, jitter=1e-6, pca_target=30)
        emb = run_dr(X_scaled, X_pre, labels)
        outputs[name] = (emb.pca, emb.tsne, emb.umap, emb.metrics)
        LOG.info("%s metrics: %s", name, emb.metrics)
    ts = int(time.time())
    out_png = f"{out_prefix}_{ts}.png"
    plot_grid(outputs, list(df_keep['cwe_id'].values), out_png)
    LOG.info("Saved plot: %s", out_png)
    emb_records = []
    for name in ['CSS', 'Graph', 'Combined']:
        p2, t2, u2, metrics = outputs[name]
        for i in range(len(df_keep)):
            emb_records.append({
                'id': int(df_keep.iloc[i]['id']),
                'cwe_id': df_keep.iloc[i]['cwe_id'],
                'feature_set': name,
                'pca_x': float(p2[i, 0]), 'pca_y': float(p2[i, 1]),
                'tsne_x': float(t2[i, 0]), 'tsne_y': float(t2[i, 1]),
                'umap_x': float(u2[i, 0]), 'umap_y': float(u2[i, 1])
            })
    emb_df = pd.DataFrame.from_records(emb_records)
    out_csv = f"{out_prefix}_embeddings_{ts}.csv"
    emb_df.to_csv(out_csv, index=False)
    LOG.info("Saved embeddings CSV: %s", out_csv)
    cluster_details = {}
    for name in ['CSS', 'Graph', 'Combined']:
        p2, _, _, _ = outputs[name]
        # compute simple cluster purity for a few k's (skip if too few samples)
        cds = {}
        for k in (5, 10, 20):
            if k < len(labels):
                wp, details = cluster_purity_weighted(p2, df_keep['cwe_id'].values, n_clusters=k)
                cds[f'k{k}'] = {'weighted_purity': wp, 'details': details}
            else:
                cds[f'k{k}'] = {'weighted_purity': None, 'details': []}
        cluster_details[name] = cds
    out_json = f"{out_prefix}_cluster_details_{ts}.json"
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump(cluster_details, fh, indent=2)
    LOG.info("Saved cluster details JSON: %s", out_json)
    LOG.info("Done.")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', help='input CSV path (preferred for quick tests)')
    p.add_argument('--use_db', action='store_true', help='fetch from DB using DB config hardcoded or env')
    p.add_argument('--sql', default=DEFAULT_SQL, help='SQL query to use when --use_db')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--dedup', type=lambda x: x.lower() in ('1','true','yes'), default=True)
    p.add_argument('--max_per_class', type=int, default=300)
    p.add_argument('--out_prefix', '-o', default='metrics_out')
    return p.parse_args()

def main():
    args = parse_args()
    if not args.csv and not args.use_db:
        raise SystemExit("Provide --csv or --use_db.")
    if args.csv:
        df = load_from_csv(args.csv)
    else:
        df = load_from_db(args.sql, DB_SAMPLE, limit=args.limit)
    orchestrate(df, args.out_prefix, dedup=args.dedup, max_per_class=args.max_per_class)

if __name__ == '__main__':
    main()
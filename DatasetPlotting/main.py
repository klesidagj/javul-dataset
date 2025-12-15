#!/usr/bin/env python3
"""
main.py

Orchestrator CLI for:
- loading data (CSV or DB)
- building features
- preprocessing
- dimensionality reduction (PCA / t-SNE / UMAP)
- plotting
- saving embeddings + cluster diagnostics
"""

import argparse
import json
import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

from data.db_loader import load_from_csv, load_from_db
from features.features import build_feature_matrices
from features.preprocess import preprocess
from metrics.dr import run_dr
from metrics.metrics import cluster_purity_weighted
from plot import plot_grid

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
LOG = logging.getLogger("viz_refactor")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------
def orchestrate(
    df_raw: pd.DataFrame,
    out_prefix: str,
    dedup: bool,
    max_per_class: Optional[int],
):
    LOG.info(
        "Initial dataset: rows=%d, unique CWE labels=%d",
        len(df_raw),
        df_raw["cwe_id"].nunique(),
    )

    bundle = build_feature_matrices(df_raw)
    df_keep = bundle.df
    css_X = bundle.css
    graph_X = bundle.graph
    combined_X = bundle.combined

    LOG.info("Parsed rows kept: %d", len(df_keep))

    # ---------------- Deduplication ----------------
    if dedup:
        rounded = np.round(combined_X, 6)
        _, idx = np.unique(rounded, axis=0, return_index=True)
        idx = np.sort(idx)

        removed = combined_X.shape[0] - len(idx)
        LOG.info("Dedup removed %d duplicate vectors", removed)

        combined_X = combined_X[idx]
        css_X = css_X[idx]
        graph_X = graph_X[idx]
        df_keep = df_keep.iloc[idx].reset_index(drop=True)

    # ---------------- Per-class sampling ----------------
    if max_per_class and max_per_class > 0:
        rng = np.random.default_rng(42)
        sampled_idx = []

        for lab in df_keep["cwe_id"].unique():
            ids = np.where(df_keep["cwe_id"].values == lab)[0]
            if len(ids) > max_per_class:
                ids = rng.choice(ids, size=max_per_class, replace=False)
            sampled_idx.extend(ids.tolist())

        sampled_idx = sorted(sampled_idx)

        combined_X = combined_X[sampled_idx]
        css_X = css_X[sampled_idx]
        graph_X = graph_X[sampled_idx]
        df_keep = df_keep.iloc[sampled_idx].reset_index(drop=True)

        LOG.info("After sampling per class, rows: %d", len(df_keep))

    labels = df_keep["cwe_id"].values

    # Graph column indices for preprocessing
    css_dim = css_X.shape[1]
    graph_cols_orig = list(range(css_dim, css_dim + graph_X.shape[1]))

    outputs = {}

    # ---------------- Feature sets ----------------
    for name, X in [
        ("CSS", css_X),
        ("Graph", graph_X),
        ("Combined", combined_X),
    ]:
        LOG.info("Preprocessing %s (shape=%s)", name, X.shape)

        if name == "Combined":
            graph_cols = graph_cols_orig
        elif name == "Graph":
            graph_cols = list(range(X.shape[1]))
        else:
            graph_cols = None

        X_pre, X_scaled, _ = preprocess(
            X,
            graph_cols_orig=graph_cols,
            jitter=1e-6,
            pca_target=30,
        )

        emb = run_dr(X_scaled, X_pre, labels)
        outputs[name] = (emb.pca, emb.tsne, emb.umap, emb.metrics)

        LOG.info("%s metrics: %s", name, emb.metrics)

    # ---------------- Plot ----------------
    ts = int(time.time())
    out_png = f"{out_prefix}_{ts}.png"
    plot_grid(outputs, labels, out_png)
    LOG.info("Saved plot: %s", out_png)

    # ---------------- Save embeddings ----------------
    records = []
    for name, (p2, t2, u2, _) in outputs.items():
        for i in range(len(df_keep)):
            records.append(
                {
                    "id": int(df_keep.iloc[i]["id"]),
                    "cwe_id": df_keep.iloc[i]["cwe_id"],
                    "feature_set": name,
                    "pca_x": float(p2[i, 0]),
                    "pca_y": float(p2[i, 1]),
                    "tsne_x": float(t2[i, 0]),
                    "tsne_y": float(t2[i, 1]),
                    "umap_x": float(u2[i, 0]),
                    "umap_y": float(u2[i, 1]),
                }
            )

    emb_df = pd.DataFrame.from_records(records)
    out_csv = f"{out_prefix}_embeddings_{ts}.csv"
    emb_df.to_csv(out_csv, index=False)
    LOG.info("Saved embeddings CSV: %s", out_csv)

    # ---------------- Cluster purity ----------------
    cluster_details = {}
    for name, (p2, _, _, _) in outputs.items():
        cds = {}
        for k in (5, 10, 20):
            if k < len(labels):
                wp, details = cluster_purity_weighted(
                    p2, labels, n_clusters=k
                )
                cds[f"k{k}"] = {
                    "weighted_purity": wp,
                    "details": details,
                }
            else:
                cds[f"k{k}"] = {
                    "weighted_purity": None,
                    "details": [],
                }
        cluster_details[name] = cds

    out_json = f"{out_prefix}_cluster_details_{ts}.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(cluster_details, fh, indent=2)

    LOG.info("Saved cluster details JSON: %s", out_json)
    LOG.info("Done.")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", help="input CSV path")
    p.add_argument("--use_db", action="store_true")
    p.add_argument("--sql", default=DEFAULT_SQL)
    p.add_argument("--limit", type=int)
    p.add_argument("--dedup", type=lambda x: x.lower() in ("1", "true", "yes"), default=True)
    p.add_argument("--max_per_class", type=int, default=300)
    p.add_argument("--out_prefix", "-o", default="metrics_out")
    return p.parse_args()

def main():
    args = parse_args()

    if not args.csv and not args.use_db:
        raise SystemExit("Provide --csv or --use_db.")

    if args.csv:
        df = load_from_csv(args.csv)
    else:
        df = load_from_db(args.sql, DB_SAMPLE)

    orchestrate(
        df,
        out_prefix=args.out_prefix,
        dedup=args.dedup,
        max_per_class=args.max_per_class,
    )

    LOG.info("Pipeline finished successfully")

if __name__ == "__main__":
    main()
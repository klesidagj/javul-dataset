#!/usr/bin/env python3
"""
viz_from_db.py
Fetch labeled rows (cwe_id present) with css_vector and ast/cfg/dfg graphs from Postgres,
parse features, preprocess, optionally deduplicate & sample, then produce PCA/t-SNE/UMAP
visualizations (3x3 grid: CSS / Graph / Combined x PCA / t-SNE / UMAP).

Edit the DB connection dict below before running.
"""
import os
import sys
import json
import re
import argparse
import time
from collections import Counter
from textwrap import shorten

import numpy as np
import pandas as pd
import psycopg2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------- CONFIGURE DB HERE --------------------
DB = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "klesi",
    "password": ""
}
# ----------------------------------------------------------

SEED = 42
np.random.seed(SEED)

# ------------------- SQL (edit LIMIT if you want) ---------------
DEFAULT_SQL = """
SELECT id, cwe_id, is_vulnerable,
       css_vector, ast_graph, cfg_graph, dfg_graph
FROM javul
WHERE cwe_id IS NOT NULL
  AND cwe_id <> 'NA'
  AND css_vector IS NOT NULL
  AND ast_graph IS NOT NULL
  AND cfg_graph IS NOT NULL
  AND dfg_graph IS NOT NULL
"""  # optionally append "LIMIT N" in CLI

# ----------------- robust parsing helpers -----------------
def parse_css_value(v):
    """Parse css_vector from various DB-returned forms into numpy array or None."""
    if v is None:
        return None
    # if already a list/tuple/ndarray (psycopg2 may return list for postgres arrays)
    if isinstance(v, (list, tuple, np.ndarray)):
        arr = np.array(v, dtype=float)
        return arr if arr.size > 0 else None
    # bytes
    if isinstance(v, (bytes, bytearray)):
        try:
            s = v.decode('utf-8', errors='ignore')
            return parse_css_value(s)
        except:
            return None
    s = str(v).strip()
    # Postgres array literal e.g. {0.1,0.2,...}
    if s.startswith('{') and s.endswith('}'):
        inner = s[1:-1]
        parts = inner.split(',')
        try:
            return np.array([float(p) for p in parts], dtype=float)
        except:
            # try strip quotes
            parts2 = [p.strip().strip('"').strip("'") for p in parts]
            try:
                return np.array([float(p) for p in parts2], dtype=float)
            except:
                return None
    # JSON-like arrays
    try:
        j = json.loads(s)
        if isinstance(j, list):
            return np.array(j, dtype=float)
        if isinstance(j, dict):
            for k in ('vector','embedding','emb','css','css_vector'):
                if k in j and isinstance(j[k], list):
                    return np.array(j[k], dtype=float)
    except:
        pass
    # attempt to unescape then parse
    try:
        s2 = s.encode('utf-8').decode('unicode_escape')
        j = json.loads(s2)
        if isinstance(j, list):
            return np.array(j, dtype=float)
    except:
        pass
    # fallback: extract floats with regex
    nums = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+', s)
    if nums:
        try:
            return np.array([float(x) for x in nums], dtype=float)
        except:
            return None
    return None

def parse_graph_flexible(g):
    """Parse graph value (jsonb/dict/string) into dict/list or None."""
    if g is None:
        return None
    if isinstance(g, dict):
        return g
    if isinstance(g, (bytes, bytearray)):
        try:
            s = g.decode('utf-8', errors='ignore')
            return parse_graph_flexible(s)
        except:
            return None
    s = str(g)
    # try direct JSON
    try:
        j = json.loads(s)
        if isinstance(j, (dict, list)):
            return j
    except:
        pass
    # try unescape then parse
    try:
        s2 = s.encode('utf-8').decode('unicode_escape')
        j = json.loads(s2)
        if isinstance(j, (dict, list)):
            return j
    except:
        pass
    # find first {...} block and try parse
    idx1 = s.find('{')
    idx2 = s.rfind('}')
    if idx1 != -1 and idx2 != -1 and idx2 > idx1:
        sub = s[idx1:idx2+1]
        try:
            j = json.loads(sub)
            if isinstance(j, (dict, list)):
                return j
        except:
            pass
    return None

def graph_summary_anyshape(parsed):
    """
    Return [n_nodes, n_edges, unique_node_types] from many possible graph shapes.
    If unable to detect, use heuristics.
    """
    if parsed is None:
        return [0,0,0]
    # list-of-nodes
    if isinstance(parsed, list):
        nodes = parsed
        types = set()
        for n in nodes:
            if isinstance(n, dict):
                for k in ('type','node_type','kind','label'):
                    if k in n and n[k] is not None:
                        types.add(str(n[k])); break
        return [len(nodes), 0, len(types)]
    # dict with node-like keys
    node_keys = ['nodes','vertices','body','children','statements','Nodes']
    for k in node_keys:
        if isinstance(parsed, dict) and k in parsed and isinstance(parsed[k], list):
            nodes = parsed[k]
            # detect edges if present
            edge_keys = ['edges','links','connections']
            edges = []
            for ek in edge_keys:
                if ek in parsed and isinstance(parsed[ek], list):
                    edges = parsed[ek]; break
            types = set()
            for n in nodes:
                if isinstance(n, dict):
                    for tkey in ('type','node_type','kind','label'):
                        if tkey in n and n[tkey] is not None:
                            types.add(str(n[tkey])); break
            return [len(nodes), len(edges), len(types)]
    # dict-of-nodes mapping (id->node)
    if isinstance(parsed, dict) and all(isinstance(v, dict) for v in parsed.values()):
        types = set()
        for v in parsed.values():
            for tkey in ('type','node_type','kind','label'):
                if tkey in v and v[tkey] is not None:
                    types.add(str(v[tkey])); break
        edges = parsed.get('edges') if isinstance(parsed.get('edges'), list) else []
        return [len(parsed), len(edges), len(types)]
    # fallback heuristic: count occurrences of "type" / "id" tokens
    s = json.dumps(parsed) if not isinstance(parsed, str) else parsed
    approx_nodes = s.count('"type"') + s.count('"node"') + s.count('"id"')
    return [approx_nodes, 0, 0]

# ---------------- DB fetch ----------------
def fetch_from_db(sql, db_config, limit=None):
    if limit:
        sql = sql.strip().rstrip(';') + f"\nLIMIT {limit};"
    conn = psycopg2.connect(**db_config)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df

# ---------------- feature building ----------------
def build_feature_matrices(df):
    # parse css
    css_parsed = []
    asts = []; cfgs = []; dfgs = []
    failed_css = 0
    failed_graphs = 0
    for i, r in df.iterrows():
        css = parse_css_value(r['css_vector'])
        ast = parse_graph_flexible(r['ast_graph'])
        cfg = parse_graph_flexible(r['cfg_graph'])
        dfg = parse_graph_flexible(r['dfg_graph'])
        if css is None:
            failed_css += 1
        if ast is None or cfg is None or dfg is None:
            failed_graphs += 1
        css_parsed.append(css)
        asts.append(ast); cfgs.append(cfg); dfgs.append(dfg)
    print(f"parse results: css failures={failed_css}, graph failures (any of ast/cfg/dfg)={failed_graphs}")
    # filter to rows where css parsed and all graphs parsed
    mask = [ (c is not None) and (a is not None) and (b is not None) and (d is not None)
            for c,a,b,d in zip(css_parsed, asts, cfgs, dfgs) ]
    n_keep = sum(mask)
    print(f"Rows with parsed css+graphs: {n_keep} / {len(df)}")
    if n_keep == 0:
        raise SystemExit("No usable rows after parsing. Inspect raw DB content.")
    df_kept = df.loc[mask].reset_index(drop=True)
    css_list = [c for c,m in zip(css_parsed, mask) if m]
    asts = [a for a,m in zip(asts, mask) if m]
    cfgs = [b for b,m in zip(cfgs, mask) if m]
    dfgs = [d for d,m in zip(dfgs, mask) if m]

    # css matrix: pad to max length
    max_len = max(len(c) for c in css_list)
    css_mat = np.vstack([np.pad(c, (0, max_len - len(c))) for c in css_list]).astype(float)

    # graph features: 3 summaries * 3 graphs = 9 features
    gf = []
    for a,b,d in zip(asts, cfgs, dfgs):
        gf.extend(graph_summary_anyshape(a))
        gf.extend(graph_summary_anyshape(b))
        gf.extend(graph_summary_anyshape(d))
    graph_X = np.array(gf, dtype=float).reshape(len(css_list), 9)

    combined = np.hstack([css_mat, graph_X])
    return df_kept, css_mat, graph_X, combined

# ---------------- preprocessing + DR ----------------
def preprocess(X, graph_cols_orig=None, jitter=1e-6, pca_target=30):
    var = X.var(axis=0)
    nz_mask = var > 0
    if not np.all(nz_mask):
        print(f"Removed {np.sum(~nz_mask)} zero-variance columns.")
    X = X[:, nz_mask]
    if graph_cols_orig is not None:
        orig_len = len(var)
        graph_mask_orig = np.zeros(orig_len, dtype=bool)
        graph_mask_orig[graph_cols_orig] = True
        graph_mask = graph_mask_orig[nz_mask]
        if graph_mask.any():
            X[:, graph_mask] = np.log1p(X[:, graph_mask])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rng = np.random.default_rng(SEED)
    Xs = Xs + rng.normal(scale=jitter, size=Xs.shape)
    n_comp = min(pca_target, max(2, Xs.shape[1]-1))
    if Xs.shape[1] > n_comp:
        pca = PCA(n_components=n_comp, random_state=SEED)
        Xr = pca.fit_transform(Xs)
        print(f"PCA pre-reduced {Xs.shape[1]} -> {Xr.shape[1]} dims (explained {pca.explained_variance_ratio_.sum():.3f})")
    else:
        Xr = Xs
    return Xr, Xs, nz_mask

def run_dr(X_scaled, X_pre, labels):
    # PCA projection 2D from scaled
    pca2 = PCA(n_components=2, random_state=SEED).fit_transform(X_scaled)
    # t-SNE (run on pre-reduced)
    tsne_in = X_pre if X_pre.shape[1] <= 50 else PCA(n_components=50, random_state=SEED).fit_transform(X_scaled)
    perp = 30 if len(labels) > 100 else max(5, len(labels)//5)
    tsne2 = TSNE(n_components=2, init='pca', perplexity=perp, n_iter=1000, random_state=SEED).fit_transform(tsne_in)
    # UMAP (safe options)
    um = umap.UMAP(n_neighbors=15, min_dist=0.1, init='random', metric='cosine', random_state=SEED)
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
    except Exception as e:
        metrics['error'] = str(e)
    return pca2, tsne2, um2, metrics

# ---------------- plotting ----------------
def scatter(ax, arr2d, labels, title):
    unique = sorted(list(set(labels)))
    cmap = plt.cm.get_cmap('tab20', max(2,len(unique)))
    color_map = {u:i for i,u in enumerate(unique)}
    colors = [color_map[l] for l in labels]
    ax.scatter(arr2d[:,0], arr2d[:,1], c=colors, s=18, alpha=0.85)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])

# ---------------- main ----------------
def main(args):
    sql = DEFAULT_SQL
    if args.limit:
        sql = sql.strip().rstrip(';') + f"\nLIMIT {args.limit};"
    print("Querying DB...")
    df = fetch_from_db(sql, DB, limit=None)
    print("Rows fetched:", len(df))

    # sanity
    if df.empty:
        raise SystemExit("No rows returned. Check DB and SQL filter.")

    # build features
    print("Parsing and building features...")
    df_keep, css_X, graph_X, combined_X = build_feature_matrices(df)
    print("Kept rows:", len(df_keep))
    print("CSS dim:", css_X.shape[1], "Graph features:", graph_X.shape[1], "Combined shape:", combined_X.shape)

    # show label counts
    print("\nLabel counts:")
    print(df_keep['cwe_id'].value_counts().head(50))

    # optional deduplication (rounded)
    if args.dedup:
        rounded = np.round(combined_X, 6)
        uniq, idx = np.unique(rounded, axis=0, return_index=True)
        idx_sorted = np.sort(idx)
        dup_count = combined_X.shape[0] - uniq.shape[0]
        print(f"Deduplicating visualization set: removed {dup_count} duplicate vectors (rounded to 6 decimals).")
        combined_X = combined_X[idx_sorted,:]
        css_X = css_X[idx_sorted,:]
        graph_X = graph_X[idx_sorted,:]
        df_keep = df_keep.iloc[idx_sorted].reset_index(drop=True)
    else:
        print("Skipping deduplication (may cause UMAP spectral init warnings).")

    # optional per-class sampling for readability
    if args.max_per_class and args.max_per_class > 0:
        sampled_idx = []
        rng = np.random.default_rng(SEED)
        max_per = int(args.max_per_class)
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
        print("After sampling per class, rows:", len(df_keep))

    # compute graph original column indices (css dims first)
    css_dim = css_X.shape[1]
    graph_cols_orig = list(range(css_dim, css_dim + graph_X.shape[1]))

    # Run DR for three feature sets: CSS-only, Graph-only, Combined
    outputs = {}
    for name, X in [('CSS', css_X), ('Graph', graph_X), ('Combined', combined_X)]:
        print(f"\nProcessing {name} features...")
        X_pre, X_scaled, nz_mask = preprocess(X, graph_cols_orig if name=='Combined' else (list(range(X.shape[1])) if name=='Graph' else None), jitter=1e-6, pca_target=30)
        p2, t2, u2, metrics = run_dr(X_scaled, X_pre, df_keep['cwe_id'].values)
        outputs[name] = (p2, t2, u2, metrics)
        print(f"{name} metrics:", metrics)

    # Plot 3x3 grid: rows CSS/Graph/Combined, cols PCA/t-SNE/UMAP
    fig, axes = plt.subplots(3,3, figsize=(15,12))
    row_names = ['CSS','Graph','Combined']
    col_names = ['PCA','t-SNE','UMAP']
    for i, rname in enumerate(row_names):
        p2, t2, u2, _ = outputs[rname]
        scatter(axes[i,0], p2, df_keep['cwe_id'].values, f"{rname} — PCA")
        scatter(axes[i,1], t2, df_keep['cwe_id'].values, f"{rname} — t-SNE")
        scatter(axes[i,2], u2, df_keep['cwe_id'].values, f"{rname} — UMAP")
    plt.tight_layout()
    ts = int(time.time())
    out_png = f"embeddings_db_{ts}.png"
    plt.savefig(out_png, dpi=150)
    print("Saved plot:", out_png)

    # Save embeddings CSV (combine all into one file)
    emb_records = []
    for name in row_names:
        p2, t2, u2, metrics = outputs[name]
        for i, idx in enumerate(range(len(df_keep))):
            emb_records.append({
                'id': df_keep.iloc[i]['id'],
                'cwe_id': df_keep.iloc[i]['cwe_id'],
                'feature_set': name,
                'pca_x': p2[i,0], 'pca_y': p2[i,1],
                'tsne_x': t2[i,0], 'tsne_y': t2[i,1],
                'umap_x': u2[i,0], 'umap_y': u2[i,1]
            })
    emb_df = pd.DataFrame.from_records(emb_records)
    out_csv = f"embeddings_db_{ts}.csv"
    emb_df.to_csv(out_csv, index=False)
    print("Saved embeddings csv:", out_csv)

    print("\nCompleted. Key notes:")
    print("- If UMAP prints 'Spectral initialisation failed', try increasing jitter or dedup. This script uses init='random' to reduce that risk.")
    print("- To visualize vulnerability (binary) instead of CWE labels, change color mapping to use 'is_vulnerable' in the scatter call.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='optional SQL LIMIT for quick tests')
    parser.add_argument('--dedup', type=lambda x: x.lower() in ('1','true','yes'), default=True, help='deduplicate rounded vectors for visualization')
    parser.add_argument('--max_per_class', type=int, default=300, help='max samples per class to plot (set 0 or None for no limit)')
    args = parser.parse_args()
    main(args)
# --- FILE: features.py ---
#!/usr/bin/env python3
"""
Build feature matrices from parsed raw DataFrame.
"""
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pandas as pd
from DatasetPlotting.data.parsers import parse_css_value, parse_graph_flexible, graph_summary_anyshape

@dataclass
class FeatureBundle:
    df: pd.DataFrame
    css: np.ndarray
    graph: np.ndarray
    combined: np.ndarray
    css_dim: int

def build_feature_matrices(df: pd.DataFrame) -> FeatureBundle:
    css_parsed = []
    asts = []; cfgs = []; dfgs = []
    for _, r in df.iterrows():
        css = parse_css_value(r.get('css_vector'))
        ast = parse_graph_flexible(r.get('ast_graph'))
        cfg = parse_graph_flexible(r.get('cfg_graph'))
        dfg = parse_graph_flexible(r.get('dfg_graph'))
        css_parsed.append(css)
        asts.append(ast); cfgs.append(cfg); dfgs.append(dfg)
    mask = [ (c is not None) and (a is not None) and (b is not None) and (d is not None)
            for c,a,b,d in zip(css_parsed, asts, cfgs, dfgs) ]
    keep = sum(mask)
    if keep == 0:
        raise SystemExit("No usable rows after parsing.")
    df_kept = df.loc[mask].reset_index(drop=True)
    css_list = [c for c,m in zip(css_parsed, mask) if m]
    asts = [a for a,m in zip(asts, mask) if m]
    cfgs = [b for b,m in zip(cfgs, mask) if m]
    dfgs = [d for d,m in zip(dfgs, mask) if m]
    max_len = max(len(c) for c in css_list)
    css_mat = np.vstack([np.pad(c, (0, max_len - len(c))) for c in css_list]).astype(float)
    gf = []
    for a,b,d in zip(asts, cfgs, dfgs):
        gf.extend(graph_summary_anyshape(a))
        gf.extend(graph_summary_anyshape(b))
        gf.extend(graph_summary_anyshape(d))
    graph_X = np.array(gf, dtype=float).reshape(len(css_list), 9)
    combined = np.hstack([css_mat, graph_X])
    return FeatureBundle(df=df_kept, css=css_mat, graph=graph_X, combined=combined, css_dim=css_mat.shape[1])

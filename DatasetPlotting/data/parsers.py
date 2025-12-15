# --- FILE: parsers.py ---
#!/usr/bin/env python3
"""
Flexible parsers for css vectors and graphs.
"""
import json
import re
from typing import Optional, Union, Tuple, List
import numpy as np

def parse_css_value(v) -> Optional[np.ndarray]:
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        arr = np.array(v, dtype=float)
        return arr if arr.size > 0 else None
    if isinstance(v, (bytes, bytearray)):
        try:
            s = v.decode('utf-8', errors='ignore')
            return parse_css_value(s)
        except Exception:
            return None
    s = str(v).strip()
    if s.startswith('{') and s.endswith('}'):
        inner = s[1:-1]
        parts = inner.split(',')
        try:
            return np.array([float(p) for p in parts], dtype=float)
        except Exception:
            parts2 = [p.strip().strip('"').strip("'") for p in parts]
            try:
                return np.array([float(p) for p in parts2], dtype=float)
            except Exception:
                return None
    try:
        j = json.loads(s)
        if isinstance(j, list):
            return np.array(j, dtype=float)
        if isinstance(j, dict):
            for k in ('vector','embedding','emb','css','css_vector'):
                if k in j and isinstance(j[k], list):
                    return np.array(j[k], dtype=float)
    except Exception:
        pass
    try:
        s2 = s.encode('utf-8').decode('unicode_escape')
        j = json.loads(s2)
        if isinstance(j, list):
            return np.array(j, dtype=float)
    except Exception:
        pass
    nums = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+', s)
    if nums:
        try:
            return np.array([float(x) for x in nums], dtype=float)
        except Exception:
            return None
    return None

def parse_graph_flexible(g) -> Optional[Union[dict, list]]:
    if g is None:
        return None
    if isinstance(g, (dict, list)):
        return g
    if isinstance(g, (bytes, bytearray)):
        try:
            s = g.decode('utf-8', errors='ignore')
            return parse_graph_flexible(s)
        except Exception:
            return None
    s = str(g)
    try:
        j = json.loads(s)
        if isinstance(j, (dict, list)):
            return j
    except Exception:
        pass
    try:
        s2 = s.encode('utf-8').decode('unicode_escape')
        j = json.loads(s2)
        if isinstance(j, (dict, list)):
            return j
    except Exception:
        pass
    idx1 = s.find('{')
    idx2 = s.rfind('}')
    if idx1 != -1 and idx2 != -1 and idx2 > idx1:
        sub = s[idx1:idx2+1]
        try:
            j = json.loads(sub)
            if isinstance(j, (dict, list)):
                return j
        except Exception:
            pass
    return None

def graph_summary_anyshape(parsed) -> Tuple[int, int, int]:
    if parsed is None:
        return (0, 0, 0)
    if isinstance(parsed, list):
        types = set()
        for n in parsed:
            if isinstance(n, dict):
                for k in ('type','node_type','kind','label'):
                    if k in n and n[k] is not None:
                        types.add(str(n[k])); break
        return (len(parsed), 0, len(types))
    node_keys = ['nodes','vertices','body','children','statements','Nodes']
    if isinstance(parsed, dict):
        for k in node_keys:
            if k in parsed and isinstance(parsed[k], list):
                nodes = parsed[k]
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
                return (len(nodes), len(edges), len(types))
        if all(isinstance(v, dict) for v in parsed.values()):
            types = set()
            for v in parsed.values():
                for tkey in ('type','node_type','kind','label'):
                    if tkey in v and v[tkey] is not None:
                        types.add(str(v[tkey])); break
            edges = parsed.get('edges') if isinstance(parsed.get('edges'), list) else []
            return (len(parsed), len(edges), len(types))
    s = json.dumps(parsed) if not isinstance(parsed, str) else parsed
    approx_nodes = s.count('"type"') + s.count('"node"') + s.count('"id"')
    return (approx_nodes, 0, 0)
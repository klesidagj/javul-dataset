INSERT INTO javul_cl(id, raw_code, ast_graph, cfg_graph, dfg_graph, css_vector, cwe_id, is_vulnerable, "source")
VALUES %s
ON CONFLICT (id) DO NOTHING;



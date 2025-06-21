import uuid
import pandas as pd
from pathlib import Path

def extract_owasp_code_snippets(csv_path, java_base):

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df.rename(columns={
        '# test name': 'testcaseid',
        'real vulnerability': 'expectedresult',
        'cwe': 'cwe_id'
    }, inplace=True)

    records = []
    java_base = Path(java_base)

    for _, row in df.iterrows():
        test_id = row["testcaseid"]
        is_vulnerable = str(row["expectedresult"]).lower() == "true"
        cwe_id = str(int(row["cwe_id"])) if is_vulnerable and not pd.isna(row["cwe_id"]) else None

        java_file = java_base / f"{test_id}.java"
        if not java_file.exists():
            continue

        try:
            with open(java_file, "r", encoding="utf-8") as f:
                raw_code = f.read()
        except Exception as e:
            print(f"Error reading {java_file}: {e}")
            continue

        records.append({
            "id": str(uuid.uuid4()),
            "raw_code": raw_code,
            "ast_graph": None,
            "cfg_graph": None,
            "dfg_graph": None,
            "css_vector": None,
            "cwe_id": cwe_id,
            "is_vulnerable": is_vulnerable,
            "source": test_id
        })

    return pd.DataFrame(records)

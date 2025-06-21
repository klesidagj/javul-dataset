import os
import re
import uuid
import json
import pandas as pd

from db import insert_snippets_to_postgres


def extract_methods(java_text):
    """
    Extracts methods named 'bad' or starting with 'good' from the given Java source.
    Handles multi-line signatures and tracks braces properly.
    Returns a list of (method_name, full_method_text) tuples.
    """
    lines = java_text.splitlines(keepends=True)
    results = []
    in_method = False
    method_lines = []
    method_name = None
    brace_depth = 0

    for i, line in enumerate(lines):
        # Start of method signature (look for bad or good*)
        m = re.search(r'\bvoid\s+(bad|good[A-Za-z0-9_]*)\s*\(', line)
        if not in_method and m:
            method_name = m.group(1)
            in_method = True
            method_lines = [line]
            brace_depth = line.count('{') - line.count('}')
            continue

        if in_method:
            method_lines.append(line)
            brace_depth += line.count('{') - line.count('}')
            if brace_depth <= 0 and '{' in ''.join(method_lines):
                # Method body is complete
                results.append((method_name, ''.join(method_lines)))
                in_method = False
                method_lines = []
                method_name = None

    return results


def parse_juliet(root_dir, source_label):
    """
    Walks the Juliet Java test suite directory, extracts relevant fields,
    and returns a pandas DataFrame with columns:
    id, raw_code, cwe_id, is_vulnerable, source
    """
    filename_regex = re.compile(r'^(CWE\d+)_.*__.*\.java$')
    records = []

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.endswith('.java'):
                continue
            m = filename_regex.match(fname)
            if not m:
                continue
            cwe_id = m.group(1)  # Already formatted as 'CWE###'
            file_path = os.path.join(dirpath, fname)
            print(f"✅ Matched file: {fname}, CWE: {cwe_id}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            for method_name, method_text in extract_methods(content):
                records.append({
                    "id": str(uuid.uuid4()),
                    "raw_code": method_text,
                    "cwe_id": cwe_id,
                    "is_vulnerable": (method_name == "bad"),
                    "source": source_label
                })

    # Create DataFrame
    df = pd.DataFrame(records)
    print(f"✅ Extracted {len(df)} snippets from Juliet.")
    print(df.head())  # optional: see first few rows

    return df

def main():
    # Configuration
    ROOT_DIR = "/Users/klesi/Downloads/Java/src/testcases"

    SOURCE_LABEL = "Juliet Java Test Suite"

    # Parse and build DataFrame
    df = parse_juliet(ROOT_DIR, SOURCE_LABEL)
    insert_snippets_to_postgres(df)



if __name__ == "__main__":
    main()

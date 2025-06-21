import uuid

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import sqlite3

def insert_snippets_to_postgres(df):
    
# PostgreSQL connection parameters
    conn = psycopg2.connect(
        dbname="postgres",
        user="klesi",
        password="",
        host="localhost",
        port="5432"
    )

    with conn.cursor() as cur:
        insert_query = """
              INSERT INTO public.code_snippets (
                  id, raw_code, ast_graph, cfg_graph, dfg_graph, css_vector,
                  cwe_id, is_vulnerable, source
              ) VALUES %s
              ON CONFLICT (id) DO NOTHING
          """
        values = [
            (
                row["id"],
                row["raw_code"],
                None,
                None,
                None,
                None,
                row["cwe_id"] if pd.notna(row["cwe_id"]) else None,
                row["is_vulnerable"],
                row["source"]
            )
            for _, row in df.iterrows()
        ]
        execute_values(cur, insert_query, values)

    conn.commit()
    conn.close()
    print("✅ Snippets inserted into PostgreSQL.")


def extract_java_snippets_from_sqlite(sqlite_path):
    conn = sqlite3.connect(sqlite_path)
    query = """
    SELECT 
        m.code           AS raw_code,
        m.before_change  AS is_vulnerable,
        fi.cve_id        AS source,
        cc.cwe_id
    FROM method_change m
    JOIN file_change f      ON m.file_change_id   = f.file_change_id
    JOIN fixes fi           ON f.hash             = fi.hash
    JOIN cwe_classification cc ON fi.cve_id        = cc.cve_id
    WHERE f.programming_language = 'Java'
      AND m.code IS NOT NULL
      AND cc.cwe_id IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Add the fields your table requires
    df["id"]          = [str(uuid.uuid4()) for _ in range(len(df))]
    df["ast_graph"]   = None
    df["cfg_graph"]   = None
    df["dfg_graph"]   = None
    df["css_vector"]  = None

    return df[[
        "id", "raw_code", "ast_graph", "cfg_graph", "dfg_graph",
        "css_vector", "cwe_id", "is_vulnerable", "source"
    ]]

def insert_into_postgres(df, pg_credentials):
    conn = psycopg2.connect(**pg_credentials)
    with conn.cursor() as cur:
        insert_query = """
            INSERT INTO public.code_snippets (
                id, raw_code, ast_graph, cfg_graph, dfg_graph, css_vector,
                cwe_id, is_vulnerable, source
            ) VALUES %s
            ON CONFLICT (id) DO NOTHING
        """
        values = [tuple(row) for row in df.to_numpy()]
        execute_values(cur, insert_query, values)
    conn.commit()
    conn.close()
    print(f"✅ Inserted {len(df)} rows into PostgreSQL.")



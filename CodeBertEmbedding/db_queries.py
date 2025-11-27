# db_queries.py

import psycopg2
from psycopg2.extras import DictCursor

def fetch_code_snippets(conn_string, table, id_col, code_col, limit=1000, offset=0):
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor(cursor_factory=DictCursor)

    query = f"""
        SELECT {id_col}, {code_col}
        FROM {table}
        WHERE {code_col} IS NOT NULL
        ORDER BY {id_col}
        LIMIT %s OFFSET %s;
    """

    cur.execute(query, (limit, offset))
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows

# db/save_vectors.py

import psycopg2
from psycopg2.extras import Json

def save_vectors(conn_string, table, id_col, vec_col, vector_map):
    """
    vector_map = { id: [vector] }
    """
    conn = psycopg2.connect(conn_string)
    cur = conn.cursor()

    query = f"""
        UPDATE {table}
        SET {vec_col} = %s
        WHERE {id_col} = %s;
    """

    for _id, vec in vector_map.items():
        cur.execute(query, (vec, _id))

    conn.commit()
    cur.close()
    conn.close()
# DatasetCreation/main.py
import argparse
from config.log import get_logger
from extractors.juliet_reader import load_juliet
from extractors.owasp_reader import extract_owasp_dataset
from extractors.cvefixes_reader import read_java_snippets
from loaders.postgresql_loader import insert_rows_batch, load_sql_template
from DatasetCreation.utils.record_utils import make_record
from config.config import PSQL_LOADER_SQL, PG_CONN_STRING, OWASP_CSV_PATH, OWASP_JAVA_DIR

logger = get_logger("main")


def build_and_save_all(extract_juliet: bool, extract_owasp: bool, extract_cve: bool, sql_path: str, conn_string: str):
    frames = []
    if extract_juliet:
        logger.info("Extracting Juliet...")
        df_j = load_juliet()
        frames.append(df_j)

    if extract_owasp:
        logger.info("Extracting OWASP...")
        df_o = extract_owasp_dataset()
        frames.append(df_o)

    if extract_cve:
        logger.info("Extracting CVEFixes...")
        df_c = read_java_snippets()
        frames.append(df_c)

    if not frames:
        logger.warning("No datasets selected or no data extracted.")
        return

    import pandas as pd
    df_all = pd.concat(frames, ignore_index=True, sort=False)

    # final normalization
    from DatasetCreation.utils.cwe_utils import enforce_cwe_rules
    df_all = enforce_cwe_rules(df_all)

    # Convert DataFrame to list of dicts and write to DB
    records = df_all.to_dict(orient="records")
    insert_rows_batch(records, conn_string=conn_string, sql_path=sql_path)
    logger.info("All datasets saved.")


def _parse_args():
    p = argparse.ArgumentParser(description="DatasetCreation ETL")
    p.add_argument("--juliet", action="store_true", help="include Juliet extraction")
    p.add_argument("--owasp", action="store_true", help="include OWASP extraction")
    p.add_argument("--cve", action="store_true", help="include CVEFixes extraction")
    # p.add_argument("--sql", default=None, help="DatasetCreation/sql/psql_loader.sql")
    # p.add_argument("--conn", default=None, help="Postgres connection string")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sql_path = PSQL_LOADER_SQL
    conn = PG_CONN_STRING
    build_and_save_all(args.juliet, args.owasp, args.cve, sql_path, conn)
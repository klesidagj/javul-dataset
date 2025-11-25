# DatasetCreation/readers/owasp_reader.py
from pathlib import Path
import pandas as pd

from DatasetCreation.config.log import get_logger
from DatasetCreation.utils.cwe_utils import normalize_cwe_token
from DatasetCreation.utils.record_utils import make_record
from DatasetCreation.config.config import OWASP_CSV_PATH, OWASP_JAVA_DIR, OWASP_SOURCE_NAME
from DatasetCreation.utils.juliet_utils import clean_java_source

logger = get_logger("owasp_reader")


def extract_owasp_dataset(csv_path: str = OWASP_CSV_PATH, java_dir: str = OWASP_JAVA_DIR) -> pd.DataFrame:
    """
    Read the OWASP CSV metadata and return a DataFrame with the full Java class
    saved in raw_code. Keeps cwe normalization and vulnerability flag from CSV.
    """
    logger.info("Loading OWASP CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df.rename(columns={'# test name': 'testcaseid', 'real vulnerability': 'expectedresult', 'cwe': 'cwe_id'}, inplace=True)

    java_base = Path(java_dir)
    records = []

    for _, row in df.iterrows():
        test_id = str(row.get("testcaseid", "")).strip()
        is_vulnerable = str(row.get("expectedresult", "")).strip().lower() == "true"
        cwe_id = normalize_cwe_token(row.get("cwe_id")) if is_vulnerable else None

        java_path = java_base / f"{test_id}.java"
        if not java_path.exists():
            logger.debug("Missing java file for %s", test_id)
            continue

        try:
            raw_code = java_path.read_text(encoding="utf-8", errors="ignore")
            raw_code = clean_java_source(raw_code)
        except Exception as e:
            logger.warning("Error reading %s: %s", java_path, e)
            continue

        rec = make_record(raw_code=raw_code, cwe_id=cwe_id, is_vulnerable=is_vulnerable, source=OWASP_SOURCE_NAME)
        records.append(rec)

    df_out = pd.DataFrame.from_records(records)
    logger.info("OWASP reader produced %d records", len(df_out))
    return df_out
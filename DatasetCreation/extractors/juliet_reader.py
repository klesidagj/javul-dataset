# DatasetCreation/readers/juliet_reader.py
import os
from pathlib import Path
import pandas as pd

from DatasetCreation.config.log import get_logger
from DatasetCreation.utils.juliet_utils import (
    clean_java_source,
    extract_bad_method,
    extract_class_name,
    build_synthetic_class,
)
from DatasetCreation.utils.record_utils import make_record
from DatasetCreation.utils.cwe_utils import extract_cwe_from_classname
from DatasetCreation.config.config import JULIET_PATH, JULIET_SOURCE_NAME
logger = get_logger("juliet_reader")


def load_juliet() -> pd.DataFrame:
    """
    Scan the Juliet root directory and produce a dataframe of vulnerable-only records.
    Each record contains a minimal class wrapper that preserves the bad() method exactly.
    """
    records = []
    root = Path(JULIET_PATH)
    if not root.exists():
        logger.warning("Juliet root does not exist: %s", JULIET_PATH)
        return pd.DataFrame(columns=list(make_record("", None, False, "Juliet").keys()))

    for dirpath, _, filenames in os.walk(JULIET_PATH):
        for fname in filenames:
            if not fname.endswith(".java"):
                continue
            file_path = os.path.join(dirpath, fname)
            try:
                content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logger.warning("Failed to read %s: %s", file_path, e)
                continue

            stripped = clean_java_source(content)
            class_name = extract_class_name(stripped)
            if not class_name:
                logger.debug("No class name found in %s", file_path)
                continue

            cwe_id = extract_cwe_from_classname(class_name)
            if not cwe_id:
                logger.warning(
                    "Juliet file %s contains bad() but no valid CWE in class name, skipping.",
                    file_path
                )
                continue

            bad_src = extract_bad_method(stripped)
            if not bad_src:
                logger.debug("No bad() method in %s", file_path)
                continue

            synthetic = build_synthetic_class(class_name, bad_src)
            cwe = extract_cwe_from_classname(class_name)
            rec = make_record(raw_code=synthetic, cwe_id=cwe, is_vulnerable=True, source=JULIET_SOURCE_NAME)
            records.append(rec)

    df = pd.DataFrame.from_records(records)
    logger.info("Juliet reader produced %d records", len(df))
    return df
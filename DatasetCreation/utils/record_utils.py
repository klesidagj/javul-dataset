# DatasetCreation/utils/record_utils.py
import uuid
from typing import Optional, Dict, Any


def make_record(raw_code: str, cwe_id: Optional[str], is_vulnerable: bool, source: str) -> Dict[str, Any]:
    """
    Standardized record for loader. Always returns these keys:
      id, raw_code, ast_graph, cfg_graph, dfg_graph, css_vector, cwe_id, is_vulnerable, source
    Graph/vector fields are filled with None to match DB schema.
    """
    return {
        "id": str(uuid.uuid4()),
        "raw_code": raw_code,
        "ast_graph": None,
        "cfg_graph": None,
        "dfg_graph": None,
        "css_vector": None,
        "cwe_id": cwe_id,
        "is_vulnerable": bool(is_vulnerable),
        "source": source
    }
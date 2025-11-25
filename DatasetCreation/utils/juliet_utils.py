# DatasetCreation/utils/juliet_utils.py
import re
from typing import Optional

def clean_java_source(java_text: str) -> str:
    """
    Removes:
      - package lines
      - import lines
      - all comments (//, /* */)
    Preserves:
      - full class structure
      - all indentation
      - all code inside the class
    """
    # 1) Remove package + import lines
    lines = java_text.splitlines(True)
    no_pkg = [
        line for line in lines
        if not line.strip().startswith(("package ", "import "))
    ]
    text = "".join(no_pkg)

    # 2) Remove block comments (/* */)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # 3) Remove single-line comments (//)
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)

    return text


def extract_class_name(java_text: str) -> Optional[str]:
    """Return the first class name found, or None."""
    m = re.search(r"\bclass\s+([A-Za-z0-9_]+)", java_text)
    return m.group(1) if m else None


def extract_bad_method(java_text: str) -> Optional[str]:
    """
    Find public void bad(...) { ... } using brace-matching.
    Preserves exact indentation and internal comments.
    """
    m = re.search(r"\bpublic\s+void\s+bad\s*\(", java_text)
    if not m:
        return None

    start = m.start()
    open_brace = java_text.find("{", start)
    if open_brace == -1:
        return None

    depth = 0
    i = open_brace
    n = len(java_text)
    while i < n:
        ch = java_text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return java_text[start:i + 1]
        i += 1
    return None


def build_synthetic_class(class_name: str, bad_method: str) -> str:
    """Return a minimal class wrapper containing only bad_method. Preserve method text."""
    return f"public class {class_name}\n{{\n{bad_method}\n}}\n"
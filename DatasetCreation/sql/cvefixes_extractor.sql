SELECT
    m.code           AS raw_code,
    m.before_change  AS is_vulnerable,
    fi.cve_id        AS source,
    cc.cwe_id
FROM method_change m
JOIN file_change f           ON m.file_change_id = f.file_change_id
JOIN fixes fi                ON f.hash           = fi.hash
JOIN cwe_classification cc   ON fi.cve_id        = cc.cve_id
WHERE f.programming_language = 'Java'
  AND m.code IS NOT NULL
  AND cc.cwe_id IS NOT NULL;

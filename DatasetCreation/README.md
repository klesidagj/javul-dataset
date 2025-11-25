# Javul ETL Pipeline  
*A unified extraction, transformation, and loading pipeline for multiple Java vulnerability datasets.*

## 📌 Overview
Javul is a unified ETL pipeline designed to extract Java code examples from multiple vulnerability datasets, normalize their structure, and store them in a consistent PostgreSQL schema for downstream program analysis, machine learning, or vulnerability research.

The pipeline currently supports:

- **Juliet Test Suite (Java edition)**  
- **OWASP Benchmark**  
- **CVEfixes dataset (source-level CVE vulnerability & fix pairs)**  

Each dataset is processed into a standardized format:

```json
{
  "id": "...",
  "raw_code": "... full Java class or snippet ...",
  "cwe_id": "CWE-XXX",
  "is_vulnerable": "true/false",
  "source": "Juliet | Owasp | CVEFixes",
  "ast_graph": null,
  "cfg_graph": null,
  "dfg_graph": null,
  "css_vector": null
}
```

1. CVEFixes dataset
The pipeline integrates and processes CVEfixes, a dataset containing real-world vulnerability–fix pairs mined from open-source projects.
Citation:
Bhandari, G., Naseer, A., & Moonen, L. (2021).
“CVEfixes: Automated Collection of Vulnerabilities and Their Fixes from Open-Source Software.”
In: Proceedings of the 17th International Conference on Predictive Models and Data Analytics in Software Engineering (PROMISE '21).
ACM. DOI: 10.1145/3475960.3475985.
BibTeX:

@inproceedings{bhandari2021:cvefixes,
    title = {{CVEfixes: Automated Collection of Vulnerabilities and Their Fixes from Open-Source Software}},
    booktitle = {{Proceedings of the 17th International Conference on Predictive Models and Data Analytics in Software Engineering (PROMISE '21)}},
    author = {Bhandari, Guru and Naseer, Amara and Moonen, Leon},
    year = {2021},
    pages = {10},
    publisher = {{ACM}},
    doi = {10.1145/3475960.3475985},
    isbn = {978-1-4503-8680-7},
    language = {en}
}
Dataset repository:
https://github.com/secureIT-project/CVEfixes
2. OWASP Benchmark for Java
The project uses Java source files from the OWASP Benchmark to extract method-level vulnerable and non-vulnerable examples.
Dataset repository:
https://github.com/OWASP-Benchmark/BenchmarkJava
3. Juliet Test Suite (Java)
The Juliet Test Suite offers thousands of synthetic vulnerability test cases.
This project extracts only the bad() variants to produce vulnerable minimal Java classes.
Dataset repository (Java version):
https://github.com/find-sec-bugs/juliet-test-suite/tree/master/src

I
nstall dependencies
pip install -r requirements.txt


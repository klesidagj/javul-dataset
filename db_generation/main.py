# This is a sample Python script.
from db import insert_snippets_to_postgres, extract_java_snippets_from_sqlite, insert_into_postgres
from owasp_bench import extract_owasp_code_snippets


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# TODO make strings dynamic
csv_path = "/Users/klesi/IdeaProjects/BenchmarkJava/expectedresults-1.2.csv",
java_base = "/Users/klesi/IdeaProjects/BenchmarkJava/src/main/java/org/owasp/benchmark/testcode",
output_path = "owasp_code_snippets.csv"


# if __name__ == "__main__":
#     df = extract_owasp_code_snippets(
#         "/Users/klesi/IdeaProjects/BenchmarkJava/expectedresults-1.2.csv",
#         "/Users/klesi/IdeaProjects/BenchmarkJava/src/main/java/org/owasp/benchmark/testcode"
#     )
#     insert_snippets_to_postgres(df)

if __name__ == "__main__":
    # SQLite file
    sqlite_file = "CVEfixes.db"  # Make sure it's in the same folder

    # PostgreSQL connection details
    pg_credentials = {
        "dbname": "postgres",
        "user": "klesi",
        "password": "",
        "host": "localhost",
        "port": "5432"
    }

    df = extract_java_snippets_from_sqlite(sqlite_file)
    insert_into_postgres(df, pg_credentials)
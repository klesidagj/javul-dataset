from extractors.sqlite_reader import read_java_snippets
from extractors.postgres_loader import PostgresLoader

df = read_java_snippets(
    sqlite_path="config/sources/cvefixes.db",
    sql_path="sql/extract_cvefixes_java.sql"
)

loader = PostgresLoader("postgresql://user:password@localhost:5432/central")
loader.load_dataframe(df, table="raw_snippets")

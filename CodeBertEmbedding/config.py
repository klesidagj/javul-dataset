#config.py

POSTGRES_CONN = "dbname=postgres user=klesi host=localhost port=5432"
TABLE_NAME = "javul_cl"
CODE_COL = "raw_code"
VEC_COL = "css_vector"
ID_COL = "id"
BATCH_SIZE = 64
MAX_LENGTH = 256
PCA_COMPONENTS = 256
PCA_PATH = "pca_model.pkl"
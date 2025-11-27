import numpy as np
import torch

from config  import (
    POSTGRES_CONN, TABLE_NAME, ID_COL, CODE_COL, VEC_COL,
    BATCH_SIZE, MAX_LENGTH, PCA_COMPONENTS, PCA_PATH
)

from db_queries import (fetch_code_snippets, save_vectors)
from codebert_model import load_codebert, extract_vectors
from pca_manager import fit_pca, load_pca, apply_pca


def main():
    tokenizer, model, device = load_codebert()

    offset = 0
    pca = None
    all_raw_cls = []

    print("\n### STEP 1 — Extract ALL CLS vectors for PCA training ###")
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
    print("MPS:", torch.backends.mps.is_available())
    # First pass: extract raw 768-d vectors for PCA training
    while True:
        rows = fetch_code_snippets(POSTGRES_CONN, TABLE_NAME, ID_COL, CODE_COL,
                                   limit=BATCH_SIZE, offset=offset)

        if not rows:
            break

        snippets = [r[CODE_COL] for r in rows]
        cls_vecs = extract_vectors(snippets, tokenizer, model, device,
                                       batch_size=BATCH_SIZE, max_length=MAX_LENGTH)

        all_raw_cls.append(cls_vecs)
        offset += BATCH_SIZE

    all_raw_cls = np.vstack(all_raw_cls)

    print("\n### STEP 2 — Fit PCA (768 → 256 dims) ###")
    reduced_vectors, pca = fit_pca(all_raw_cls, PCA_COMPONENTS, PCA_PATH)

    print(f"PCA trained. Shape after reduction: {reduced_vectors.shape}")

    print("\n### STEP 3 — Second pass: Embed + Reduce + Save ###")
    offset = 0

    while True:
        rows = fetch_code_snippets(POSTGRES_CONN, TABLE_NAME, ID_COL, CODE_COL,
                                   limit=BATCH_SIZE, offset=offset)

        if not rows:
            print("Done! All vectors saved to DB.")
            break

        ids = [r[ID_COL] for r in rows]
        snippets = [r[CODE_COL] for r in rows]

        cls_vecs = extract_vectors(snippets, tokenizer, model, device)
        reduced = apply_pca(pca, cls_vecs)

        # Convert to Python lists for Postgres
        vec_map = {id_: vec.tolist() for id_, vec in zip(ids, reduced)}
        save_vectors(POSTGRES_CONN, TABLE_NAME, ID_COL, VEC_COL, vec_map)

        offset += BATCH_SIZE
        print(f"Processed batch offset={offset}")


if __name__ == "__main__":
    main()
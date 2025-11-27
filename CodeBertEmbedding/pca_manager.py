import pickle
from sklearn.decomposition import PCA
import numpy as np

def fit_pca(vectors, dims, save_path):
    pca = PCA(n_components=dims)
    reduced = pca.fit_transform(vectors)

    with open(save_path, "wb") as f:
        pickle.dump(pca, f)

    return reduced, pca


def load_pca(save_path):
    with open(save_path, "rb") as f:
        pca = pickle.load(f)
    return pca


def apply_pca(pca, vectors):
    return pca.transform(vectors)
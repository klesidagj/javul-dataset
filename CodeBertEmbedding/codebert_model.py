#codebert_model.py

import torch
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_codebert():
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    model.to(DEVICE)
    model.eval()
    return tokenizer, model, DEVICE


def extract_vectors(snippets, tokenizer, model, device, batch_size=16, max_length=256):
    vectors = []

    for i in tqdm(range(0, len(snippets), batch_size), desc="Embedding"):
        batch = snippets[i:i + batch_size]

        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = model(**tokens)

        # CLS embedding = index 0
        cls_vecs = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # (B, 768)

        vectors.append(cls_vecs)

    return np.vstack(vectors)

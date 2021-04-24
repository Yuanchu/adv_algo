"""
@author: Yuanchu Dang
"""

from sentence_transformers import SentenceTransformer


def get_pretrained_embeddings(model_name, sentences):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    return embeddings
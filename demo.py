"""
@author: Yuanchu Dang
"""

from embeddings.utils import get_pretrained_embeddings

if __name__ == "main":
    model_name = 'paraphrase-distilroberta-base-v1'
    sentences = ['This ', 'The quick brown fox jumps over the lazy dog.']
    embeddings = get_pretrained_embeddings(model_name, sentences)
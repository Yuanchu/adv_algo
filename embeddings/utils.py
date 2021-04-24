"""
@author: Yuanchu Dang
"""

from sentence_transformers import SentenceTransformer


def get_pretrained_embeddings(model_name, sentences):
    """
    :param model_name: string representing the pretrained model
    :param sentences: a list of strings (sentences)
    :return: a list of sentence embeddings (equal length)
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    return embeddings
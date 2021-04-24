"""
@author: Yuanchu Dang
"""

from embeddings.utils import get_pretrained_embeddings, train_word_embeddings

if __name__ == "main":
    model_name = 'paraphrase-distilroberta-base-v1'
    sentences = ['This ', 'This', 'The quick brown fox jumps over the lazy dog.']
    embeddings1 = get_pretrained_embeddings(model_name, sentences)

    sentence = 'hello hello hello world, what is going on'
    embedding_size = 10
    vocab, embeddings2 = train_word_embeddings(sentence, embedding_size)
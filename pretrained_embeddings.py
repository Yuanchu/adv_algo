from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-distilroberta-base-v1')

sentences = ['This ',
    'The quick brown fox jumps over the lazy dog.']
sentence_embeddings = model.encode(sentences)

for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")

sentence_embeddings.shape
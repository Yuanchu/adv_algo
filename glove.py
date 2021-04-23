from nltk.tokenize import word_tokenize
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot

body_length = 3
embedding_size = 2
len_of_batches = 20
num_terms = 5
ratio = 0.001

text_file = open('story.txt', 'r')
read_text = text_file.read().lower()
text_file.close()

# Create vocabulary and word lists
word_list = word_tokenize(read_text)
vocabulary = np.unique(word_list)

# Create word to index mapping
word_to_index = {word: index for index, word in enumerate(vocabulary)}

# Create vocabulary and word lists
word_list_size = len(word_list)
vocabulary_size = len(vocabulary)
maxlength = len(read_text)


# Construct co-occurence matrix
comation = np.zeros((len(vocabulary), len(vocabulary)))
for i in range(word_list_size):
	for j in range(1, body_length+1):
		index = word_to_index[word_list[i]]
		if i-j > 0:
			left_index = word_to_index[word_list[i-j]]
			comation[index, left_index] += 1.0/j
		if i+j < word_list_size:
			right_index = word_to_index[word_list[i+j]]
			comation[index, right_index] += 1.0/j

co_occurrences = np.transpose(np.nonzero(comation))  # Guarantee Non-zero co-occurrences

# GET Weight function
def f_x(Y):
	Xmax = 2
	alpha_value = 0.75
	if Y < Xmax:
		return (Y/Xmax)**alpha_value
	return 1

# Set up word vectors and biases
left_embedded, right_embedded = [
	[Variable(torch.from_numpy(np.random.normal(0, 0.01, (embedding_size, 1))),
		requires_grad=True) for j in range(vocabulary_size)] for i in range(2)]
left_biase, right_biase = [
	[Variable(torch.from_numpy(np.random.normal(0, 0.01, 1)), 
		requires_grad=True) for j in range(vocabulary_size)] for i in range(2)]

improvement = optim.Adam(left_biase + right_biase + left_embedded + right_embedded, lr=ratio)  # Set an improvement or (an optimizer)

# Batch sampling function
def show_batches():
	left_vectors = []
	right_vectors = []
	covals = []
	left_vector_bias = []
	right_vector_bias = []
	samples = np.random.choice(np.arange(len(co_occurrences)), size=len_of_batches, replace=False)
	for example in samples:
		index = tuple(co_occurrences[example])
		left_vectors.append(left_embedded[index[0]])
		right_vectors.append(right_embedded[index[1]])
		covals.append(comation[index])
		left_vector_bias.append(left_biase[index[0]])
		right_vector_bias.append(right_biase[index[1]])
	return left_vectors, right_vectors, covals, left_vector_bias, right_vector_bias

# Train model
for term in range(num_terms):
	cnt_batch = int(word_list_size/len_of_batches)
	agv_lossless = 1
	for batch in range(cnt_batch):
		improvement.zero_grad()
		left_vectors, right_vectors, covals, left_vector_bias, right_vector_bias = show_batches()
		loss = sum([torch.mul((torch.dot(left_vectors[i].view(-1), right_vectors[i].view(-1)) +
				left_vector_bias[i] + right_vector_bias[i] - np.log(covals[i]))**2,
				f_x(covals[i])) for i in range(len_of_batches)])
		agv_lossless += loss.data[0]/cnt_batch
		loss.backward()
		improvement.step()
	print("lossless of term on average "+str(term+1)+": ", agv_lossless)

# embedding to graphs
# if embedding_size == 2:
	# Pick some random words
	word_indexs = np.random.choice(np.arange(len(vocabulary)), size=num_terms, replace=False)
	for word_index in word_indexs:
		# Create embedding by summing left and right embeddings
		word_embd = (left_embedded[word_index].data + right_embedded[word_index].data).numpy()
		x, y = word_embd[0][0], word_embd[1][0]
		pyplot.scatter(x, y)
		pyplot.annotate(vocabulary[word_index], xy=(x, y), xytext=(5, 2),
			textcoords='offset points', ha='right', va='bottom')
	pyplot.savefig("glove.png")



# if __name__ == '__main__':
# 	parser = argparse.ArgumentParser(description="type in")
# 	parser.add_argument('len_of_batches', type=int, help='len_of_batches')
# 	# parser.add_argument('num_terms', type=int, help='num_terms')
#
# 	main(parser.parse_args())
import numpy as np
import timeit
import falconn
import pandas as pd

from __future__ import print_function
from embeddings.utils import get_pretrained_embeddings


def get_model_param():
    """
    :return: Return model paramter object.
    """
    params_cp = falconn.LSHConstructionParameters()
    params_cp.dimension = len(dataset[0])
    params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
    params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
    params_cp.l = number_of_tables
    # we set one rotation, since the data is dense enough,
    # for sparse data set it to 2
    params_cp.num_rotations = 1
    params_cp.seed = 5721840
    # we want to use all the available threads to set up
    params_cp.num_setup_threads = 0
    params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
    return params_cp


if __name__ == '__main__':

    # # Read the first N lines
    # sample_lines = 1000
    # with open('falconn/dataset/glove.840B.300d.txt', 'r')as myfile:
    #     head = [next(myfile) for x in range(sample_lines)]
    # print(len(head))
    #
    # # Convert to numpy data
    # all_lines = []
    # for line in head:
    #     row = [float(x) for x in line.split()[1:]]
    #     all_lines.append(row)
    # all_lines = np.array(all_lines)
    # dataset = all_lines

    # Read the first N lines
    sample_lines = 1000
    cap = 100
    df = pd.read_csv('data/algo_proj.csv')
    df = df.head(sample_lines)
    abstracts = df['abstract'].to_numpy()
    abstracts = [str(x) for x in abstracts]
    # lengths = [len(x) for x in abstracts]
    # abstracts_capped = [x[0:cap] for x in abstracts]
    model_name = 'paraphrase-distilroberta-base-v1'
    embeddings = get_pretrained_embeddings(model_name, abstracts)
    dataset = embeddings

    number_of_queries = 100
    # we build only 50 tables, increasing this quantity will improve the query time
    # at a cost of slower preprocessing and larger memory footprint, feel free to
    # play with this number
    number_of_tables = 20

    # It's important not to use doubles, unless they are strictly necessary.
    # If your dataset consists of doubles, convert it to floats using `astype`.
    # assert dataset.dtype == np.float32
    assert dataset.dtype == np.float32

    # Normalize all the lenghts, since we care about the cosine similarity.
    print('Normalizing the dataset')
    dataset /= np.linalg.norm(dataset, axis=1).reshape(-1, 1)
    print('Done')

    # Choose random data points to be queries.
    print('Generating queries')
    np.random.seed(4057218)
    np.random.shuffle(dataset)
    queries = dataset[len(dataset) - number_of_queries:]
    dataset = dataset[:len(dataset) - number_of_queries]
    print('Done')

    # Perform linear scan using NumPy to get answers to the queries.
    print('Solving queries using linear scan')
    t1 = timeit.default_timer()
    answers = []
    for query in queries:
        answers.append(np.dot(dataset, query).argmax())
    t2 = timeit.default_timer()
    print('Done')
    print('Linear scan time: {} per query'.format((t2 - t1) / float(
        len(queries))))

    # Center the dataset and the queries: this improves the performance of LSH quite a bit.
    print('Centering the dataset and queries')
    center = np.mean(dataset, axis=0)
    dataset -= center
    queries -= center
    print('Done')

    # Get model paramter
    params_cp = get_model_param()

    # we build 18-bit hashes so that each table has
    # 2^18 bins; this is a good choise since 2^18 is of the same
    # order of magnitude as the number of data points
    falconn.compute_number_of_hash_functions(18, params_cp)

    print('Constructing the LSH table')
    t1 = timeit.default_timer()
    table = falconn.LSHIndex(params_cp)
    table.setup(dataset)
    t2 = timeit.default_timer()
    print('Done')
    print('Construction time: {}'.format(t2 - t1))

    query_object = table.construct_query_object()

    # find the smallest number of probes to achieve accuracy 0.9
    # using the binary search
    print('Choosing number of probes')
    number_of_probes = number_of_tables

    def evaluate_number_of_probes(number_of_probes):
        query_object.set_num_probes(number_of_probes)
        score = 0
        for (i, query) in enumerate(queries):
            if answers[i] in query_object.get_candidates_with_duplicates(
                    query):
                score += 1
        return float(score) / len(queries)

    while True:
        accuracy = evaluate_number_of_probes(number_of_probes)
        print('{} -> {}'.format(number_of_probes, accuracy))
        if accuracy >= 0.9:
            break
        number_of_probes = number_of_probes * 2

    if number_of_probes > number_of_tables:
        left = number_of_probes // 2
        right = number_of_probes
        while right - left > 1:
            number_of_probes = (left + right) // 2
            accuracy = evaluate_number_of_probes(number_of_probes)
            print('{} -> {}'.format(number_of_probes, accuracy))
            if accuracy >= 0.9:
                right = number_of_probes
            else:
                left = number_of_probes
        number_of_probes = right
    print('Done')
    print('{} probes'.format(number_of_probes))

    # final evaluation
    t1 = timeit.default_timer()
    score = 0
    for (i, query) in enumerate(queries):
        if query_object.find_nearest_neighbor(query) == answers[i]:
            score += 1
    t2 = timeit.default_timer()

    print('Query time: {}'.format((t2 - t1) / len(queries)))
    print('Precision: {}'.format(float(score) / len(queries)))
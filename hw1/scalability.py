import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import itertools
import glob
import matplotlib.pyplot as plt
import time

MAXIMUM_BYTES = 4  # the maximum number of bytes that we want to use to represent an integer
maximum_bits = 2 ** (MAXIMUM_BYTES * 8) - 1  # the number of bits used to represent an integer

convert_back_to_string = {} # key: integer from hashed string, value: string



def simple_hash(string_to_hash):
    """
    Uses Python's built-in hashing function to hash a string into a integer. Then performs a modulo operation to
    convert the resulting integer into an integer that can be represented using 4 bytes.
    :param string_to_hash: string to be hashed
    :return: a hash value of the string that can be represented with 4 bytes
    """
    hashed_string = hash(string_to_hash.lower())
    res = hashed_string % maximum_bits
    convert_back_to_string[res] = string_to_hash
    return res


def k_shingle(doc, k=9):
    """
    Represents a document in the form of a set of its hashed k-shingles.
    :param doc: a document to be shingled
    :param k: number of characters per shingle
    :return: a set with the document's hashed k-shingles
    """
    substring_shingles = [doc[i:i + k] for i in range(len(doc) - k + 1)]
    hashed_shingles_set = set(list(map(lambda x: simple_hash(x), substring_shingles)))
    return hashed_shingles_set


def compare_sets(set1, set2):
    """
    Jaccard Similarity of two sets. This measure of similarity is defined as the cardinality of the intersection of the
    two sets divided by the cardinality of the union of the two sets.
    :param set1: a set of integers
    :param set2: another set of integers
    :return: the Jaccard Similarity of the two sets
    """
    set1, set2 = set(set1), set(set2)
    return round(len(set1.intersection(set2)) / len(set1.union(set2)), 3)


def hashing_f(x, a, b, p):

    return ((a * x + b) % p) % maximum_bits


def min_hash(characteristic_matrix, n, a, b, p):
    """
    Calculates the min hash signature matrix using the characteristics matrix of the documents
    :param characteristic_matrix: a matrix with row values in first column, and boolean entries in every cell that
            indicate if a document contains a specific shingle or not
    :param n: number of hashing functions
    :param a: list of random parameters for the first parameter to be used in the universal hashing function
    :param b: list of random parameters for the second parameter to be used in the universal hashing function
    :param p: a prime number to be used in the universal hashing function
    :return:
    """

    #for row in characteristic_matrix:
    #    print(convert_back_to_string[row[0]], row[1], row[2])

    signature_matrix = np.full((n, len(characteristic_matrix[0]) - 1), np.inf)

    for row in characteristic_matrix:
        row_value = row[0]

        row_hashes = []
        for i in range(n):
            row_hashes.append(hashing_f(row_value, a[i], b[i], p)) # store hashed values for every hash function
                                                                   # with some shingle as input, to be used in next
                                                                   # for-loop when a one is encountered in a row/col.

        for c in range(1, len(row)):
            if row[c] == 0:
                continue  # this is equal to "do nothing" -> just continue the loop
            else:
                for i in range(n):
                    signature_matrix[i][c - 1] = min(signature_matrix[i][c - 1], row_hashes[i])

    return signature_matrix


def LSH(signature_matrix, a, b, p, t=0.8, bands=20):
    """
    Returns candidate pairs to compare
    :param signature_matrix:
    :param a: a random parameter for the first parameter to be used in the universal hashing function
    :param b: a random parameter for the second parameter to be used in the universal hashing function
    :param p: a prime number to be used in the universal hashing function
    :param bands: determines the number of sub-matrices that the signature matrix will be divided into
    :param t: threshold of similar components for candidate pairs
    :return: pairs with atleast t similar components
    """

    r = int(signature_matrix.shape[0] / bands)
    cols = np.array([x for x in range(signature_matrix.shape[1])])
    candidate_pairs = []

    for row in range(0, signature_matrix.shape[0], r): # iterating over bands

        band = np.sum(signature_matrix[row:row+r, cols], axis=0) # sum each column into a single value
        buckets = list(map(lambda x: hashing_f(x, a, b, p), band))
        unique_vals = np.unique(buckets)

        for val in unique_vals:
            pairs = np.where(buckets == val)[0] # find docs in same bucket
            if len(pairs) < 2: # no candidate pair found
                continue

            candidate_pairs += list(itertools.combinations(cols[pairs], 2)) # store candidate pairs

    candidate_pairs = set(candidate_pairs) # Filter on unique pairs found
    similar_pairs = []

    for pair in candidate_pairs:    # Look for pairs with t similar components
        if compare_signatures(signature_matrix, pair[0], pair[1]) >= t:
            similar_pairs.append(pair)

    return similar_pairs


def compare_signatures(signature_matrix, doc_idx1, doc_idx2):
    """
    Estimates the similarity of two minhash signatures as a fraction of components in which they agree.
    :param signature_matrix:
    :param doc_idx1:
    :param doc_idx2:
    :return:
    """
    # Note: [:, idx] is numpy syntax for selecting a specific column in a 2D-array
    return np.mean(signature_matrix[:, doc_idx1] == signature_matrix[:, doc_idx2])


def implementation(document_subset, a, b, NUMBER_OF_HASHING_FUNCTIONS, p):

    SHINGLE_SIZE = 2

    document_shingles = []  # list of lists with hashed shingles for every document
    union_shingles = set()
    for document in document_subset:
        doc_shingles = k_shingle(document, SHINGLE_SIZE)  # Hashed shingles for a document
        document_shingles.append(sorted(doc_shingles))
        union_shingles.update(doc_shingles)

    union_shingles = sorted(union_shingles)

    # implementation from https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    indptr = [0]
    indices = []
    data = []
    shingles = []

    for idx, shingle in enumerate(union_shingles):
        indices.append(idx)
        data.append(shingle)
    indptr.append(len(indices))

    for d in document_shingles:
        for shingle in d:
            index = union_shingles.index(shingle)
            indices.append(index)
            data.append(1)
            if shingle not in shingles:
                shingles.append(shingle)
        indptr.append(len(indices))

    # DATA: [ Hashed shingles, as many ones as shingles (?) ]
    # INDICES: [ range( # shingles ), corresponding index for each shingle in data ]
    # INDPTR: ?

    characteristic_matrix = csr_matrix((data, indices, indptr), dtype=int).toarray().transpose()
    # First column, every hashed shingle, followed by 1 for columns containing that shingle and 0 otherwise

    signature_matrix = min_hash(characteristic_matrix, NUMBER_OF_HASHING_FUNCTIONS, a, b, p)

    return signature_matrix



def plot_xy(y, x, z):

    plt.title("Elapsed time/document, average from 10 runs")
    plt.xlabel("Elapsed time (s)")
    plt.ylabel("Number of documents")

    plt.plot(x, y)
    plt.scatter(x, y)
    plt.plot(z, y)
    plt.scatter(z, y)

    plt.show()

    return




def main():

    #documents_raw = ['rävenraskaröverisen', 'nävenraskaröverisen', 'random', 'pandom', 'fävenraskaröverisen']

    document_paths = glob.glob("**/*.txt", recursive=True)
    documents_raw = []
    for doc_path in document_paths:
        try:
            with open(doc_path) as f:
                documents_raw.append(f.read())
        except UnicodeDecodeError:
            print(f'File with path {doc_path} could not be decoded correctly. Skipping...')


    NUMBER_OF_HASHING_FUNCTIONS = 100
    p = 4294967311


    ### similar pairs found with and without lsh, atleast t
    sim_docs = []
    sim_docs_lsh = []

    ### Used for plotting ###
    elapsed_time = []
    elapsed_time_lsh = []
    num_docs = []
    #########################

    for size in range(10, 20): # pairs to compare in an iteration, takes 3 min to run this loop

        document_subset = documents_raw[:size]
        average = []
        average_lsh = []

        for iter in range(10): # to get average time from 10 runs to find sim pair with lsh and without lsh

            t0 = time.perf_counter()

            a = [np.random.randint(1, p // 2) for _ in
                 range(NUMBER_OF_HASHING_FUNCTIONS)]  # Random num from lower half of prime-range
            b = [np.random.randint(0, p) for _ in range(NUMBER_OF_HASHING_FUNCTIONS)]  # Random num from prime-range

            signature_matrix = implementation(document_subset, a, b, NUMBER_OF_HASHING_FUNCTIONS, p)
            t1 = time.perf_counter()


            check_pairs = itertools.combinations([doc_idx for doc_idx in range(0, size)], 2)

            sim_d = [f"SIZE: {size}"]
            for pair in check_pairs:
                sim = compare_signatures(signature_matrix, pair[0], pair[1])
                if sim >= 0.8:
                    sim_d.append(pair)


            t2 = time.perf_counter()
            candidate_pairs = LSH(signature_matrix, a[0], b[0], p) # returns candidate pairs with atleast t similar
            t3 = time.perf_counter()                               # components

            if iter == 9:
                sim_docs.append(sim_d)
                sim_docs_lsh.append([f"SIZE: {size}", candidate_pairs])




            average.append(t2-t0) # time to find sim docs without lsh (i.e. comparing every doc with every other in
                                  # signature matrix)

            average_lsh.append(t1 + t3 - t0 - t2) # time to find sim docs with lsh

        elapsed_time.append(np.mean(average)) 
        elapsed_time_lsh.append(np.mean(average_lsh))
        num_docs.append(size)


    print(sim_docs)
    print(sim_docs_lsh)
    plot_xy(num_docs, elapsed_time, elapsed_time_lsh)

    #candidate_pairs = LSH(signature_matrix, a[0], b[0], p)

#    print('True similarity:', compare_sets(document_shingles[0], document_shingles[1]), 'Estimation:',
#          compare_signatures(signature_matrix, 0, 1))
#    print('True similarity:', compare_sets(document_shingles[0], document_shingles[2]), 'Estimation:',
#          compare_signatures(signature_matrix, 0, 2))
#    print('True similarity:', compare_sets(document_shingles[1], document_shingles[2]), 'Estimation:',
#          compare_signatures(signature_matrix, 1, 2))

    #print(candidate_pairs)


if __name__ == '__main__':
    main()

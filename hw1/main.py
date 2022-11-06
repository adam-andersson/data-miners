import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

MAXIMUM_BYTES = 4  # the maximum number of bytes that we want to use to represent an integer
maximum_bits = 2 ** (MAXIMUM_BYTES * 8) - 1  # the number of bits used to represent an integer

convert_back_to_string = {}


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

    for row in characteristic_matrix:
        print(convert_back_to_string[row[0]], row[1], row[2])

    signature_matrix = np.full((n, len(characteristic_matrix[0]) - 1), np.inf)

    for row in characteristic_matrix:
        row_value = row[0]

        row_hashes = []
        for i in range(n):
            row_hashes.append(hashing_f(row_value, a[i], b[i], p))

        for c in range(1, len(row)):
            if row[c] == 0:
                continue  # this is equal to "do nothing" -> just continue the loop
            else:
                for i in range(n):
                    signature_matrix[i][c - 1] = min(signature_matrix[i][c - 1], row_hashes[i])

    return signature_matrix


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


def main():
    SHINGLE_SIZE = 2
    NUMBER_OF_HASHING_FUNCTIONS = 100
    p = 4294967311  # the first prime number after 2^32 - 1.
    a = [np.random.randint(1, p//2) for _ in range(NUMBER_OF_HASHING_FUNCTIONS)]
    b = [np.random.randint(0, p) for _ in range(NUMBER_OF_HASHING_FUNCTIONS)]

    documents_raw = ['rävenraskaröverisen', 'nävenraskaröverisen', 'random']
    document_shingles = []

    union_shingles = set()
    for document in documents_raw:
        doc_shingles = k_shingle(document, SHINGLE_SIZE)
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

    characteristic_matrix = csr_matrix((data, indices, indptr), dtype=int).toarray().transpose()

    signature_matrix = min_hash(characteristic_matrix, NUMBER_OF_HASHING_FUNCTIONS, a, b, p)

    print('True similarity:', compare_sets(document_shingles[0], document_shingles[1]), 'Estimation:',
          compare_signatures(signature_matrix, 0, 1))
    print('True similarity:', compare_sets(document_shingles[0], document_shingles[2]), 'Estimation:',
          compare_signatures(signature_matrix, 0, 2))
    print('True similarity:', compare_sets(document_shingles[1], document_shingles[2]), 'Estimation:',
          compare_signatures(signature_matrix, 1, 2))


if __name__ == '__main__':
    main()

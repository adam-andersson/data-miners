MAXIMUM_BYTES = 4  # the maximum number of bytes that we want to use to represent an integer
maximum_bits = 2 ** (MAXIMUM_BYTES * 8) - 1  # the number of bits used to represent an integer


def simple_hash(string_to_hash):
    """
    Uses Python's built-in hashing function to hash a string into a integer. Then performs a modulo operation to
    convert the resulting integer into an integer that can be represented using 4 bytes.
    :param string_to_hash: string to be hashed
    :return: a hash value of the string that can be represented with 4 bytes
    """
    hashed_string = hash(string_to_hash.lower())
    return hashed_string % maximum_bits


def k_shingle(doc, k):
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
    return round(len(set1.intersection(set2)) / len(set1.union(set2)), 3)


räven = k_shingle('räven raskar över isen', 3)
apan = k_shingle('apan raskar över isen', 3)

print(compare_sets(räven, apan))

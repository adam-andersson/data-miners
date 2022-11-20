import argparse
import itertools
import time


def read_transaction_dataset(filepath="T10I4D100K.dat"):
    """
    Reads a file with baskets from the specified filepath and creates a list of list of the baskets in the file.
    :param filepath: filepath to the file that contains a sale transaction dataset
    :return: list of lists, each list is a basket, each item in a basket is an integer
    """
    with open(filepath, "r") as f:
        read_file = f.readlines()

    list_of_baskets = []

    for line in read_file:
        split_line = line.strip(' \n').split(" ")  # remove the newline characters at the end of line and
        # split every item
        basket = list(map(int, split_line))  # map hashed values to integers
        list_of_baskets.append(basket)

    return list_of_baskets


def a_priori_first_pass(list_of_baskets):
    """
    Counts the number of times an item occurs in all baskets and stores this number in an array at the index
    corresponding to the hashed item.
    :param list_of_baskets: list of baskets containing one or several hashed value items
    :return: list containing number of times every item have occurred
    """
    occurrences_of_items = []  # use the hashed value of an item to index into the array of counts of all hashed values

    for basket in list_of_baskets:
        for item in basket:
            if item + 1 > len(occurrences_of_items):
                diff = item - len(occurrences_of_items)
                occurrences_of_items += [0 for _ in range(diff)]
                occurrences_of_items.append(1)
            else:
                occurrences_of_items[item] += 1

    return occurrences_of_items


def create_frequent_items_table(occurrences, num_baskets, threshold=0.01):
    """
    Creates a frequent-items table that depicts if an item is frequent or not in the dataset, if an item is frequent
    or not is determined by the threshold.
    :param occurrences: list containing number of times every item have occurred
    :param num_baskets: number of baskets in the dataset
    :param threshold: aka s - an item is frequent if it appears in s percent of baskets.
    :return: a tuple - (frequent-items table, and L1)
    """
    L_1 = []  # L_1 is a list of tuples with only one element per tuple. It contains one tuple for every frequent item.
    freq_items_table = []  # array indexed 0 to n-1, where the entry for i is either 0, if item i is not frequent or a
    # unique integer if item i is frequent

    required_occurrences_to_be_frequent = threshold * num_baskets  # calculate this number once so we do not have to
    # do division for every item's frequency

    for idx, item_occ in enumerate(occurrences):
        if item_occ >= required_occurrences_to_be_frequent:
            freq_items_table.append(idx)  # append the hashed item-value to the frequent-items table.
            L_1.append([(idx,), item_occ / num_baskets])  # append a list with the singleton as a tuple and with the
            # item's support to L_1
        else:
            freq_items_table.append(0)

    return freq_items_table, L_1


def a_priori_second_pass(list_of_baskets, frequent_items, L_k, k, num_basket, threshold=0.01):
    """
    Performs the A-priori second pass to find frequent item-sets of length k+1.
    :param list_of_baskets: list of lists, each list is a basket, each item in a basket is an integer
    :param frequent_items: array indexed 0 to n-1, where the entry for i is either 0, if item i is not frequent or a
    unique integer if item i is frequent
    :param L_k: list of frequent item(sets) of length k
    :param k: length of item-sets from the previous pass
    :param num_basket: number of baskets in the data set
    :param threshold: the support threshold for a item set to be considered frequent
    :return: a list of tuple/triplet/quadruples that are frequent item-sets of length k+1
    """
    next_frequent_items = []
    frequent_in_L_k = [item[0] for item in L_k]  # L_k is a list of list, where every inner list has a frequent
    # item/item-set on the first index, and the support of this item/item-set on the second index. Extract only the
    # frequent item/item-sets.

    for basket in list_of_baskets:
        count_frequent_items_in_basket = {}
        for item in basket:
            if frequent_items[item] > 0:
                count_frequent_items_in_basket[frequent_items[item]] = 0

        frequent_items_in_basket = list(count_frequent_items_in_basket.keys())  # freq items found in basket

        # all k-item sets out of the frequent items found in the basket
        # here, we assume that the potential frequent k-item-sets in a basket << the number of frequent items found
        potential_frequent_k_itemsets_in_basket = list(itertools.combinations(frequent_items_in_basket, k))

        # iterate over item-sets consisting of k items in basket
        for itemset_k in potential_frequent_k_itemsets_in_basket:
            if itemset_k in frequent_in_L_k:  # check if item-set is frequent by looking it up in L_k
                for item in itemset_k:
                    count_frequent_items_in_basket[item] += 1  # an item must appear at least k times in basket to be
                    # part of a frequent k+1-item set found in basket

        # Filter on items meeting criteria of appearing k times
        candidates = [i for i in frequent_items_in_basket if count_frequent_items_in_basket[i] >= k]

        if len(candidates) >= k + 1:  # no candidate item sets of size k+1 can be created if this criteria is not met
            candidates = list(
                itertools.combinations(candidates, k + 1))  # create candidate k+1-item sets for the basket
            next_frequent_items += candidates  # add the candidate k+1-item sets to the list of all candidates
            # found in all baskets

    one_occurrence_value = 1 / num_basket  # determine how much one occurrence of an item-set in a basket should
    # contribute to this item-set's total

    support = dict()
    for i in next_frequent_items:
        # count in how many baskets each candidate item-set appears
        support[i] = support.get(i, 0) + one_occurrence_value

    # the frequent item-sets of length k+1 are those that appears in at least the threshold's frequency of baskets.
    next_L = []
    for key, value in support.items():
        if value >= threshold:
            next_L.append([key, value])

    return next_L


def calculate_association_confidence(union_set, arrow_position, item_set_support_lookup):
    """
    Finds the confidence level of an association rule by looking up support for subsets.
    :param union_set: an item-set tuple, e.g., (A, ) or (A, B) or (A, B, C)
    :param arrow_position: an integer that denotes where the item-set tuple is split into two subsets, i.e.,
    (1) item_set = (A, B, C) and arrow_position = 1 would mean we are investigating the rule (A, ) -> (B, C)
    (2) item_set = (A, B, C) and arrow_position = 2 would mean we are investigating the rule (A, B) -> (C)
    :param item_set_support_lookup: a dictionary to lookup the support for any frequent item-set tuple. The key of the
    dictionary items is the tuple, e.g., (A, ) or (A, B) and the value at that key is the support for that item-set.
    :return: a float, the ratio between the support for the union item-set to the support for the first set in the rule.
    """
    first_set = union_set[:arrow_position]  # the item-set to the left of the arrow in the association rule

    # sort the first item-set and the union item-set to be able to lookup in the dictionary
    first_set_sorted = tuple(sorted(first_set))
    first_set_support = item_set_support_lookup[first_set_sorted]

    union_set_sorted = tuple(sorted(union_set))
    union_set_support = item_set_support_lookup[union_set_sorted]

    return union_set_support / first_set_support


def main():
    # --- Expose command line arguments --- #
    parser = argparse.ArgumentParser(
        prog='Frequent Item-sets Finder',
        description='Finds frequent item-sets and association rules using A-priori algorithm.',
    )

    parser.add_argument('-p', '--path', type=str, default='T10I4D100K.dat',
                        help='Specify the path to the transaction data set to be used.')

    parser.add_argument('-s', '--support', type=float, default=0.01, metavar='f',
                        help='The support for I is the number of baskets for which I is a subset.'
                             'Should be a float between 0 and 1.')

    parser.add_argument('-c', '--confidence', type=float, default=0.5, metavar='f',
                        help='The confidence of the rule is the fraction of the baskets with all of I that '
                             'also contain j. Should be a float between 0 and 1.')

    args = parser.parse_args()

    # Update the constants to the values of the args or default values
    DATASET_FILEPATH = args.path
    S_THRESHOLD = args.support
    CONFIDENCE = args.confidence

    print(f'Using a support threshold of {S_THRESHOLD} and a confidence level of {CONFIDENCE} \n')
    # --- #

    # --- Setup and first pass --- #
    starting_time = time.perf_counter()
    baskets = read_transaction_dataset(DATASET_FILEPATH)
    number_of_baskets = len(baskets)

    item_occurrences = a_priori_first_pass(baskets)

    frequent_items_table, L_1 = create_frequent_items_table(item_occurrences, number_of_baskets, S_THRESHOLD)

    print(f'Found {len(L_1)} frequent items. \n')
    # --- #

    # --- Finding frequent item-sets (of length >= 2) --- #
    all_frequent_items = []

    L_i = L_1
    i = 2
    while len(L_i) > 0:
        all_frequent_items.append(L_i)
        L_i = a_priori_second_pass(baskets, frequent_items_table, L_i, i - 1, number_of_baskets, S_THRESHOLD)

        if len(L_i) > 0:
            print(f'Found {len(L_i)} frequent item-sets with length {i} \n'
                  f'{L_i} \n')
            i += 1

    print(f'Found no frequent items of length {i} \n')
    # --- #

    # --- Finding association rules --- #
    item_set_support_lookup = {}
    for frequent_item_k in all_frequent_items:  # iterate over L_1, L_2, ..., L_i's frequent item-sets
        for frequent_item in frequent_item_k:
            item, support_for_item = frequent_item[0], frequent_item[1]
            item_set_support_lookup[item] = support_for_item

    found_associations = {}

    # flatten all item-sets and filter on item-sets of length >= 2
    frequent_item_sets = [item_sets[0] for item_sets_of_len_k in all_frequent_items[1:]
                          for item_sets in item_sets_of_len_k]

    for freq_item_set in frequent_item_sets:
        # permute the item-set to create all possible fragmented sets of the item-set.
        item_set_permutations = itertools.permutations(freq_item_set, len(freq_item_set))
        for permutation in item_set_permutations:
            # Start with the arrow as far "to the right" as possible to make the left subset as large as possible,
            # since this would mean not having to continue the loop if we found that this subset is not confident.
            for arrow_pos in range(len(permutation) - 1, 0, -1):
                association_confidence = calculate_association_confidence(permutation, arrow_pos,
                                                                          item_set_support_lookup)
                if association_confidence >= CONFIDENCE:
                    association_key = str(str(sorted(permutation[:arrow_pos])) + '$' +
                                          str(sorted(permutation[arrow_pos:])))
                    # do not add duplicate assoc rules, e.g., {A, B} -> {C} == {B, A} -> {C}, thus only add it once
                    if association_key not in found_associations:
                        found_associations[association_key] = association_confidence
                else:
                    # If {K,L,M} -> {N} is below confidence, so is {K,L} -> {M,N}
                    break

    for assoc_rule, assoc_conf in found_associations.items():
        # the association rule is on the form: [A, B]$[C], do some cleaning to prettify the prints
        cleaned_rule = assoc_rule.replace('[', '').replace(']', '')
        set1, set2 = cleaned_rule.split('$')

        # convert back key to a tuple to lookup it's support
        set2_tuple = tuple(map(int, set2.split(', ')))
        assoc_interest = assoc_conf - item_set_support_lookup[set2_tuple]

        print(f'Found association [Confidence {round(assoc_conf, 3)}] [Interest {round(assoc_interest, 3)}]: '
              f'{{{ set1 }}} -> {{{ set2 }}}')


    # --- #

    # --- Calculating run time --- #
    end_time = time.perf_counter()
    print(f'\n--- Execution Time: {round(end_time-starting_time, 3)} seconds ---')


if __name__ == '__main__':
    main()



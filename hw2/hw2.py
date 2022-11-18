import numpy as np
import itertools



def read_data(file):
    """

    :param file: Name of file containing data
    :return: list of lists, each list is a basket
    """

    file = open(file)
    file = file.readlines()
    dataset = []

    for line in file:
        line = line.split(" ")
        basket = []

        for item in line:
            try:
                if int(item) in basket:
                    pass
                else:
                    basket.append(int(item))
            except:
                pass

        basket = list(basket)
        dataset.append(basket)

    return dataset


def first_pass(dataset): # C_1
    """
    Counts the number of times an item occurs in all baskets and stores this number in an array at
    the index corresponding to the hashed item.
    :param file: dataset with baskets
    :return: Array containing number of times items have occured
    """

    occurences = []

    count = 0 # keep track of number of baskets

    for basket in dataset:
        count += 1

        for item in basket:
            try:
                item = int(item)

                if item + 1 > len(occurences):
                    diff = item - len(occurences)
                    occurences += [0 for _ in range(diff)]
                    occurences.append(1)
                else:
                    occurences[item] += 1
            except: pass

    return occurences, count


def frequent(occurences, num_baskets, threshold=0.01): # used for L1 to find frequent pairs


    L1 = []
    frequent_items = []
    for index, item in enumerate(occurences):
        if item / num_baskets >= threshold:
            frequent_items.append(index)
            L1.append((index,))

        else:
            frequent_items.append(0)

                                # L1 is a list of tuples with only one element per tuples. It contains all frequent items
    return frequent_items, L1   # and this format is neccessary in order to send it as input to second_pass()


def second_pass(dataset, frequent_items, L_k, k, num_basket, threshold=0.01): # C_2, L_2

    support = {}
    count = []
    L_kplus1 = set()
    f = []

    p = 0
    for basket in dataset:
        investigate = {}
        for item in basket:
            if frequent_items[item] != 0:
                investigate[frequent_items[item]] = 0

        freq = list(investigate.keys())   # frequent items found in basket
        freq_k = list(itertools.combinations(freq, k)) # create all k-itemsets out of the frequent items


        for itemset_k in freq_k:            # iterate over itemsets consisting of k items in basket
            if itemset_k in L_k:            # check if itemset is frequent by looking it up in L_k
                for item in itemset_k:
                    investigate[item] += 1 # an item most appear atleast k times in basket to be part of a
#                                                # frequent k+1-itemset found in basket


        candidates = [i for i in freq if investigate[i] >= k] # Filter on items meeting criteria
                                                                           # of appearing k times


        if len(candidates) >= k+1: # no candidate itemsets of size k+1 can be created if this criteria is not met
            candidates = list(itertools.combinations(candidates, k+1)) # create candidate k+1-itemsets
            count += candidates

    for candidate in count:                         # check candidates that meet support criteria
        if candidate in support.keys():
            support[candidate] += 1 / num_basket
        else:
            support[candidate] = 1 / num_basket

        if support[candidate] >= threshold:
            L_kplus1.add(candidate)

    return list(L_kplus1)





data = read_data("../edward/T10I4D100K.dat")
#data = read_data("test.txt")

occ, number_of_baskets = first_pass(data)

freq, L1 = frequent(occ, number_of_baskets, 0.01) # 0.5 threshold for test.txt


L2 = second_pass(data, freq, L1, 1, number_of_baskets, 0.01)
print("Number of pairs in L2: ", len(L2), "\n L2: ", L2, "\n")
#

L3 = second_pass(data, freq, L2, 2, number_of_baskets, 0.01) # 0.5 threshold for test.txt

print("Number of triples in L3: ", len(L3), "\n L3: ", L3)

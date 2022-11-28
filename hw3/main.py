import argparse
import random
from collections import defaultdict
from tqdm import tqdm


class Graph:
    """
    Implementation of an undirected graph using adjacency lists, and a set that contains all the edges in the graph.
    """
    def __init__(self):
        self.adjacency_list = defaultdict(set)
        self.all_edges = set()

    def add_edge(self, u, v):
        """
        Adds an edge (u, v) to the adjacency lists of the parameter nodes and to the set of all edges in the graph.
        (u, v) is added as both (u, v) and (v, u) for the graph to be undirectional.
        :param u: a source node
        :param v: a destination node
        """
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)

        self.all_edges.add((u, v))

    def remove_edge(self, u, v):
        """
        Removes an edge (u, v) from the nodes' adjecency lists and from the set of all edges in the graph.
        :param u: a source node
        :param v: a destination node
        """
        self.adjacency_list[u].remove(v)
        self.adjacency_list[v].remove(u)

        self.all_edges.remove((u, v))


class TriestBase:
    """
    An implementation of the base version of the Triest algorithm.
    Source: http://www.kdd.org/kdd2016/papers/files/rfp0465-de-stefaniA.pdf
    """
    def __init__(self, M):
        self.M = M
        self.sample_graph = Graph()
        self.tao = 0
        self.t = 0

    def calculate_xi(self):
        """
        Calculates Xi which is used to scale tao to estimate the global triangles count.
        """
        nominator = self.t * (self.t - 1) * (self.t - 2)
        denominator = self.M * (self.M - 1) * (self.M - 2)

        return max(1, nominator / denominator)

    def estimate_global_triangles(self):
        """
        Estimates the count of global triangles
        :return: for t<M: the count of global triangles, else: the estimation of global triangles
        """
        if self.t < self.M:
            return self.tao
        else:
            return self.calculate_xi() * self.tao

    def update_counter(self, operation, u, v):
        """
        Updates the global counter, tao, that is used to estimate triangles in the graph.
        Tao is incremented/decremented for every common neighbor of u and v.
        :param operation: either + or -, depending on if we have just added (+) or removed (-) an edge.
        :param u: a source node
        :param v: a destination node
        """
        neighbourhood_uv = self.sample_graph.adjacency_list[u].intersection(self.sample_graph.adjacency_list[v])

        for _ in range(len(neighbourhood_uv)):
            if operation == '+':
                self.tao += 1
            elif operation == '-':
                self.tao -= 1

    def flip_biased_coin(self):
        """
        Flips a biased coins. The bias is decided by the current time-step and the maximum edge sample size.
        :return: boolean, if coin landed heads or tail.
        """
        probability = self.M / self.t
        return random.random() < probability

    def sample_edge(self):
        """
        Determines if an edge should be added or not. If the sample graph is not full yet, it should always be.
        Otherwise, we flip a biased coin:
        - if it lands head -> we remove a random edge; update our counter, tao; and return True
        - if the coin lands tails -> we return false.
        :return: boolean, true if an edge should be added to the sample graph, otherwise false
        """
        if self.t <= self.M:
            return True
        elif self.flip_biased_coin():
            # in case of 'heads' on our biased coin flip
            source_node, destination_node = random.choice(list(self.sample_graph.all_edges))
            self.sample_graph.remove_edge(source_node, destination_node)
            self.update_counter('-', source_node, destination_node)
            return True
        return False

    def new_edge_from_stream(self, u, v):
        """
        Handles the logic of potentially adding a new edge from the stream to the sample graph.
        - Updates the time-step, t, every time the method is called.
        - Updates the counter, tao, based on the edge (u, v) only if the edge was added to the sample graph.
        :param u: a source node
        :param v: a destination node
        """
        self.t += 1
        if self.sample_edge():
            self.sample_graph.add_edge(u, v)
            self.update_counter('+', u, v)


class TriestImpr:
    """
    An implementation of the improved (lower variance) version of the Triest algorithm.
    Source: http://www.kdd.org/kdd2016/papers/files/rfp0465-de-stefaniA.pdf
    """
    def __init__(self, M):
        self.M = M
        self.sample_graph = Graph()
        self.tao = 0
        self.t = 0

    def tao_scaling_xi(self):
        """
        Calculates Xi, the scaler of the counter, tao.
        """
        nominator = (self.t - 1) * (self.t - 2)
        denominator = self.M * (self.M - 1)

        return max(1, nominator / denominator)

    def estimate_global_triangles(self):
        """
        Estimates the count of global triangles
        :return: the estimation, which is estimated by the counter, tao.
        """
        return self.tao

    def update_counter(self, u, v):
        """
        Updates the global counter, tao, that is used to estimate triangles in the graph.
        Tao is increased by Xi for every common neighbor of u and v.
        :param u: a source node
        :param v: a destination node
        """
        neighbourhood_uv = self.sample_graph.adjacency_list[u].intersection(self.sample_graph.adjacency_list[v])

        for _ in range(len(neighbourhood_uv)):
            self.tao += self.tao_scaling_xi()  # we scale tao with xi(t) when doing TriestImpr

    def flip_biased_coin(self):
        """
        Flips a biased coins. The bias is decided by the current time-step and the maximum edge sample size.
        :return: boolean, if coin landed heads or tail.
        """
        probability = self.M / self.t
        return random.random() < probability

    def sample_edge(self):
        """
        Determines if an edge should be added or not. If the sample graph is not full yet, it should always be.
        Otherwise, we flip a biased coin:
        - if it lands head -> we remove a random edge, and return true
        - if the coin lands tails -> we return false.
        :return: boolean, true if an edge should be added to the sample graph, otherwise false
        """
        if self.t <= self.M:
            return True
        elif self.flip_biased_coin():
            # in case of 'heads' on our biased coin flip
            source_node, destination_node = random.choice(list(self.sample_graph.all_edges))
            self.sample_graph.remove_edge(source_node, destination_node)
            # self.update_counter('-', source_node, destination_node)  # this is removed in the TriestImpr
            return True
        return False

    def new_edge_from_stream(self, u, v):
        """
        Handles the logic of potentially adding a new edge from the stream to the sample graph.
        - Updates the time-step, t, every time the method is called.
        - Updates the counter, tao, based on the edge (u, v)
        :param u: a source node
        :param v: a destination node
        """
        self.t += 1
        self.update_counter(u, v)  # moved to here in the TriestImpr
        if self.sample_edge():
            self.sample_graph.add_edge(u, v)


if __name__ == "__main__":
    # --- CONSTANTS --- #
    FILE_PATH = 'web-Stanford.txt'
    FILE_SPACE_SEP = '	'
    FILE_COMMENT_CHAR = '#'
    CORRECT_NO_TRIANGLES = 11329473
    # --- #

    # --- Expose command line arguments --- #
    parser = argparse.ArgumentParser(
        prog='Global Triangle Estimator',
        description='Estimates the count of global triangles in a graph by a stream of edges',
    )

    parser.add_argument('-m', type=int, default=15000,
                        help='The maximum number of edges to be included in the sample graph at any one time '
                             '[Default: 15000]')

    parser.add_argument('-b', '--base', action='store_true',
                        help='Use TriestBase. Default is to use TriestImpr.')

    args = parser.parse_args()

    # Update the constants to the values of the args or default values
    MAX_SET_SIZE = args.m
    USE_BASE = args.base

    print(f'Using a maximum edge sample size of {MAX_SET_SIZE} and the {"BASE" if USE_BASE else "IMPROVED" } '
          f'version of the Triest algorithm \n')
    # --- #

    seen_edges = set()
    if USE_BASE:
        triest_model = TriestBase(MAX_SET_SIZE)
    else:
        triest_model = TriestImpr(MAX_SET_SIZE)

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f):

            if line[0] == FILE_COMMENT_CHAR:
                # skip comments
                continue

            src_node, dest_node = sorted(list(map(int, line.split(FILE_SPACE_SEP))))

            edge_tuple = (src_node, dest_node)

            # Do not add reoccurring edges.
            # In our case (u, v) = (v, u), but the data set is directed so both edges can appear in the stream.
            if edge_tuple not in seen_edges:
                triest_model.new_edge_from_stream(src_node, dest_node)
                seen_edges.add(edge_tuple)

    global_triangles = int(triest_model.estimate_global_triangles())
    print(f'\nIn total, we found {global_triangles} triangles \n'
          f'Which is {int(abs(CORRECT_NO_TRIANGLES - global_triangles))} off of the true count.')

import random
from collections import defaultdict


class Graph:
    def __init__(self):
        self.adjacency_list = defaultdict(set)
        self.all_edges = set()

    def add_edge(self, u, v):
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)
        self.all_edges.add((u, v))
        self.all_edges.add((v, u))

    def remove_edge(self, u, v):
        self.adjacency_list[u].remove(v)
        self.adjacency_list[v].remove(u)
        self.all_edges.remove((u, v))
        self.all_edges.remove((v, u))

        if len(self.adjacency_list[u]) == 0:
            del self.adjacency_list[u]
        if len(self.adjacency_list[v]) == 0:
            del self.adjacency_list[v]


def flip_biased_coin(M, t):
    probability = M/t
    return random.random() < probability


def sample_edge(sample_graph, timestep, M):
    if timestep <= M:
        return True
    elif flip_biased_coin(M, timestep):
        # in case of 'heads' on our biased coin flip
        source_node, destination_node = random.choice(list(sample_graph.all_edges))
        sample_graph.remove_edge(source_node, destination_node)
        return True
    return False


def triest_base(graph, t, M, u, v):
    if sample_edge(graph, t, M):
        graph.add_edge(u, v)


if __name__ == "__main__":
    # --- CONSTANTS --- #
    FILE_PATH = 'web-Google.txt'
    NUMBER_OF_LINES_TO_READ = 1000
    MAX_SET_SIZE = 30
    # --- #
    graph = Graph()

    t = 0
    with open(FILE_PATH, 'r') as f:
        while t < NUMBER_OF_LINES_TO_READ:
            line = f.readline()

            if line[0] == '#':
                # skip comments
                continue

            t += 1
            src_node, dest_node = line.split('	')
            int_src, int_dest = int(src_node), int(dest_node)
            triest_base(graph, t, MAX_SET_SIZE, int_src, int_dest)

    print(graph.all_edges)
    print(graph.adjacency_list)
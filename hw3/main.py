import random
from collections import defaultdict
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    def __init__(self):
        self.adjacency_list = defaultdict(set)
        self.all_edges = set()

    def add_edge(self, u, v):
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)

        self.all_edges.add((u, v))

    def remove_edge(self, u, v):
        self.adjacency_list[u].remove(v)
        self.adjacency_list[v].remove(u)

        self.all_edges.remove((u, v))


class TriestBase:
    def __init__(self, M):
        self.M = M
        self.sample_graph = Graph()
        self.tao = 0
        self.t = 0

    def calculate_xi(self):
        nominator = self.t * (self.t - 1) * (self.t - 2)
        denominator = self.M * (self.M - 1) * (self.M - 2)

        return max(1, nominator / denominator)

    def estimate_global_triangles(self):
        if self.t < self.M:
            return self.tao
        else:
            return self.calculate_xi() * self.tao

    def update_counter(self, operation, u, v):
        neighbourhood_uv = self.sample_graph.adjacency_list[u].intersection(self.sample_graph.adjacency_list[v])

        for _ in range(len(neighbourhood_uv)):
            if operation == '+':
                self.tao += 1
                print(self.tao, '+')
            elif operation == '-':
                self.tao -= 1
                print(self.tao, '-')

    def flip_biased_coin(self):
        probability = self.M / self.t
        return random.random() < probability

    def sample_edge(self):
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
        self.t += 1
        if self.sample_edge():
            self.sample_graph.add_edge(u, v)
            self.update_counter('+', u, v)


if __name__ == "__main__":
    # --- CONSTANTS --- #
    FILE_PATH = 'web-Stanford.txt'
    FILE_SPACE_SEP = '	'
    FILE_COMMENT_CHAR = '#'
    MAX_SET_SIZE = 12000
    # --- #

    seen_edges = set()
    triest_base = TriestBase(MAX_SET_SIZE)

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f):

            if line[0] == FILE_COMMENT_CHAR:
                # skip comments
                continue

            src_node, dest_node = sorted(list(map(int, line.split(FILE_SPACE_SEP))))
            edge_tuple = (src_node, dest_node)

            if edge_tuple not in seen_edges:
                triest_base.new_edge_from_stream(src_node, dest_node)
                seen_edges.add(edge_tuple)

    global_triangles = triest_base.estimate_global_triangles()
    print(f'In total, we found {int(global_triangles)} triangles')

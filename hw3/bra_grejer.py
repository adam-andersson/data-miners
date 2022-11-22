import random
from collections import defaultdict

class EstimateTriangles:
    def __init__(self):
        self.tao = 0

    def updatecounter(self, add, S, u, v):

        neighbourhood_uv = list(S[u] & S[v])
        print("S[u]: ", S[u], "  S[v]: ", S[v], "   neighbourhood_uv: ", neighbourhood_uv)

        if add:
            for _ in neighbourhood_uv:
                self.tao += 1

        else:
            for _ in neighbourhood_uv:
                self.tao -= 1

    def estimate(self, t, M):

        nom = t * (t - 1) * (t - 2)
        denom = M * (M - 1) * (M - 2)

        return max(1, nom / denom)

    def estimate_global_traingles(self, t, M):

        if t < M:
            return self.tao

        else:
            return self.estimate(t, M) * self.tao


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


def sample_edge(sample_graph, timestep, M, tao):
    if timestep <= M:
        return True
    elif flip_biased_coin(M, timestep):
        # in case of 'heads' on our biased coin flip
        source_node, destination_node = random.choice(list(sample_graph.all_edges))
        sample_graph.remove_edge(source_node, destination_node)
        tao.updatecounter(False, sample_graph.adjacency_list, source_node, destination_node)
        return True
    return False


def triest_base(graph, t, M, u, v, tao):
    if sample_edge(graph, t, M, tao):
        graph.add_edge(u, v)
        tao.updatecounter(True, graph.adjacency_list, u, v)


if __name__ == "__main__":
    # --- CONSTANTS --- #
    FILE_PATH = 'web-Google.txt'
    NUMBER_OF_LINES_TO_READ = 1000
    MAX_SET_SIZE = 30
    # --- #
    graph = Graph()
    tao = EstimateTriangles()
    global_triangles = []
    t = 0


    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        while t < NUMBER_OF_LINES_TO_READ:
            line = f.readline()

            if line[0] == '#':
                # skip comments
                continue

            t += 1
            src_node, dest_node = line.split('	')
            int_src, int_dest = int(src_node), int(dest_node)
            triest_base(graph, t, MAX_SET_SIZE, int_src, int_dest, tao)
            global_triangles.append(tao.estimate_global_traingles(t, MAX_SET_SIZE))

    print(global_triangles)

    #print(graph.all_edges)
    #print(graph.adjacency_list)



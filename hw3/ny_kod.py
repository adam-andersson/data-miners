import random
from collections import defaultdict

class EstimateTriangles:
    def __init__(self):
        self.tao = 0

    def updatecounter(self, add, S, u, v):

        neighbourhood_uv = list(S[u] & S[v])

        if add:
            for _ in neighbourhood_uv:
                self.tao += 1

        else:
            for _ in neighbourhood_uv:
                self.tao -= 1

        if self.tao <= 0:
            self.tao = 1

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
        self.seen_edges = {}

    def add_edge(self, u, v):
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)
        self.all_edges.add((u, v))


    def remove_edge(self, u, v, t):
        self.adjacency_list[u].remove(v)
        self.adjacency_list[v].remove(u)
        self.all_edges.remove((u, v))




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
        sample_graph.remove_edge(source_node, destination_node, timestep)
        tao.updatecounter(False, sample_graph.adjacency_list, source_node, destination_node)
        return True
    return False


def triest_base(graph, t, M, min_node, max_node, tao):
    graph.seen_edges[(min_node, max_node)] = ""
    if sample_edge(graph, t, M, tao):
        graph.add_edge(min_node, max_node)
        tao.updatecounter(True, graph.adjacency_list, min_node, max_node)


if __name__ == "__main__":
    # --- CONSTANTS --- #
    FILE_PATH = 'web-Google.txt' #'twitter_combined.txt'#'textset.txt'
    NUMBER_OF_LINES_TO_READ = 400000
    MAX_SET_SIZE = 400
    # --- #
    graph = Graph()
    tao = EstimateTriangles()
    global_triangles = []
    t = 0


    with open(FILE_PATH, 'r') as f:
        while t < NUMBER_OF_LINES_TO_READ:
            line = f.readline()

            if line[0] == '#':
                # skip comments
                continue

            t += 1
            #print(line.split('  '))
            src_node, dest_node = line.split('\t') # line.split(" ")
            int_src, int_dest = int(src_node), int(dest_node)

            min_node = min(int_src, int_dest)
            max_node = max(int_src, int_dest)

            try:
                test_edge = graph.seen_edges[(min_node, max_node)]
                t -= 1

            except:
                triest_base(graph, t, MAX_SET_SIZE, min_node, max_node, tao)
                global_triangles.append(tao.estimate_global_traingles(t, MAX_SET_SIZE))

            if t % 1000 == 0:
                print("SEEN EDGES: ", len(graph.seen_edges.keys()))
                print("ALL EDGES: ", len(graph.all_edges))
                print("ADJ LIST: ", len(graph.adjacency_list))
                print("LEN ADJ")
                b = list(map(lambda x: x[0], graph.all_edges))
                b +=  list(map(lambda x: x[1], graph.all_edges))

                a = list(filter(lambda x: x if len(graph.adjacency_list[x]) >= 1 else None, graph.adjacency_list))
                print("A: ", len(a), "  B: ", len(b))
                print("INTERSECTION: ", len(list(set(a) & set(b))))

                print("T: ", t, "  TRIANGLES: ", global_triangles[-1], "     TAO: ", tao.tao, "\n")

    print(global_triangles[-1])
    print(tao.tao)

    #print(graph.all_edges)
    #print(graph.adjacency_list)

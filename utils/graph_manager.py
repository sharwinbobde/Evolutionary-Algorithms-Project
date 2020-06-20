import networkx as nx
import re
import matplotlib.pyplot as plt

class GraphManager:

    def __init__(self, filename:str, verbose=False):
        self.verbose = verbose

        self.read_file(filename)
        self.make_graph()

        bkv_filename = re.findall('^[\.]{0,1}.*(?=[\.])', filename)[0] + '.bkv'
        with open(bkv_filename, 'r') as f:
            self.bkv = list(map(int, f.readline().split()))[0]
        if self.verbose:
            print('bkv = ' + str(self.bkv))

    def read_file(self, filename:str):
        with open(filename, 'r') as f:
            self.n, self.e = map(int, f.readline().split())

            if self.verbose:
                print('|N| = ' + str(self.n))
                print('|E| = ' + str(self.e))
            
            edges = []
            for i, line in enumerate(f):
                if i == self.e:
                    break
                edge = list(map(int, line.split()))
                edges.append(edge)
            # print(edges)
            self.raw_edges = edges

    def make_graph(self):
        G = nx.Graph()

        G.add_nodes_from(range(1, self.n + 1))
        for edge in self.raw_edges:
            G.add_edge(edge[0], edge[1], weight=edge[2])
        self.G = G

    def draw_graph(self):
        fig = plt.figure(figsize=(14,8))
        # pos = nx.planar_layout(self.G)
        pos = nx.nx_agraph.graphviz_layout(self.G)
        nx.draw_networkx(self.G, pos)

        labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels)

    def evaluate_individual(self, individual):
        setA = []
        setB = []
        for i in range(0, len(individual)):
            if individual[i] == 0:
                setA.append(i + 1)
            else:
                setB.append(i + 1)
        return self.cut_cost(setA, setB)

    def cut_cost(self, set_A, set_B):
        return nx.cut_size(self.G, set_A, set_B, weight='weight')

    @staticmethod
    def get_graph_files():
        from collections import defaultdict
        '''
        param set: None = all, 'set0a', 'set0b', 'set0c', 'set0d'
        '''
        all_sets = ['set0a', 'set0b', 'set0c', 'set0d']
        V = {
            'set0a':[6, 12, 25, 50, 100],
            'set0b':[9, 16, 25, 49, 100],
            'set0c':[9, 16, 25, 49, 100], # has upto 1600 but didnt consider
            'set0d':[6, 12, 25, 50, 100], # has upto 200 but didnt consider
            'set0e':[6, 12, 25, 50, 100], # has upto 200 but didnt consider
        }
        instances = range(0,10)
        stem = '../data/maxcut/'

        out = defaultdict(dict)
        for s in all_sets:
            for n in V[s]:
                arr = []
                for i in instances:
                    arr.append(stem + s +'/n' + "{:07d}".format(n) + 'i' + "{:02d}".format(i) + 'txt')
                out[s][n] = arr

        return out




if __name__ == "__main__":
    GraphManager('maxcut/set0a/n0000006i00.txt', verbose=True)

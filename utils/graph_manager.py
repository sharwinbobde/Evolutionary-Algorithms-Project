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
                edge = list(map(int, f.readline().split()))
                if len(edge) == 3:
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

if __name__ == "__main__":
    GraphManager('maxcut/set0a/n0000006i00.txt', verbose=True)

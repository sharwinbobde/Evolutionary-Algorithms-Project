import numpy as np


class Particle:

    def __init__(self, graph_G, bkv:int):
        super().__init__()
        self.G = graph_G

        self.edges = self.G.edges(data=False)
        self.array_ = np.zeros((self(len(self.edges))))

    def compute_cost(self):
        ''' cost function '''
        
        pass
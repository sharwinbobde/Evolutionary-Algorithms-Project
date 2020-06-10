import numpy as np


class Particle:

    def __init__(self, graph_manager):
        super().__init__()
        self.gm = graph_manager
        self.bkv = graph_manager.bkv

        # dimentions are number of nodes
        # 0 1 indicates sets A and B
        self.nodes = self.gm.G.nodes(data=False)
        self.array = np.zeros(self.gm.G.number_of_nodes())

    def compute_cost(self, param):
        ''' cost function 
        aarry element being 0 indicates set_A and 1 indicates set_B
        '''
        set_A = []
        set_B = []

        for index, x in np.ndenumerate(param):
            if x == 0:
                set_A.append(index[0] + 1)
            else:
                set_B.append(index[0] + 1)
        cost = self.gm.cut_cost(set_A, set_B)

        # turn maximization to minimization
        return self.bkv - cost
    
import numpy as np

# ['cut', 'cond', 'norm_cut', 'e_exp', 'm_exp']

COSTS_PER_SET = {
            'set0a':['cut','norm_cut'],
            'set0b':['cut', 'e_exp'],
            'set0c':['cut','norm_cut'],
            'set0d':['cut','norm_cut'],
            'set0e':['cut','norm_cut'],
        }



class Particle:

    def __init__(self, graph_manager):
        super().__init__()
        self.gm = graph_manager
        self.bkv = graph_manager.bkv

        # dimentions are number of nodes
        # 0 1 indicates sets A and B
        self.nodes = self.gm.G.nodes(data=False)
        self.array = np.zeros(self.gm.G.number_of_nodes())

    def compute_cost(self, param, set_name='BlackBox'):
        ''' cost function 
        aarry element being 0 indicates set_A and 1 indicates set_B
        '''
        if self.causes_null_set(param):
            return 10e6

        set_A = []
        set_B = []

        for index, x in np.ndenumerate(param):
            if x == 0:
                set_A.append(index[0] + 1)
            else:
                set_B.append(index[0] + 1)
        cost = 0 


        if set_name == 'BlackBox':
            return self.cut_cost(set_A, set_B)


        if 'cut' in COSTS_PER_SET[set_name] :
            cost += self.cut_cost(set_A, set_B)

        if 'norm_cut' in COSTS_PER_SET[set_name]:
            cost += self.norm_cut(set_A, set_B)

        # if 'cond' in COSTS_PER_SET[set_name]:
        #     cost += self.cut_cost(set_A, set_B)

        if 'e_exp' in COSTS_PER_SET[set_name]:
            cost += self.e_exp(set_A, set_B)

        # if 'm_exp' in COSTS_PER_SET[set_name]:
        #     cost += self.cut_cost(set_A, set_B)

        return cost

    def causes_null_set(self, param):
        
        return np.all((param == 0)) or np.all((param == 1))

    
    def cut_cost(self, set_A, set_B):
        cost = self.gm.cut_cost(set_A, set_B)
        return self.bkv - cost # minimise to optimise
    
    def norm_cut(self, set_A, set_B):
        cost = self.gm.cut_normalised_cost(set_A, set_B)
        return - cost # minimise to optimise
    
    def e_exp(self, set_A, set_B):
        cost = self.gm.cut_edge_expansion(set_A, set_B)
        return - cost # minimise to optimise
    
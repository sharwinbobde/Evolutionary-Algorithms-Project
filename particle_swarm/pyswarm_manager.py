import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

class PySwarmManager:

    def __init__(self, max_iter = 1000):
        super().__init__()
        self.max_iter = max_iter
    
    def initialise_solver(self, particles=10, edges=10, options=None):
        if options == None:
            options = {'c1': 0.5, 'c2':0.3, 'w':0.9, 'k':3, 'p':2}
        self.optimizer =ps.binary.BinaryPSO(n_particles=particles, dimensions=edges, options=options)

    def make_function(self):
        self.fx = fx.sphere

    def optimise(self):
        cost, pos = self.optimizer.optimize(self.fx, self.max_iter)
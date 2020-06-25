import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from particle_swarm.particle import Particle
import numpy as np

class PySwarmManager:

    def __init__(self, particles, graph_manager, options=None, max_iter = 1000):
        super().__init__()
        self.max_iter = max_iter
        self.n_particles = particles
        self.gm = graph_manager

        self.optima_found = False

        self.particles_array = [Particle(graph_manager) for i in range(particles)]

        if options == None:
            options = {'c1': 0.5, 'c2':0.3, 'w':0.9, 'k':5, 'p':2}

        # dont let neighbours be an issue
        options['k'] = np.min([particles, options['k']])

        # bounds = (0.0, self.gm.G.size(weight='weight')) # unused :( no bounds for binary po
        # print(bounds)

        # dimensions is same as no of nodes
        self.optimizer =ps.binary.BinaryPSO(n_particles=particles,
                                            dimensions=graph_manager.G.number_of_nodes(),
                                            options=options)



    def objective_function(self, params):
        ''' used internally '''
        losses = []
        for i in range(self.n_particles):
            losses.append(self.particles_array[i].compute_cost(params[i]))
        return np.array(losses)

    def optimise(self, verbose=False):

        cost, pos = self.optimizer.optimize(self.objective_function,
                                            iters=self.max_iter, 
                                            verbose=verbose,
                                            optima = 0.0)

        if cost == 0.0:
            self.optima_found=True

        self.elite_cost = cost
        self.elite_param = pos

    def get_elite(self):
        return self.elite_param

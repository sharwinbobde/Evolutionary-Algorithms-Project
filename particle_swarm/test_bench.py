from particle_swarm.pyswarm_manager import PySwarmManager
import numpy as np
from collections import defaultdict

from tqdm import tqdm

import logging
import threading
import time
from threading import Thread
from multiprocessing import Process, Queue, Lock
import math
from copy import deepcopy
import pickle

lock = Lock()

class TestBench:

    def __init__(self, ):
        super().__init__()

    @staticmethod
    def single_run(particles, graph_manager, options=None, max_iter=1000, verbose=False):
        result = {} 
        solver = PySwarmManager(particles, graph_manager=graph_manager, options=options, max_iter=max_iter)
        solver.optimise(verbose)        
        # print(solver)


        cost_per_iteration = np.array(solver.optimizer.cost_history)

        result['elite'] = solver.elite_param
        result['elite_cost'] = solver.elite_cost
        result['optima_found'] = solver.optima_found
        result['cost_per_iteration'] = solver.optimizer.cost_history
        if solver.optima_found:
            result['elite at iteration'] = np.where(cost_per_iteration == 0.0)[0][0]
        # print(result)
        return result
        

    @staticmethod
    def ten_runs(particles, graph_manager, options=None, max_iter=1000, strict_optima=False, verbose=False):
        '''
        strict_optima: if optima not found in a run then dont do other runs
        '''
        start = time.time()

        result = defaultdict(list)
        result['successful_runs'] = 0
        for i in tqdm(range(10)):
            res = TestBench.single_run(particles, graph_manager, options, max_iter, verbose)
            if res['optima_found']:
                result['successful_runs'] += 1
                result['elite at iteration'].append(res['elite at iteration'])
            else:
                # did not rach optima in a run
                if strict_optima:
                    print('strict_optima: NOT MET :(')
                    exit(0)
        # print(result)

        end = time.time()
        print('Time for 10 runs = ' +str(end-start))
        return result


    #################################### PARALLEL CODE #################################### 

    
    @staticmethod
    def single_run_parallel(particles, graph_manager, options, max_iter, verbose, i):            
        
        result = {}
        solver = PySwarmManager(particles, graph_manager=graph_manager, options=options, max_iter=max_iter)
        print(solver)
        solver.optimise(verbose)

        cost_per_iteration = np.array(solver.optimizer.cost_history)
        result['elite'] = solver.elite_param
        result['elite_cost'] = solver.elite_cost
        result['optima_found'] = solver.optima_found
        result['cost_per_iteration'] = solver.optimizer.cost_history
        if solver.optima_found:
            result['elite at iteration'] = np.where(cost_per_iteration == 0.0)[0][0]
        print('single_run_parallel: ' + str(i))
        # print(solver)
        print(result)
        
        print('$', end='')
        # dump pickle
        pickle.dump(result, open('temp/temp_res_process_' + str(i)+'.pkl', 'wb'))

        return

    @staticmethod
    def ten_runs_parallel(particles, graph_manager, options=None, max_iter=1000, strict_optima=False, verbose=False):
        '''
        strict_optima: if optima not found in a run then dont do other runs
        '''
        result = {}
        result['successful_runs'] = 0
        result['elite at iteration'] = []

        start = time.time()

        pickled = pickle.dumps(SingleJob().single_run_parallel)

        # unique instances for jobs and solvers :/
        jobs = [SingleJob() for i in range(10)]
        solvers = [PySwarmManager(particles, graph_manager=graph_manager, options=options, max_iter=max_iter) for i in range(10)]
        
        
        processes = []
        for i in range(10):
            p = Process(target=jobs[i].single_run_parallel, args=(solvers[i], verbose, i))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            p.close()
        
        # res_.close()
        
        for i in range(10):
            res = pickle.load(open('./temp/temp_res_process_' + str(i)+'.pkl', 'rb'))
            print('.', end='')
            if res['optima_found']:
                result['successful_runs'] += 1
                result['elite at iteration'].append(res['elite at iteration'])
            else:
                # did not rach optima in a run
                if strict_optima:
                    print('strict_optima: NOT MET :(')
                    exit(0)
        end = time.time()
        print('Time for 10 runs = ' +str(end-start))
        return result


class SingleJob:


    def single_run_parallel(self, solver, verbose, i):            
        result = {}
        # solver = deepcopy(obj)
        # print(solver)
        solver.optimise(verbose)

        cost_per_iteration = np.array(solver.optimizer.cost_history)
        result['elite'] = solver.elite_param
        result['elite_cost'] = solver.elite_cost
        result['optima_found'] = solver.optima_found
        result['cost_per_iteration'] = solver.optimizer.cost_history
        if solver.optima_found:
            result['elite at iteration'] = np.where(cost_per_iteration == 0.0)[0][0]
        # print('single_run_parallel: ' + str(i))
        # print(solver)
        # print(result)
        
        print('$', end='')
        # dump pickle
        pickle.dump(result, open('temp/temp_res_process_' + str(i)+'.pkl', 'wb'))

        return
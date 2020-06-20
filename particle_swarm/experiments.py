
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from importlib import import_module

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from utils.graph_manager import GraphManager
from utils.plotting import plot_errorbars
from particle_swarm.test_bench import TestBench

from pyswarms.discrete import BinaryPSO

import numpy as np
import pandas as pd


def test_plots():
    X = [[0, 1],[0.5, 1.5]]
    Y = [[2, 3],[3, 4]]
    yerr= [[0.1, 0.1], [0.1, 0.3]]
    xerr= [[0.1, 0.1], [0.1, 0.3]]
    labels=['gomea', 'eda']

    plot_errorbars(X, Y,xerr, yerr, labels, xlab='haha', savepath='test_fig.jpg')

def test_TestBench():
    gm = GraphManager('data/maxcut/set0a/n0000012i00.txt', verbose=True)
    out = TestBench.ten_runs(60, graph_manager=gm, max_iter=100)
    print(out)

def find_n_req():
    set_files = GraphManager.get_graph_files()
    # all_sets = ['set0a', 'set0b', 'set0c', 'set0d', 'set0e']
    all_sets = ['set0a']
    V = {
        # 'set0a':[6, 12, 25, 50, 100],
        'set0a':[100],
        'set0b':[9, 16, 25, 49, 100],
        'set0c':[9, 16, 25, 49, 100], # has upto 1600 but didnt consider
        'set0d':[6, 12, 25, 50, 100], # has upto 200 but didnt consider
        'set0e':[6, 12, 25, 50, 100], # has upto 200 but didnt consider
    }

    csv = pd.DataFrame(columns=['set_name', 'v', 'n_req_mean', 'n_req_max', 'n_req_min'])
    for set_name in all_sets:
        for v in V[set_name]:
            files = set_files[set_name][v]
            n_req_arr = []
            for file in files:
                gm = GraphManager(file)

                # find n_upper
                n_upper = 2
                while True: 
                    print('trying n_upper = '+str(n_upper))
                    res = TestBench.ten_runs(int(n_upper), gm, max_iter=2000, strict_optima=True)
                    if res['successful_runs'] == 10:
                        print('successful_runs = ' + res['successful_runs'])
                        break
                    else:
                        n_upper = int(n_upper * 2)
                        continue
                
                n_lower = n_upper/2

                # n_upper can be 2
                n_lower = np.max([n_lower, 2])

                n_req = n_lower
                while True: 
                    print('trying n_req = '+str(n_req))
                    res = TestBench.ten_runs(int(n_req), gm, max_iter=2000, strict_optima=True)
                    if res['successful_runs'] == 10:
                        break
                    else:
                        n_req = n_req * 1.1
                        continue
                n_req_arr.append(n_req)
            # print(n_req_arr)
            n_req_mean = np.mean(n_req_arr)
            # n_req_dev = (np.max(n_req_arr) - np.min(n_req_arr))/2
            csv = csv.append(pd.DataFrame([[set_name, v, n_req_mean, np.max(n_req_arr), np.min(n_req_arr)]], columns=['set_name', 'v', 'n_req_mean', 'n_req_max', 'n_req_min']))
            
    print(csv)
    csv.to_csv('particle_swarm_scalability_analysis.csv', index=False)
    pass

def vary_c1_c2():
    w = 0.9
    k = 3
    C1 = np.arange(1, 5, 0.25)
    C2 = np.arange(1, 5, 0.25)

    n = 64
    V = {
        'set0a':12,
        'set0b':16,
        'set0c':16, 
        'set0d':12, 
        'set0e':12, 
    }

    set_files = GraphManager.get_graph_files()
    for set_name in ['set0e']: #list(set_files.keys()):
        v = V[set_name]
        gm = GraphManager(set_files[set_name][v][0], verbose=True)
        sets = []
        X=[]
        Y=[]
        Z=[]

        for c1 in C1:
            for c2 in C2:
                print(set_name)
                options = {'c1': c1, 'c2':c2, 'w':w, 'k':k, 'p':2}
                print(options)
                res = TestBench.ten_runs_parallel(n, gm,
                                options=options,
                                max_iter=10000,
                                strict_optima=True,
                                verbose=False)
                # res = TestBench.single_run(n, gm, options=options,max_iter=1000, verbose=True)
                print('in experiments')
                # print(res)
                sets.append(set_name)
                X.append(c1)
                Y.append(c2)
                Z.append(np.mean(res['elite at iteration']))

                data = {
                    'c1': X,
                    'c2': Y,
                    'Z': Z,
                }
                df = pd.DataFrame(data)
                df.to_csv('results/c1_c2_variation_'+set_name+'.csv', index=False)

def plot_c1_c2():

    for set_name in ['set0e']: #['set0a', 'set0b', 'set0c', 'set0d', 'set0e']: 
        df = pd.read_csv('results/c1_c2_variation_'+set_name+'.csv')
        fig = plt.figure(figsize=(14,8))
        fig.suptitle('c1, c2 variation plot for ' + set_name)

        is_inlier = df['Z'] < 250
        df1 = df[is_inlier][['c1', 'c2', 'Z']]
        # df1 = df[['c1', 'c2', 'Z']]
        ax = fig.gca(projection='3d')
        surf = ax.plot_trisurf(df1['c1'], df1['c2'], df1['Z'],
                                linewidth=1, antialiased=True, cmap=plt.cm.coolwarm, alpha=0.8,)
        ax.set_zlabel('mean iteations for finding elite')
        ax.set_xlabel('c1')
        ax.set_ylabel('c2')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


def vary_k():
    K = range(1, 30)
    n = 64
    X=[]
    Y=[]
    yerr=[]

    set_files = GraphManager.get_graph_files()
    V = {
        'set0a':12,
        'set0b':16,
        'set0c':16, 
        'set0d':12, 
        'set0e':12, 
    }
    for set_name in list(set_files.keys()):
        for k in K:
            v = V[set_name]
            gm = GraphManager(set_files[set_name][v][0], verbose=True)
            options = {'c1': 2, 'c2':2, 'w':0.9, 'k':k, 'p':2}

            res = TestBench.ten_runs_parallel(n, gm,
                            options=options,
                            max_iter=1000,
                            strict_optima=True,
                            verbose=False)
            X.append(k)
            Y.append(np.mean(res['elite at iteration']))
            yerr.append(np.std(res['elite at iteration']))

            data = {
                'k': X,
                'Y': Y,
                'yerr': yerr
            }
            df = pd.DataFrame(data)
            df.to_csv('results/k_variation_'+set_name+'.csv', index=False)

def plot_k():
    set_name = 'set0a'
    df = pd.read_csv('results/k_variation_'+set_name+'.csv')




if __name__ == "__main__":

    # individual part
    # vary_c1_c2()
    plot_c1_c2()



    # group part
    # find_n_req()



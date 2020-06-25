
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

import math

from pyswarms.discrete import BinaryPSO
from collections import defaultdict

import numpy as np
import pandas as pd

optimal_options={'c1': 4.25, 'c2':4.0, 'w':0.96, 'k':6, 'p':2}


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

def n_req_analysis():
    set_files = GraphManager.get_graph_files()
    all_sets = ['set0a', 'set0b', 'set0c', 'set0d', 'set0e']
    # all_sets = ['set0b', 'set0c', 'set0d', 'set0e']

    # copied from graph_manager.py
    # V = {
    #     'set0a':[6, 12, 25, 50, 100],
    #     'set0b':[9, 16, 25, 49, 100],
    #     'set0c':[9, 16, 25, 49, 100], # has upto 1600 but didnt consider
    #     'set0d':[6, 12, 25, 50, 100], # has upto 200 but didnt consider
    #     'set0e':[6, 12, 25, 50, 100], # has upto 200 but didnt consider
    # }
    V = {
        'set0a':[6, 12, 25],
        'set0b':[9, 16, 25],
        'set0c':[9, 16, 25], # has upto 1600 but didnt consider
        'set0d':[6, 12, 25], # has upto 200 but didnt consider
        'set0e':[6, 12, 25], # has upto 200 but didnt consider
    }

    for set_name in all_sets:
        print('set_name = '+ set_name)
        v_ = []
        n_req_ = []
        for v in V[set_name]:
            v_.append(v)
            files = set_files[set_name][v]
            n_req_arr = []
            for file in files:
                gm = GraphManager(file)
            
                n_upper = 512 # placed here so that dont need to recompute from 2 for every new file
                n_lower = 2
                n_req = int(n_lower + (n_upper - n_lower)/2)

                # binary search
                while n_upper > n_lower and (n_upper-n_lower) > 1:
                    print('trying (n_lower, n_req, n_upper) = ('+str(n_lower)+ ', '+str(n_req)+ ', '+str(n_upper)+')')
                    res = TestBench.ten_runs_parallel(int(n_req), gm, max_iter=5000, # strict_optima=True,
                                            options=optimal_options)
                    if res['successful_runs'] == 10:
                        # print('successful_runs = ' + str(res['successful_runs']))
                        n_upper = n_req
                    else:
                        n_lower = n_req
                    n_req = int(n_lower + (n_upper - n_lower)/2)

                n_req_arr.append(n_req)

            print('for v = ' + str(v)+' \t n_req_arr = '+ str(n_req_arr))

            n_req_.append(n_req_arr)
            data = {
                'v': v_,
                'n_req_arr': n_req_
            }
            df = pd.DataFrame(data)
            df.to_csv('results/n_req_analysis_'+set_name+'.csv', index=False)

def plot_n_req():
    V = []
    Y = []
    Yerr = []
    labels = []
    points = []

    for set_name in ['set0a', 'set0b', 'set0c', 'set0d', 'set0e']: 
        df = pd.read_csv('results/n_req_analysis_'+set_name+'.csv')
        # print(df)
        df['n_req_arr'] = df['n_req_arr'].str.replace('[', '')
        df['n_req_arr'] = df['n_req_arr'].str.replace(']', '')
        df['n_req_arr'] = df['n_req_arr'].str.replace(' ', '')
        n_req_arr = list((df['n_req_arr'].str.split(',')))
        n_req_arr = np.array(n_req_arr).astype(int)
        print(n_req_arr)

        v = np.array(df['v'])
        n_req_mean = []
        n_req_std = []
        print(v)
        for i in range(v.shape[0]):
            n_req_mean.append(np.mean(n_req_arr[i, :]))
            n_req_std.append(np.std(n_req_arr[i, :]))

        V.append(v)
        Y.append(n_req_mean)
        Yerr.append(n_req_std)
        labels.append(set_name)

        p = defaultdict(list)
        for i in range(v.shape[0]):
            for j in range(n_req_arr[i].shape[0]):
                p['X'].append(v[i])
                p['Y'].append(n_req_arr[i,j])
        points.append(p)
        # k.append(df['k'])
        # Y.append(df['Y'])
        # yerr.append(df['yerr'])
        # labels.append(set_name)
        # plot_errorbars(X=[df['k']], Y=[df['Y']], yerr=[df['yerr']], 
        #                 ylab='iterations till optima', xlab='k',
        #                 ylim=(-1,50),
        #                 title='Set '+set_name)

    plot_errorbars(X= V, Y=Y,
                    yerr=Yerr,
                    ylim=(0,256),
                    # yscale='logit',
                    # xscale='log',
                    ylab='n_req', xlab='| V |',
                    labels=labels,
                    # points=points,
                    savepath='images/n_req_analysis.jpeg',
                    show_lims=True)


def vary_c1_c2():
    w = 0.9
    k = 3
    C1 = np.arange(1, 6, 0.25)
    C2 = np.arange(1, 6, 0.25)

    n = 64
    V = {
        'set0a':12,
        'set0b':16,
        'set0c':16, 
        'set0d':12, 
        'set0e':12, 
    }

    set_files = GraphManager.get_graph_files()
    for set_name in list(set_files.keys()):
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

    for set_name in ['set0a', 'set0b', 'set0c', 'set0d', 'set0e']: 
        df = pd.read_csv('results/c1_c2_variation_'+set_name+'.csv')
        fig = plt.figure(figsize=(14,8))
        fig.suptitle('c1, c2 variation plot for ' + set_name)

        # is_inlier = df['Z'] < 250 # uncomment for removing outliers for set0e :/
        # df1 = df[is_inlier][['c1', 'c2', 'Z']]

        df1 = df[['c1', 'c2', 'Z']] # comment for removing outliers for set0e :/

        ax = fig.gca(projection='3d')
        surf = ax.plot_trisurf(df1['c1'], df1['c2'], df1['Z'],
                                linewidth=1, antialiased=True, cmap=plt.cm.coolwarm, alpha=0.8,)
        ax.set_zlabel('mean iteations for finding elite')
        ax.set_xlabel('c1')
        ax.set_ylabel('c2')
        # rotate for convenience
        ax.view_init(elev=40, azim=45)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig('images/c1_c2_'+set_name+'.jpeg')
        plt.show()


def vary_k():
    K = range(1, 12)
    n = 64

    set_files = GraphManager.get_graph_files()
    V = {
        'set0a':12,
        'set0b':16,
        'set0c':16, 
        'set0d':12, 
        'set0e':12, 
    }
    for set_name in list(set_files.keys()):
        X=[]
        Y=[]
        yerr=[]
        for k in K:
            v = V[set_name]
            gm = GraphManager(set_files[set_name][v][0], verbose=False)
            options = {'c1': 4, 'c2':4, 'w':0.9, 'k':k, 'p':2}

            res = TestBench.ten_runs_parallel(n, gm,
                            options=options,
                            max_iter=10000,
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
    k = []
    Y = []
    yerr = []
    labels = []

    for set_name in ['set0a', 'set0b', 'set0c', 'set0d', 'set0e']: 
        df = pd.read_csv('results/k_variation_'+set_name+'.csv')
        k.append(df['k'])
        Y.append(df['Y'])
        yerr.append(df['yerr'])
        labels.append(set_name)
        # plot_errorbars(X=[df['k']], Y=[df['Y']], yerr=[df['yerr']], 
        #                 ylab='iterations till optima', xlab='k',
        #                 ylim=(-1,50),
        #                 title='Set '+set_name)

    plot_errorbars(X= k, Y=Y, yerr=yerr,
                    ylim=(-1,50),
                    ylab='iterations till optima', xlab='k',
                    labels=labels,
                    savepath='images/particle_swarm_k_variation.jpeg')



def vary_w():
    W = np.arange(0.8, 1.02, 0.005)
    n = 64

    set_files = GraphManager.get_graph_files()
    V = {
        'set0a':12,
        'set0b':16,
        'set0c':16, 
        'set0d':12, 
        'set0e':12, 
    }
    for set_name in list(set_files.keys()):
        X=[]
        Y=[]
        yerr=[]
        for w in W:
            v = V[set_name]
            gm = GraphManager(set_files[set_name][v][0], verbose=False)
            options = {'c1': 4, 'c2':4, 'w':w, 'k':3, 'p':2}
            print(set_name+', w = ' + str(w))
            res = TestBench.ten_runs_parallel(n, gm,
                            options=options,
                            max_iter=10000,
                            strict_optima=True,
                            verbose=False)
            X.append(w)
            Y.append(np.mean(res['elite at iteration']))
            yerr.append(np.std(res['elite at iteration']))

            data = {
                'w': X,
                'Y': Y,
                'yerr': yerr
            }
            df = pd.DataFrame(data)
            df.to_csv('results/w_variation_'+set_name+'.csv', index=False)

def plot_w():
    w = []
    Y = []
    yerr = []
    labels = []

    for set_name in ['set0a', 'set0b', 'set0c', 'set0d', 'set0e']: 
        df = pd.read_csv('results/w_variation_'+set_name+'.csv')
        w.append(df['w'])
        Y.append(df['Y'])
        yerr.append(df['yerr'])
        labels.append(set_name)

    plot_errorbars(X= w, Y=Y, yerr=yerr,
                    ylim=(-1,50),
                    ylab='iterations till optima (log)', xlab='w',
                    labels=labels,
                    savepath='images/particle_swarm_w_variation.jpeg')

def black_Box_evaluations():
    n_req_arr = {
        'set0a': {
            6: 2,
            12: 22,
            25: 82
        },
        'set0b':{
            9: 5,
            16: 30,
            25: 40
        },
        'set0c':{
            9: 15,
            16: 20,
            25: 256
        },
        'set0d':{
            6: 2,
            12: 10,
            25: 65,
        }, 
        'set0e':{
            6: 7,
            12: 30,
            25: 260,
        },  
    }
    files = GraphManager.get_graph_files()
    all_sets = ['set0a', 'set0b', 'set0c', 'set0d', 'set0e']
    # all_sets = ['set0c', 'set0d', 'set0e']
    for set_name in all_sets:
        print(set_name)
        V = []
        Evals_mean, Evals_std = [], []
        Runtimes_mean, Runtimes_std = [], []
        Gens_mean, Gens_std = [], []
        for v in list(n_req_arr[set_name].keys()):
            print('v = ' + str(v))
            V.append(v)
            evals = []
            runtimes = []
            gens = []

            for file in files[set_name][v]:
                gm = GraphManager(file)
                n_req = n_req_arr[set_name][v] 
                res = TestBench.ten_runs_parallel(n_req, gm,
                            max_iter=10000,
                            strict_optima=True,
                            # verbose=True,
                            options=optimal_options)
                evals.append(res['num_evals'])
                runtimes.append(res['runtime'])
                gens.append(res['elite at iteration'])

            Evals_mean.append(np.mean(evals))
            Evals_std.append(np.std(evals))

            Runtimes_mean.append(np.mean(runtimes))
            Runtimes_std.append(np.std(runtimes))
            
            Gens_mean.append(np.mean(gens))
            Gens_std.append(np.std(gens))

            pd.DataFrame({
                    'v': V,
                    'num_eval_mean': Evals_mean,
                    'num_eval_std': Evals_std,
                }).to_csv('results/particle_swarm-B-num_eval-'+set_name+'.csv', index=False)

            pd.DataFrame({
                    'v': V,
                    'runtime_mean': Runtimes_mean,
                    'runtime_std': Runtimes_std,
                }).to_csv('results/particle_swarm-B-runtime-'+set_name+'.csv', index=False)

            pd.DataFrame({
                    'v': V,
                    'gen_mean': Gens_mean,
                    'gen_std': Gens_std,
                }).to_csv('results/particle_swarm-B-gen-'+set_name+'.csv', index=False)


if __name__ == "__main__":

    # Individual part

    # C1 C2 analysis
    # Uncomment the following line if you want to run the analysis again
    # vary_c1_c2()
    # plot_c1_c2()

    # Intertia (w) analysis
    # vary_w()
    # plot_w()

    # Neighbours (k) analysis
    # vary_k()
    # plot_k()

    # Black-box results
    # n_req_analysis()
    # plot_n_req()

    black_Box_evaluations()



    # Grey/White-Box

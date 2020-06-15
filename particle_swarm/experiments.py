
import networkx as nx
import matplotlib as plt
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from importlib import import_module

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from utils.graph_manager import GraphManager
from utils.plotting import plot_errorbars

def test_plots():
    X = [[0, 1],[0.5, 1.5]]
    Y = [[2, 3],[3, 4]]
    yerr= [[0.1, 0.1], [0.1, 0.3]]
    xerr= [[0.1, 0.1], [0.1, 0.3]]
    labels=['gomea', 'eda']

    plot_errorbars(X, Y,xerr, yerr, labels, xlab='haha', savepath='test_fig.jpg')



if __name__ == "__main__":
    
    # sys.path.append('..')
    # gm = GraphManager('data/maxcut/set0a/n0000012i00.txt', verbose=True)
    # G = gm.G
    # set_files = GraphManager.get_graph_files()
    # set_name = 'set0a'
    # set_files = set_files[set_name]
    # V = list(set_files.keys())
    # vs
    # print(set_files)

    test_plots()



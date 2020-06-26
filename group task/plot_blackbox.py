import matplotlib.pyplot as plt
import pandas as pd

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from utils.plotting import plot_errorbars

METRIC_PRETTY_NAME = {
    'num_eval': 'Number of Evaluations',
    'gen': 'Number of Generations/Iterations',
    'runtime': 'Runtime in sec'
}

Y_SCALE_PER_METRIC = {
    'num_eval': 'symlog',
    'gen': 'linear',
    'runtime': 'symlog'
}

if __name__ == "__main__":
    for set_name in ['set0a', 'set0b', 'set0c', 'set0d', 'set0e']:
        for metric in ['num_eval', 'runtime']:
            X = []
            Y = []
            Yerr = []
            labels = []
            for EA in ['particle_swarm', 'tabu_sGA', 'simple_ga', 'fast_ga']:
                filename = 'combined_results/'+EA+'-B-'+metric+'-'+set_name+'.csv'
                print(filename)
                df = pd.read_csv(filename,sep=',')
                # print(df)
                X.append(df['v'])
                Y.append(df[metric + '_mean'])
                Yerr.append(df[metric + '_std'])
                labels.append(EA)
        
            # plot here: separate graph for each graph structure and metric
            plot_errorbars(X=X, Y=Y, yerr=Yerr, labels=labels,
                            xlab='number of vertices',
                            ylab=metric + '(scale: '+Y_SCALE_PER_METRIC[metric]+')',
                            yscale=Y_SCALE_PER_METRIC[metric],
                            y_start_at_0=True,
                            show_lims=True,
                           title= 'BlackBox comparison for '+METRIC_PRETTY_NAME[metric] + ' for ' + set_name,
                           savepath='images/BlackBox/BlackBox-'+metric+'-'+set_name+'.jpeg')
            # exit(0)
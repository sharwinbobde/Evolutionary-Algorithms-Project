import matplotlib.pyplot as plt
import pandas as pd

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from utils.plotting import plot_improvement_errorbars

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
            for EA in ['particle_swarm', 'tabu_sGA', 'simple_ga', 'fast_ga']:
                X_W = []
                Y_W = []
                Yerr_W = []
                labels_W = []

                X_B = []
                Y_B = []
                Yerr_B = []
                labels_B = []

                # white box
                filename = 'combined_results/'+EA+'-W-'+metric+'-'+set_name+'.csv'
                print(filename)
                df = pd.read_csv(filename,sep=',')
                # print(df)
                X_W.append(df['v'])
                Y_W.append(df[metric + '_mean'])
                Yerr_W.append(df[metric + '_std'])
                labels_W.append(EA+' WhiteBox')

                # black box
                filename = 'combined_results/'+EA+'-B-'+metric+'-'+set_name+'.csv'
                print(filename)
                df = pd.read_csv(filename,sep=',')
                # print(df)
                X_B.append(df['v'])
                Y_B.append(df[metric + '_mean'])
                Yerr_B.append(df[metric + '_std'])
                labels_B.append(EA+' BlackBox')
        
                # plot here
                plot_improvement_errorbars(
                                X_W=X_W, Y_W=Y_W, yerr_W=Yerr_W, labels_W=labels_W,
                                X_B=X_B, Y_B=Y_B, yerr_B=Yerr_B, labels_B=labels_B,
                                xlab='number of vertices',
                                ylab=metric + '(scale: '+Y_SCALE_PER_METRIC[metric]+')',
                                yscale=Y_SCALE_PER_METRIC[metric],
                                y_start_at_0=True,
                                title= 'WhiteBox vs BlackBox '+EA+' '+METRIC_PRETTY_NAME[metric] + ' for ' + set_name,
                                savepath='images/Improvements/Improvement-'+EA+'-'+metric+'-'+set_name+'.jpeg')
                # exit(0)
import matplotlib.pyplot as plt

def plot_errorbars(X, Y, xerr=None, yerr=None, labels:list= None,
                   xlab=None, ylab=None, title:str='',
                   xscale='linear', yscale='linear',
                   savepath=None, xlim=None, ylim=None,
                   points=None,
                   show_lims=False):
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(title)
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        if yerr:
            ye = yerr[i]
        else:
            ye = None

        if xerr:
            xe = xerr[i]
        else:
            xe = None
        if labels:
            label = labels[i]
        else:
            label = None
        
        if show_lims:
            plt.errorbar(x, y, yerr=ye, xerr=xe, label=label, alpha=0.5, fmt='o-', lolims=True, uplims=True)
        else:
            plt.errorbar(x, y, yerr=ye, xerr=xe, label=label, alpha=0.5, fmt='o-')

        if points:
            g = plt.scatter(points[i]['X'], points[i]['X'], label=label )
            g.set_edgecolor('r')
    plt.legend()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.xscale(xscale)
    plt.yscale(yscale)
    if savepath:
        plt.savefig(savepath, dpi=120)
    
    plt.show()

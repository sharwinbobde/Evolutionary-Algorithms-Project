import matplotlib.pyplot as plt

def plot_errorbars(X, Y, xerr=None, yerr=None, labels:list= None, xlab=None, ylab=None, title:str='', savepath=None):
    fig = plt.figure(figsize=(14, 7))

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
        plt.errorbar(x, y, yerr=ye, xerr=xe, label=label, alpha=0.6, fmt='o-')
    plt.legend()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if savepath:
        plt.savefig(savepath, dpi=120)
    else:
        plt.show()

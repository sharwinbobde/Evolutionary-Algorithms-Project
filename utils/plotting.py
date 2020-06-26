import matplotlib.pyplot as plt

def plot_errorbars(X, Y, xerr=None, yerr=None, labels:list= None,
                   xlab=None, ylab=None, title:str='',
                   xscale='linear', yscale='linear',
                   savepath=None, xlim=None, ylim=None,
                   points=None,
                   show_lims=False,
                   y_start_at_0 = False):
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(title)
    for i in range(len(Y)):
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

    if y_start_at_0:
        _, top = plt.ylim()
        plt.ylim((0, top))
    else:
        plt.ylim(ylim)

    plt.xlim(xlim)

    plt.xscale(xscale)
    plt.yscale(yscale)
    if savepath:
        plt.savefig(savepath, dpi=120)
    else:
        plt.show()


def plot_improvement_errorbars(
                   X_W, Y_W, X_B, Y_B,
                   yerr_W, labels_W:list,
                   yerr_B, labels_B:list,
                   xlab=None, ylab=None, title:str='',
                   xscale='linear', yscale='linear',
                   savepath=None, xlim=None, ylim=None,
                   points=None,
                   show_lims=False,
                   y_start_at_0 = False):
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(title)
    for i in range(len(Y_W)):
        x_w = X_W[i]
        y_w = Y_W[i]
        x_b = X_B[i]
        y_b = Y_B[i]
        
        p = plt.errorbar(x_w, y_w, yerr=yerr_W[i], label=labels_W[i],
                        alpha=0.5, fmt='o-', lolims=show_lims, uplims=show_lims)

        p = plt.errorbar(x_b, y_b, yerr=yerr_B[i], label=labels_B[i],
                        alpha=0.5, fmt='o--', lolims=show_lims, uplims=show_lims, color='k')

    plt.legend()
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    if y_start_at_0:
        _, top = plt.ylim()
        plt.ylim((0, top))
    else:
        plt.ylim(ylim)

    plt.xlim(xlim)

    plt.xscale(xscale)
    plt.yscale(yscale)
    if savepath:
        plt.savefig(savepath, dpi=120)
    else:
        plt.show()
import numpy as np
import matplotlib.pyplot as plt

def plot_mean_std_shaded(ax, x, mean, std, color, label, **args):
    ax.plot(x, mean, color=color, label=label, **args)
    ax.fill_between(x, mean-std, mean+std, color=color, alpha=0.2)
    
def plot_mean_std_cap(x, mean, std, color, label, **args):
    plt.errorbar(x, mean, yerr=std, fmt='o-', color=color, label=label, capsize=5, **args)
    
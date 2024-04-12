import numpy as np
import matplotlib.pyplot as plt

def plot_mean_std_shaded(x, mean, std, color, label):
    plt.plot(x, mean, color=color, label=label)
    plt.fill_between(x, mean-std, mean+std, color=color, alpha=0.2)
    
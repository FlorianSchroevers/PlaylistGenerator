""" visualization

Authors: Florian Schroevers

Implements a visualization to show progress while searching for a playlist,
or for debuggin purposes.

TODO: comment file

"""

# import time

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# from matplotlib import cm
import matplotlib.pyplot as plt
# from matplotlib.backend_bases import FigureCanvasBase
import pandas as pd


def plot_path(points, path, fitness, mode='none', keep=False):
    """ Live updating plot of a given path between given points """
    if isinstance(points, pd.DataFrame):
        points = points.values

    if mode == 'pca':
        pca = PCA(n_components=2)
        components = pca.fit_transform(points)
    elif mode == 'tsne':
        tsne = TSNE(n_components=2, perplexity=80)
        components = tsne.fit_transform(points)
    else:
        components = points

    plt.clf()

    for i in range(1, len(path)):
        vertex_1 = components[path[i - 1], 0], components[path[i - 1], 1]
        vertex_2 = components[path[i], 0], components[path[i], 1]

        plt.scatter(components[:, 0], components[:, 1], label=fitness)
        plt.legend()
        plt.plot(
            [vertex_1[0], vertex_2[0]],
            [vertex_1[1], vertex_2[1]],
            c='b'
        )

    if keep:
        plt.show()
    else:
        plt.draw()
    plt.pause(0.01)

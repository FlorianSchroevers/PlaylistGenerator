""" visualization

Authors: Florian Schroevers

Implements a visualization to show progress while searching for a playlist,
or for debuggin purposes.

TODO: comment file

"""

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.backend_bases import FigureCanvasBase
import pandas as pd
import time


def plot_path(x, path, fitness, mode='none', keep=False):
    if type(x) == pd.DataFrame:
        x = x.values

    if mode == 'pca':
        pca = PCA(n_components=2)
        components = pca.fit_transform(x)
    elif mode == 'tsne':
        tsne = TSNE(n_components=2, perplexity=80)
        components = tsne.fit_transform(x)
    else:
        components = x

    plt.clf()

    for i in range(1, len(path)):
        vertex_1 = components[path[i - 1], 0], components[path[i - 1], 1]
        vertex_2 = components[path[i], 0], components[path[i], 1]

        plt.scatter(components[:, 0], components[:, 1])
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

    


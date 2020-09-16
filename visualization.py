""" visualization

Authors: Florian Schroevers

Implements a visualization to show progress while searching for a playlist, 
or for debuggin purposes.

"""

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import cm
import pandas as pd

def scatter(x, dm, tm, mode='pca', dim=2):
    if mode == 'pca':
        pca = PCA(n_components=dim)
        components = pca.fit_transform(x)
    elif mode == 'tsne':
        tsne = TSNE(n_components=dim, perplexity=80)
        components = tsne.fit_transform(x)
    
    df = pd.DataFrame(
        data=components,
        columns=['component1', 'component2'] + (dim == 3) * ['component3']
    )

    if dim == 2:
        fig = px.scatter(df, x='component1', y='component2', 
                        size_max=5, size=[1]*df.shape[0],
                        color = dm[0], color_continuous_scale= 'YlOrRd')
    elif dim == 3:
        fig = px.scatter_3d(df, x='component1', y='component2', z='component3',
                        size_max=5, size=[1]*df.shape[0],
                        color = dm[0], color_continuous_scale= 'YlOrRd')

    fig.add_shape(
        type='line',
        x0=df.values[0, 0], y0=df.values[0, 1],
        x1=df.values[2, 0], y1=df.values[2, 1],
    )

    #Generate a color scale
    cmap = cm.get_cmap('YlOrRd')

    for i, final in enumerate(tm[0, 2]):

        if i == 0:
            continue

        color = cmap((final + 1)/2)[:-1]
        color = f'rgb{tuple([int(c * 255) for c in color])}'
        print(final)

        fig.add_shape(
            type='line',
            x0=df.values[2, 0], y0=df.values[2, 1],
            x1=df.values[i, 0], y1=df.values[i, 1],
            line=dict(color=color, width=2)
        )


    margin = go.layout.Margin(l=20, r=20, b=20, t=30)
    fig.update_layout(margin=margin)
    fig.show()

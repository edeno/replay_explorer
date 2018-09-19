import base64
from io import BytesIO

import bokeh.plotting as bplt
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import offsetbox
from scipy.spatial.distance import cdist

from lap import lapjv
from PIL import Image


def plot_grid(images, col_wrap=15, cmap='viridis',
              subplot_height=2, subplot_width=2):
    '''Plot the images on a grid.

    Parameters
    ----------
    images : ndarray, shape (n_images, width, height)
    col_wrap : int, optional
        Number of subplot columns
    cmap : str or `~matplotlib.colors.Colormap`, optional

    '''
    n_images = images.shape[0]
    vmax = np.quantile(images, 0.95)

    n_row = np.ceil(n_images / col_wrap).astype(int)
    figsize = (col_wrap * subplot_width, n_row * subplot_height)

    fig, axes = plt.subplots(n_row, col_wrap,
                             figsize=figsize,
                             sharex=True, sharey=True,
                             subplot_kw=dict(xticks=[], yticks=[]))

    for ax, image in zip(axes.flat, images):
        try:
            ax.pcolorfast(image.T, cmap=cmap, vmin=0.0, vmax=vmax)
        except IndexError:
            pass
        ax.axis('off')


def plot_components(model, images, labels=None, ax=None,
                    thumbnail_fraction=0.05, cmap='viridis', plot_images=True):
    '''Reduce the dimensionality of the images and plot the components in a
    scatter plot with examples of the images.

    Parameters
    ----------
    model : scikit-learn model instance
    images : ndarray, shape (n_images, width, height), optional
    labels : None or ndarray, shape (n_images,), optional
    ax : None or `.axes.Axes` object, optional
    thumbnail_fraction : float, optional
    cmap : str or `~matplotlib.colors.Colormap`, optional
    plot_images : bool, optional

    '''
    ax = ax or plt.gca()

    n_images = images.shape[0]
    projections = model.fit_transform(images.reshape((n_images, -1)))
    projections -= projections.min(axis=0)
    projections /= projections.max(axis=0)

    if (labels is not None) and np.issubdtype(labels.dtype, np.object_):
        for label_ind, label in enumerate(np.unique(labels)):
            is_label = np.isin(labels, label)
            ax.plot(projections[is_label, 0], projections[is_label, 1],
                    '.', label=label)
    else:
        ax.scatter(projections[:, 0], projections[:, 1], c=labels)

    if images is not None:
        vmin, vmax = 0.0, np.quantile(images, 0.95)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        min_dist_2 = (thumbnail_fraction *
                      max(projections.max(axis=0) -
                          projections.min(axis=0))) ** 2
        shown_image_positions = np.array([2 * projections.max(0)])

        for image_position, image in zip(projections, images):
            dist = np.sum((image_position - shown_image_positions) ** 2, 1)
            # don't show points that are too close
            if np.min(dist) >= min_dist_2:
                shown_image_positions = np.vstack(
                    [shown_image_positions, image_position])
                is_nonzero = np.sum(image, axis=1) > 0
                offset_image = offsetbox.OffsetImage(
                    image[is_nonzero].T, cmap=cmap, norm=norm)
                image_box = offsetbox.AnnotationBbox(
                    offset_image, image_position)
                ax.add_artist(image_box)


def plot_components_grid(model, images, labels=None, cmap='viridis',
                         subplot_width=2.0, subplot_height=2.0):
    '''Reduce the dimensionality and plot similar images next to each other on
    a grid.

    Parameters
    ----------
    model : scikit-learn model instance
    images : None or ndarray, shape (n_images, width, height), optional
    labels : None or ndarray, shape (n_images,), optional
    cmap : str or `~matplotlib.colors.Colormap`, optional
    subplot_width : float, optional
    subplot_height : float, optional

    '''
    BIG_NUMBER = 1E6

    n_images = images.shape[0]
    n_col = n_row = np.ceil(np.sqrt(n_images)).astype(np.int)

    projections = model.fit_transform(images.reshape((n_images, -1)))
    projections -= projections.min(axis=0)
    projections /= projections.max(axis=0)

    row_ind = column_ind = np.arange(n_col)
    grid = np.stack(np.meshgrid(row_ind, column_ind), axis=2).reshape(-1, 2)

    cost_matrix = cdist(grid, projections, "sqeuclidean").astype(np.float64)
    cost_matrix *= BIG_NUMBER / cost_matrix.max()
    _, _, col_assignments = lapjv(cost_matrix, extend_cost=True)
    grid_jv = grid[col_assignments]

    figsize = (n_col * subplot_width, n_row * subplot_width)
    vmin, vmax = 0.0, np.quantile(images, 0.95)

    fig, axes = plt.subplots(n_row, n_col, figsize=figsize,
                             sharex=True, sharey=True,
                             subplot_kw=dict(xticks=[], yticks=[]))

    for (row_ind, col_ind), images in zip(grid_jv, images):
        axes[row_ind, col_ind].pcolorfast(images, cmap=cmap,
                                          vmin=vmin, vmax=vmax)


def _serialize_image(image):
    rgb_img = ((1 - image) * 255).astype(np.uint8)
    img = Image.fromarray(rgb_img).convert('RGB')
    buff = BytesIO()
    img.save(buff, format='JPEG')
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def plot_components_interactive(model, images, labels=None, ax=None,
                                cmap='viridis'):
    '''Reduce the dimensionality of the images and plot the components in a
    scatter plot with examples of the images.

    Parameters
    ----------
    model : scikit-learn model instance
    images : ndarray, shape (n_images, width, height), optional
    labels : None or ndarray, shape (n_images,), optional
    ax : None or `.axes.Axes` object, optional
    thumbnail_fraction : float, optional
    cmap : str or `~matplotlib.colors.Colormap`, optional
    plot_images : bool, optional

    '''
    ax = ax or plt.gca()

    n_images = images.shape[0]
    projections = model.fit_transform(images.reshape((n_images, -1)))
    projections -= projections.min(axis=0)
    projections /= projections.max(axis=0)

    source = bplt.ColumnDataSource(data=dict(
        x=projections[:, 0],
        y=projections[:, 1],
        desc=labels,
        images=[_serialize_image(image) for image in images],
    ))

    TOOLTIPS = """
    <div>
        <div>
            <img
                src="data:image/jpeg;base64,@images" height="200" alt="@imgs" width="200"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
        </div>
    </div>
    """

    fig = bplt.figure(plot_width=800, plot_height=800, tooltips=TOOLTIPS,
                      title="Mouse over the dots")
    fig.circle('x', 'y', size=20, source=source)
    bplt.show(fig)


def load_replay_data(file_path, poster_density_name, replay_info_name):
    '''Load the replay data and do some preprocessing.

    Parameters
    ----------
    file_path : str
    posterior_density_name : str
    replay_info_name : str

    Returns
    -------
    posterior_density : xarray.DataArray
    replay_info : pandas.DataFrame

    '''
    posterior_density = (
        xr.open_mfdataset(file_path, group=poster_density_name)
        .sum(dim='state')
        .sel(time=slice(0, 0.100))
        .posterior_density)
    replay_info = (xr.open_mfdataset(file_path, group=replay_info_name)
                   .to_dataframe())

    return posterior_density, replay_info

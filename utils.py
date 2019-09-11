import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def get_submit_folder():
    """ Check submit details, remove warnings using tf logging warning and make directory

    Returns
    -------
    str
        path of the created directory
    """
    try:
        CONDOR_ID = os.environ['CONDOR_ID']
    except KeyError:
        sys.exit('Error: Run this script with "pygpu %file"')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    folder = 'train-CNN-%s' % CONDOR_ID  # folder for training results
    os.makedirs(folder)

    return folder


def load_data(type, data):
    """ Load toptagging data

    Parameters
    ----------
    type : str
        Which dataset should be loaded: 'images' or '4vectors'?
    data : str
        Which dataset should be loaded: 'test' or 'train'?

    Returns
    -------
    np.array, np.array
        tuple of images, labels
    """
    path = "/net/scratch/deeplearning/toptagging/"
    if type == "images":
        file = np.load(path + "images_" + data + ".npz")
    elif type == "4vectors":
        file = np.load(path + "4vectors_" + data + ".npz")
    else:
        raise AssertionError("type should be 'images or '4vectors'")
    if type == "4vectors":
        images = file["fourVectors"]
    else:
        images = file[type][..., np.newaxis]

    labels = file["labels"]

    return (images, labels)


def plot_average_images(images, labels, folder):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, ax in enumerate(axes):
        idx = np.where(labels == i)[0]
        norm = LogNorm(10**-4, images.max(), clip='True')
        im = ax.imshow(np.mean(images[idx], axis=(0, -1)), norm=norm)
        ax.set_xlabel('eta')
        ax.set_ylabel('phi')
    axes[0].set_title("qcd background")
    axes[1].set_title("top quark")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.savefig(folder+'/average_images.png')

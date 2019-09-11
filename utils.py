import os
import sys
import tensorflow as tf
import numpy as np


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
    tf.logging.set_verbosity(tf.logging.ERROR)

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
        file = np.load(path + "vectors_" + data + ".npz")
    else:
        raise AssertionError("type should be 'images or '4vectors'")

    images = file["images"][..., np.newaxis]
    labels = file["labels"]

    return (images, labels)

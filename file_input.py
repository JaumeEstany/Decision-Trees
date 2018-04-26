import numpy as np


def load_dataset(path, class_col):
    """
    :param path: path to the file we want to read
    :param class_col: index of the column with the classification of instances
    :return: the column with the classification of instances and all the attributes columns
    """

    data = np.genfromtxt(path, skip_header=0, delimiter=',', dtype=str)

    instance_class = data[:, class_col]
    instance_class = np.transpose(instance_class)
    attributes = np.delete(data, class_col, axis=1)

    return instance_class, attributes


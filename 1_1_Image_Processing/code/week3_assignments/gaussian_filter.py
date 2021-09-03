import numpy as np


def gaussian_2d(k_size=(3, 3), sigma=1.0):
    """ Gaussian filter generation; max value 1 """
    k_height, k_width = k_size
    # Make a row for width
    # (3,) [-1, 0, 1] **2 -> [1, 0, 1]
    dist_w = np.power((np.arange(k_width) - (k_width-1)/2), 2)
    # (1x3) [[1, 0, 1]]
    dist_w = dist_w.reshape(1, -1)
    # Make a column for hight
    # (3,) [-1, 0, 1] **2 -> [1, 0, 1]
    dist_h = np.power((np.arange(k_height) - (k_height-1)/2), 2)
    # (3x1) [[1],[0],[1]]
    dist_h = dist_h.reshape(-1, 1)
    # Make a kernel size matrix for width
    w_dist = [dist_w] * k_height
    w_dist = np.concatenate(w_dist, axis=0)    # (3x3)
    # Make a kernel size matrix for width
    h_dist = [dist_h] * k_width
    h_dist = np.concatenate(h_dist, axis=1)    # (3x3)

    # Make a Gaussian kernel
    kernel = np.exp(-(h_dist + w_dist) / (2 * sigma**2))
    return kernel

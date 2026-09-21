import numpy as np


def batch_euclidean_distance_np(points1, points2):
    """
    Efficiently calculate the Euclidean distances for all pairs of points between two sets of 3D points
    
    Parameters
    ----------
    points1 :
        Numpy array of shape (N, 3)
    points2 :
        Numpy array of shape (M, 3)

    Return
    ------
    distance_matrix :
        Distance matrix of shape (N, M), where distance_matrix[i,j] is the distance from points1[i] to points2[j]
    """

    norm1 = np.sum(points1**2, axis=1)  # (N,)
    norm2 = np.sum(points2**2, axis=1)  # (M,)

    norm_matrix = norm1[:, None] + norm2[None, :]

    dot_product = np.dot(points1, points2.T)

    distance_squared = norm_matrix - 2*dot_product

    distance_squared = np.maximum(distance_squared, 0.0)
    distance_matrix = np.sqrt(distance_squared)

    return distance_matrix


def filter_by_voxel(points: np.ndarray, resolution=0.1):
    """
    Filter points by voxel grid
    
    Parameters
    ----------
    points :
        Numpy array of shape (N, 3)
    resolution :
        Voxel grid resolution
    
    Return
    ------
    index
    """
    _, idx = np.unique(np.round(points/resolution), axis=0, return_index=True)
    return idx

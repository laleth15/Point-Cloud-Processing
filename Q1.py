from typing import Tuple
import numpy as np
import scipy 
import utils


import matplotlib.pyplot as plt


def q1_a(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Fit a least squares plane by taking the Eigen values and vectors
    of the sample covariance matrix

    Attributes
    ----------
    P : np.ndarray
        Nx3 matrix denoting points in 3D space

    Returns
    -------
    normal : np.ndarray
        array of shape (3,) denoting surface normal of the fitting plane
    center : np.ndarray
        array of shape (3,) denoting center of the points
    '''

    length = P.shape[0]
    
    sum_x = np.sum(P[:,0])
    sum_y = np.sum(P[:,1])
    sum_z = np.sum(P[:,2])

    #Finding the centroid of P
    centroid = [sum_x/length, sum_y/length, sum_z/length]
    centered_P = P - centroid
    
    #Finding Covariance matrix
    cov_mat = np.cov(centered_P.T)

    #Finding the smallest eigenvector
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
    normal = eigenvectors[:, np.argmin(eigenvalues)]

    center = np.array(centroid)
    normal = np.array(normal)

    return normal, center

def q1_c(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Fit a plane using RANSAC

    Attributes
    ----------
    P : np.ndarray
        Nx3 matrix denoting points in 3D space

    Returns
    -------
    normal : np.ndarray
        array of shape (3,) denoting surface normal of the fitting plane
    center : np.ndarray
        array of shape (3,) denoting center of the points
    '''
    
    ### Enter code below
    
    num_iter = 50
    d_threshold = 0.05

    min_inliners = 90
    most_inliners = 0
    best_normal = None
    best_center = None

    for i in range(num_iter):
        # Random sampling of points
        sample = P[np.random.choice(P.shape[0], 3, replace=False), :]

        normal,center = q1_a(sample)

        distances = np.abs(np.dot(P-center,normal))

        # Calculating the number of inliners
        inliners = distances < d_threshold
        num_inliners = np.count_nonzero(inliners)

        # Update the best plane
        if num_inliners > most_inliners:
            most_inliners = num_inliners
            best_normal, best_center = normal,center

            if num_inliners >= min_inliners:
                best_normal, best_center = q1_a(P[inliners])


    return best_normal, best_center





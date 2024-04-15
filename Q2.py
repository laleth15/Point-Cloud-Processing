from typing import Tuple
import numpy as np
import Q1
import random
import math

def find_inliners(P,center,radius, thres):
    inliners = []

    for p in P:
        d = np.linalg.norm(p-center)
        if d < radius + thres and d > radius - thres:
            inliners.append(True)
        else:
            inliners.append(False)

    return inliners


def q2(P: np.ndarray, N: np.ndarray) -> Tuple[np.ndarray, float]:
    '''
    Localize a sphere in the point cloud. Given a point cloud as
    input, this function should locate the position and radius
    of a sphere

    Attributes
    ----------
    P : np.ndarray
        Nx3 matrix denoting points in 3D space
    N : np.ndarray
        Nx3 matrix denoting normals of pointcloud

    Returns
    -------
    center : np.ndarray
        array of shape (3,) denoting sphere center
    radius : float
        scalar radius of sphere
    '''

    num_iter = 25
    r_threshold = 0.005

    min_inliners = 10000
    most_inliners = 0
    num_inliners = 0
    best_center = None
    best_radius = None

    # Randomly sampling a radius
    min_radius = 0.05
    max_radius = 0.11
    while num_inliners < min_inliners:
        sample_radius = round(random.uniform(min_radius, max_radius),2)
        
        for j in range(num_iter):

            sample_index = round(random.uniform(0, len(P)-1))
            sample_point, sample_normal = P[sample_index], N[sample_index]/np.linalg.norm(N[sample_index])
        
            projected_center = sample_point + sample_radius*sample_normal

            inliners = find_inliners(P,projected_center,sample_radius,r_threshold)
            num_inliners = np.count_nonzero(inliners)

            # Update the best center and radius
            if num_inliners > most_inliners:
                most_inliners = num_inliners
                best_center, best_radius = projected_center,sample_radius

            if num_inliners > min_inliners:
                break

    print(best_center, best_radius) #[0.43731631 0.3472966  0.21023389] 0.09
    return best_center, best_radius


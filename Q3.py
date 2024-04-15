from typing import Tuple
import numpy as np
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

def q3(P: np.ndarray, N: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    Localize a cylinder in the point cloud. Given a point cloud as
    input, this function should locate the position, orientation,
    and radius of the cylinder

    Attributes
    ----------
    P : np.ndarray
        Nx3 matrix denoting 100 points in 3D space
    N : np.ndarray
        Nx3 matrix denoting normals of pointcloud

    Returns
    -------
    center : np.ndarray
        array of shape (3,) denoting cylinder center
    axis : np.ndarray
        array of shape (3,) pointing along cylinder axis
    radius : float
        scalar radius of cylinder
    '''

    num_iter = 50
    r_threshold = 0.005

    min_inliners = 12000
    most_inliners = 0
    num_inliners = 0
    best_center = None
    best_axis = None
    best_radius = None

    # Randomly sampling a radius
    min_radius = 0.05
    max_radius = 0.1

    while num_inliners < min_inliners:
        sample_radius = round(random.uniform(min_radius, max_radius),2)

        for j in range(num_iter):

            i1 = round(random.uniform(0, len(P)-1))
            i2 = round(random.uniform(0, len(P)-1))

            if i1 != i2:

                p1, n1 = P[i1], N[i1]/np.linalg.norm(N[i1])
                p2, n2 = P[i2], N[i2]/np.linalg.norm(N[i2])

                sample_axis = np.cross(n1,n2)
                sample_axis = sample_axis/np.linalg.norm(sample_axis)
                estimated_center = p1 + sample_radius*n1

                projection_mat = np.eye(3) - np.outer(sample_axis,sample_axis)

                projected_center = estimated_center.dot(projection_mat.T)
                projected_points = P.dot(projection_mat.T)

                inliners = find_inliners(projected_points,projected_center,sample_radius,r_threshold)
                num_inliners = np.count_nonzero(inliners)
                # Update the best center and radius
                if num_inliners > most_inliners:
                    most_inliners = num_inliners
                    best_center, best_radius, best_axis = projected_center, sample_radius, sample_axis

                if num_inliners > min_inliners:
                    break

    return best_center, best_axis, best_radius